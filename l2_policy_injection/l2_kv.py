"""
l2_kv.py -- vLLM worker extension: pin the KV entries of selected prompt tokens
in L2 using CUDA access-policy windows.

Loaded with LLM(..., worker_extension_cls="l2_kv.L2Extension"). The methods
below become callable on the worker via llm.collective_rpc("l2_setup", ...).

Written against vLLM V1 (GPUModelRunner + Attention layers + forward context).
Assumptions (see README at bottom of run_l2_kv.py):
  * enforce_eager=True (hooks must run every step; no CUDA graphs)
  * batch size 1 (one request in flight)
  * KV cache layout is [2, num_blocks, block_size, H, D] (FlashAttention)
    or [num_blocks, 2, block_size, H, D] (FlashInfer), token-contiguous.
"""
from collections import Counter

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); \
  TORCH_CHECK(e_ == cudaSuccess, #x " failed: ", cudaGetErrorString(e_)); } while (0)

// returns {max persisting carve-out, max window size}
std::vector<int64_t> l2_limits() {
  int dev; CK(cudaGetDevice(&dev));
  int p = 0, w = 0;
  CK(cudaDeviceGetAttribute(&p, cudaDevAttrMaxPersistingL2CacheSize, dev));
  CK(cudaDeviceGetAttribute(&w, cudaDevAttrMaxAccessPolicyWindowSize, dev));
  return {(int64_t)p, (int64_t)w};
}

int64_t l2_set_aside(int64_t bytes) {
  auto lim = l2_limits();
  bytes = std::min<int64_t>(bytes, lim[0]);
  CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)bytes));
  return bytes;
}

void l2_set_window(int64_t ptr, int64_t nbytes, double hit_ratio) {
  cudaStreamAttrValue a = {};
  a.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr);
  a.accessPolicyWindow.num_bytes = (size_t)nbytes;
  a.accessPolicyWindow.hitRatio  = (float)hit_ratio;
  a.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
  a.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
  CK(cudaStreamSetAttribute(c10::cuda::getCurrentCUDAStream().stream(),
                            cudaStreamAttributeAccessPolicyWindow, &a));
}

void l2_clear_window() {
  cudaStreamAttrValue a = {};   // num_bytes = 0 disables the window
  CK(cudaStreamSetAttribute(c10::cuda::getCurrentCUDAStream().stream(),
                            cudaStreamAttributeAccessPolicyWindow, &a));
}

void l2_reset() { CK(cudaCtxResetPersistingL2Cache()); }
"""

_CPP_SRC = """
std::vector<int64_t> l2_limits();
int64_t l2_set_aside(int64_t bytes);
void l2_set_window(int64_t ptr, int64_t nbytes, double hit_ratio);
void l2_clear_window();
void l2_reset();
"""

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="l2ctl_vllm",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["l2_limits", "l2_set_aside", "l2_set_window",
                       "l2_clear_window", "l2_reset"],
            with_cuda=True,
            verbose=False,
        )
    return _EXT


def _kv_token_tensor(kv, which, phys_block):
    """Return the [block_size, H, D] view of K (which=0) or V (which=1)."""
    if kv.shape[0] == 2:          # [2, num_blocks, bs, H, D]  (FlashAttention)
        return kv[which, phys_block]
    if kv.shape[1] == 2:          # [num_blocks, 2, bs, H, D]  (FlashInfer)
        return kv[phys_block, which]
    raise RuntimeError(f"Unrecognised KV cache layout: {tuple(kv.shape)}")


class L2Extension:
    # ------------------------------------------------------------------ setup
    def l2_setup(self, t0: int, t1: int, which: str = "k",
                 carve_bytes: int = 4 << 20, thrash_mb: int = 0):
        """Persist KV of prompt tokens [t0, t1) (K or V) on every attention layer.

        thrash_mb > 0 adds an eviction kernel after each attention layer so
        the benefit of persistence is visible in ncu.
        """
        ext = _load_ext()
        self.l2_teardown()  # idempotent

        max_persist, max_window = ext.l2_limits()
        granted = ext.l2_set_aside(carve_bytes)

        cc = getattr(self, "cache_config", None)
        st = dict(ext=ext, t0=t0, t1=t1, which=0 if which == "k" else 1,
                  block_size=getattr(cc, "block_size", None),
                  granted=granted, max_window=max_window, applied=False,
                  log=[], n_calls=0, handles=[],
                  thrash=(torch.empty(thrash_mb << 18, dtype=torch.float32,
                                      device="cuda") if thrash_mb else None))
        self._l2 = st

        model = (self.model_runner.get_model()
                 if hasattr(self.model_runner, "get_model")
                 else self.model_runner.model)

        n = 0
        for m in model.modules():
            # vLLM Attention layers carry both of these.
            if hasattr(m, "kv_cache") and hasattr(m, "layer_name"):
                st["handles"].append(m.register_forward_pre_hook(self._l2_pre))
                st["handles"].append(m.register_forward_hook(self._l2_post))
                n += 1
        if n == 0:
            raise RuntimeError("No vLLM Attention layers found to hook")
        return dict(hooked_layers=n, max_persist=max_persist,
                    max_window=max_window, granted_carveout=granted)

    # ------------------------------------------------------------------ hooks
    def _l2_pre(self, module, args):
        from vllm.forward_context import get_forward_context
        st = self._l2
        st["applied"] = False

        ctx = get_forward_context()
        md = ctx.attn_metadata
        if isinstance(md, (list, tuple)):      # dual-batch-overlap builds
            md = md[0]
        if isinstance(md, dict):               # V1: layer_name -> metadata
            md = md.get(module.layer_name)
        if md is None or not hasattr(md, "block_table"):
            return                              # profiling / dummy run
        bt = md.block_table                     # [num_reqs, max_blocks]
        if bt.shape[0] != 1:
            return                              # demo supports 1 request only

        kv = module.kv_cache
        if isinstance(kv, (list, tuple)):
            kv = kv[getattr(ctx, "virtual_engine", 0)]
        if kv.numel() == 0:
            return

        bs = st["block_size"] or kv.shape[2]
        if not st.get("printed"):
            st["printed"] = True
            print(f"[l2_kv] kv_cache shape={tuple(kv.shape)} "
                  f"stride={tuple(kv.stride())} dtype={kv.dtype} "
                  f"block_size={bs} layer={module.layer_name}", flush=True)
        t0, t1 = st["t0"], st["t1"]

        # Only one window per stream and physical blocks are not contiguous,
        # so pick the block holding most of the target tokens.
        counts = Counter(p // bs for p in range(t0, t1))
        blk_idx = max(counts, key=lambda b: (counts[b], -b))
        lo = max(t0, blk_idx * bs) - blk_idx * bs
        hi = min(t1, (blk_idx + 1) * bs) - blk_idx * bs

        phys = int(bt[0, blk_idx].item())       # sync; fine for a demo
        blk = _kv_token_tensor(kv, st["which"], phys)   # one K (or V) block
        kv_dim_stride = kv.stride(0) if kv.shape[0] == 2 else kv.stride(1)
        e = blk.element_size()
        tok0 = blk[0]
        if blk.shape[0] != bs or not tok0.is_contiguous():
            raise RuntimeError(
                f"Unsupported KV block: shape={tuple(blk.shape)} "
                f"stride={blk.stride()} bs={bs}")
        r = tok0.numel()        # elements of K (or V) for one token
        p = blk.stride(0)       # elements between consecutive tokens

        if p == r:
            # K (or V) of consecutive tokens is back to back: window = just
            # the selected half.
            ptr = blk.data_ptr() + lo * p * e
            nbytes = (hi - lo) * r * e
            mode = "kv-separate"
        elif p == 2 * r and kv_dim_stride == r:
            # K and V interleaved per token: [tok][K(r) V(r)]. One contiguous
            # span starting at K of token `lo` covers K *and* V of all
            # target tokens.
            k_blk = _kv_token_tensor(kv, 0, phys)
            ptr = k_blk.data_ptr() + lo * p * e
            nbytes = (hi - lo) * p * e
            mode = "kv-interleaved"
        else:
            raise RuntimeError(
                f"Unhandled KV layout: block shape={tuple(blk.shape)} "
                f"stride={blk.stride()} kv_dim_stride={kv_dim_stride}")
        if nbytes > st["max_window"]:
            raise RuntimeError("window exceeds cudaDevAttrMaxAccessPolicyWindowSize")

        hit = min(1.0, st["granted"] / nbytes)
        st["ext"].l2_set_window(ptr, nbytes, hit)
        st["applied"] = True

        st["n_calls"] += 1
        if len(st["log"]) < 4:
            st["log"].append(dict(layer=module.layer_name, phys_block=phys,
                                  mode=mode,
                                  tokens=(blk_idx * bs + lo, blk_idx * bs + hi),
                                  ptr=hex(ptr), nbytes=nbytes, hit_ratio=hit))

    def _l2_post(self, module, args, output):
        st = self._l2
        if st["applied"]:
            st["ext"].l2_clear_window()         # lines stay persisting until reset
            st["applied"] = False
        if st["thrash"] is not None:
            st["thrash"].add_(1.0)              # streams MBs through L2

    # --------------------------------------------------------------- teardown
    def l2_teardown(self):
        st = getattr(self, "_l2", None)
        if st is None:
            return
        for h in st["handles"]:
            h.remove()
        st["ext"].l2_clear_window()
        st["ext"].l2_reset()
        st["ext"].l2_set_aside(0)
        self._l2 = None

    def l2_stats(self):
        st = getattr(self, "_l2", None)
        return None if st is None else dict(n_calls=st["n_calls"], log=st["log"])
