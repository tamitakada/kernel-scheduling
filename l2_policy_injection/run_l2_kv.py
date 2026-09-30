"""
run_l2_kv.py -- pin the KV cache of the tokens "The capital of" in L2 while
vLLM answers "The capital of France is" through llm.chat().

    python run_l2_kv.py                 # with L2 policy
    python run_l2_kv.py --no-policy     # baseline

Profile (both variants, then compare):
    ncu --target-processes all --cache-control none \
        --metrics lts__t_sector_hit_rate.pct \
        --kernel-name regex:"flash|fwd|attn" \
        --launch-skip 100 --launch-count 50 \
        python run_l2_kv.py --thrash-mb 256

NOTE --cache-control none: by default ncu flushes L2 before every kernel,
which makes any L2-residency experiment meaningless.

README / things to adjust for your vLLM version
  1. Attention layer discovery: modules having `.kv_cache` and `.layer_name`.
  2. `get_forward_context().attn_metadata[layer_name].block_table`.
  3. KV layout [2, nb, bs, H, D] or [nb, 2, bs, H, D], token-contiguous.
  4. Only one window per stream, so K *or* V is pinned (--which), not both,
     and only tokens within one physical block (block_size tokens).
"""
import argparse
import os

# Run the engine in-process so the worker extension and hooks live in the
# same process as this script (simplest to debug; single GPU only).
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--no-policy", action="store_true")
ap.add_argument("--which", choices=["k", "v"], default="k")
ap.add_argument("--carveout-mb", type=int, default=4)
ap.add_argument("--thrash-mb", type=int, default=0,
                help="stream this many MB through L2 after every attention layer")
ap.add_argument("--max-tokens", type=int, default=32)
args = ap.parse_args()

PHRASE = "The capital of"
messages = [{"role": "user", "content": "The capital of France is"}]

llm = LLM(
    model=args.model,
    enforce_eager=True,              # hooks must run each step; no CUDA graphs
    enable_prefix_caching=False,     # keep block allocation simple
    max_model_len=2048,
    gpu_memory_utilization=0.5,
    worker_extension_cls="l2_kv.L2Extension",
)

# --- map the phrase to token positions in the *chat-templated* prompt -------
tok = llm.get_tokenizer()
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)

c0 = text.index(PHRASE)
c1 = c0 + len(PHRASE)
pos = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < c1 and b > c0]
t0, t1 = pos[0], pos[-1] + 1
toks = tok.convert_ids_to_tokens(enc["input_ids"][t0:t1])
print(f"prompt has {len(enc['input_ids'])} tokens; target tokens [{t0}, {t1}) = {toks}")

# --- install hooks + carve-out inside the worker ----------------------------
if not args.no_policy:
    info = llm.collective_rpc(
        "l2_setup",
        args=(t0, t1, args.which, args.carveout_mb << 20, args.thrash_mb),
    )
    print("worker setup:", info[0])

out = llm.chat(messages, SamplingParams(temperature=0, max_tokens=args.max_tokens))
print("OUTPUT:", out[0].outputs[0].text)

if not args.no_policy:
    stats = llm.collective_rpc("l2_stats")[0]
    print(f"windows applied: {stats['n_calls']}")
    for entry in stats["log"]:
        print("  ", entry)
    llm.collective_rpc("l2_teardown")
