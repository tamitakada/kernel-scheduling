"""
vLLM worker for ONLINE serving (`vllm serve`) with the prefetch / kernel-scheduler infrastructure.

Why a worker class: the online server (AsyncLLM) always runs the engine in a separate process, so llm.apply_model(...)
from your script cannot reach the model. vLLM instead lets you name the worker class to use (--worker-cls). This one
is vllm's own GPU worker plus two things:

  load_model()  after the weights are loaded and final, register them with the agent and install the hooks
                (sched_hooks.install: the CUPTI-path tag hooks, or with SCHED_NO_CUPTI=1 the prefetch pre-hooks)
  shutdown()    flush the agent's timing rows, then let vLLM shut down (sched_hooks also registers exit hooks, in case
                this method is never reached)

Two classes. SchedWorker (the LLM; serve_online.sh) does both things above. SchedFlushWorker (Whisper; serve_whisper.sh)
only does the flush and the eager check: the tag and prefetch hooks cover decoder-only layouts, and Whisper has no
weights worth prefetching, so there its kernels are just intercepted and timed by the preloaded agent.

Usage (see serve_online.sh):
    PYTHONPATH=<dir with sched_hooks.py and sched_worker.py> SCHED_AGENT_SO=<abs path>/agent.so \
    vllm serve <model> --worker-cls sched_worker.SchedWorker --enforce-eager

Requires --enforce-eager: CUDA-graph replay neither runs Python hooks nor goes through the CUPTI launch callback, so the
scheduler would only ever see the warm-up launches. This class refuses to start without it (SCHED_ALLOW_CUDAGRAPH=1
overrides, for experiments where you know what you are doing).
"""
import os
import sys

import sched_hooks

try:
    from vllm.v1.worker.gpu_worker import Worker
except ImportError as e:  # pragma: no cover - depends on the installed vLLM
    raise ImportError(
        "sched_worker needs vLLM's GPU worker at vllm.v1.worker.gpu_worker.Worker; this vLLM does not have it. "
        "Subclass the worker your version uses and keep the two overrides below.") from e


sched_hooks._say("sched_worker imported: vLLM is using the scheduler worker class")   # proves --worker-cls took effect


class SchedFlushWorker(Worker):
    """Interception comes from the preloaded agent; this only enforces eager mode and writes the timing log."""

    def _sched_install(self, model):
        sched_hooks.install_flush(model, "kernel interception only (no weights registered, no tag or prefetch hooks)")

    def _sched_eager(self):
        try:
            return bool(self.vllm_config.model_config.enforce_eager)
        except AttributeError:
            return None          # cannot tell; do not block startup

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)                 # weights are loaded and processed once this returns
        if getattr(self, "_sched_installed", False):        # load_model can run again (e.g. sleep/wake); hook only once
            return
        if self._sched_eager() is False and not sched_hooks._env_flag("SCHED_ALLOW_CUDAGRAPH"):
            raise RuntimeError(
                "sched_worker: run vLLM with --enforce-eager. CUDA-graph replay bypasses the Python hooks and the CUPTI "
                "launch callback, so prefetching and scheduling would silently not apply. "
                "(SCHED_ALLOW_CUDAGRAPH=1 overrides this check.)")
        self._sched_install(self.get_model())
        self._sched_installed = True

    def sched_flush(self):
        """Write the agent's timing rows now. Reachable through collective_rpc("sched_flush") on the engine client."""
        sched_hooks.flush_timing()
        return os.getpid()

    def shutdown(self):
        if getattr(self, "_sched_installed", False):
            try:                                            # flush BEFORE vLLM releases the GPU resources
                sched_hooks.shutdown("worker shutdown")
            except Exception as e:                          # a logging problem must not block shutdown
                print(f"[sched] shutdown flush failed: {e!r}", file=sys.stderr)
        super().shutdown()


class SchedWorker(SchedFlushWorker):
    """The LLM: weights registered with the agent, plus the tag hooks (or, with SCHED_NO_CUPTI=1, the prefetch hooks)."""

    def _sched_install(self, model):
        sched_hooks.install(model)
