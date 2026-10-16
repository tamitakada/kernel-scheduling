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


class SchedWorker(Worker):

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
        sched_hooks.install(self.get_model())
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