"""
Scheduler / prefetch hooks, shared by the offline driver (run_vllm_exp.py) and the online vLLM worker (sched_worker.py).

This is the module-registration logic that used to live in run_vllm_exp.py, unchanged in behaviour, plus what a
long-running server needs: a periodic flush of the agent's timing rows (they are held in memory until flushed) and a
single shutdown path.

Two modes, chosen by SCHED_NO_CUPTI:
  default            CUPTI path. Tag hooks push/pop "layer_<i>/<module>"; the agent's launch callback reads the tag, the
                     SCHEDULER (daemon) decides per launch whether to prefetch, the agent prefetches that module's weights.
  SCHED_NO_CUPTI=1   Prefetch pre-hooks: every module with weights prefetches them itself right before it runs.

Environment: SCHED_AGENT_SO (path to agent.so, default ./agent.so), SCHED_NO_CUPTI,
             SCHED_FLUSH_INTERVAL_S (online only; default 0 = write the timing log ONCE, at shutdown. Set e.g. 300 to
             also write it periodically, which bounds memory in a very long run),
             SCHED_FLUSH_IDLE_S (online only; default 5: write the timing log once, this many seconds after the
             engine goes idle following activity; 0 disables),
             SCHED_FLUSH_TRIGGER (online only; default ./sched_flush_now: `touch` that file and the worker writes the
             timing log immediately, without stopping. Set to "" to disable).
Every important event is also appended to sched_trace.log (in SCHED_LOG_DIR, default the current directory), so what
happened is visible even when terminal output is lost.
"""
import atexit
import ctypes
import multiprocessing.util
import os
import sys
import threading
import time

import torch


# Everything this module calls in agent.so: (name, argument types, return type). These must match the agent's
# `extern "C"` functions exactly; load_agent() checks the names, and the test-suite compares them with agent.cpp.
AGENT_API = [
    ("register_weight_region", [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t], None),
    ("sched_push_tag", [ctypes.c_char_p], None),
    ("sched_pop_tag", [], None),
    ("sched_flush_timing", [], None),
    ("sched_prefetch", [ctypes.c_char_p, ctypes.c_longlong], ctypes.c_longlong),
    ("sched_prefetch_ranges", [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                               ctypes.c_int, ctypes.c_longlong], ctypes.c_longlong),
]


def load_agent(path=None):
    """Loads agent.so (the same instance if it was LD_PRELOADed) and declares the signatures we call."""
    path = path or os.environ.get("SCHED_AGENT_SO") or "./agent.so"
    lib = ctypes.CDLL(path)
    missing = [name for name, _, _ in AGENT_API if not hasattr(lib, name)]
    if missing:
        raise RuntimeError(f"{path} does not export {', '.join(missing)}: is agent.so built from the agent.cpp "
                           f"that matches these scripts?")
    for name, argtypes, restype in AGENT_API:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


agent = load_agent()


def _env_flag(name):
    v = os.environ.get(name, "")
    return v not in ("", "0")


def _find_layers(model) -> list:
    """The decoder layers: model.model.layers for Llama/Qwen-style text models, or, for a multimodal wrapper such as
    Qwen3-VL / Qwen3.5 (which holds a vision tower next to `language_model`), model.language_model.model.layers."""
    for path in ("model.layers", "language_model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if obj is not None:
            return obj
    raise RuntimeError("could not find the decoder layers: tried model.model.layers and "
                       "model.language_model.model.layers; print(model) and add the right path")


def _tag_module(agent, module, tag):
    """Wraps `module`'s forward in a push/pop of `tag` on the agent's tag stack."""
    tag_bytes = tag.encode("utf-8")

    def pre_hook(mod, args):
        agent.sched_push_tag(tag_bytes)

    def post_hook(mod, args, output):
        agent.sched_pop_tag()

    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)


def _param_range(p):
    if not p.is_cuda or p.numel() == 0:
        return None
    span = 1 + sum((s - 1) * st for s, st in zip(p.shape, p.stride()))
    return p.data_ptr(), span * p.element_size()


def register_model(model, should_tag_hooks=None):
    """Registers every decoder-layer parameter with the agent, then wires up the prefetch for the active mode:

      CUPTI path   (default): tag hooks. Each leaf module pushes/pops a "layer_<i>/<module>" tag; the agent's launch
                   callback reads it, the SCHEDULER decides per launch whether to prefetch, and the agent prefetches the
                   tensors registered for that module.
      SCHED_NO_CUPTI=1:       prefetch pre-hooks. Each module with weights prefetches them itself, right before it runs.

    should_tag_hooks=None picks the mode from SCHED_NO_CUPTI; pass True/False to force one. The two must not be mixed:
    the agent ignores the pre-hook prefetch while CUPTI is active.
    Call once, after the weights are on the GPU and in the process that launches the kernels (a second call on the same
    model is ignored, so a worker that loads its model twice cannot end up with doubled hooks).
    Returns (layers, tensors registered)."""

    if getattr(model, "_sched_registered", False):
        return 0, 0

    if should_tag_hooks is None:
        should_tag_hooks = not _env_flag("SCHED_NO_CUPTI")

    layers = _find_layers(model)
    n_layers = n_tensors = 0

    for i, layer in enumerate(layers):
        n_layers += 1
        for name, param in layer.named_parameters():
            tag = f"layer{i}_{name}".replace(".", "_").encode("utf-8")
            nbytes = param.numel() * param.element_size()
            agent.register_weight_region(
                tag,
                ctypes.c_void_p(param.data_ptr()),
                ctypes.c_size_t(nbytes))
            n_tensors += 1

        if should_tag_hooks:  # register prefetch tags before/after modules
            # tag the layer then any sublayers
            _tag_module(agent, layer, f"layer_{i}")

            for name, sub in layer.named_modules():
                if name and len(list(sub.children())) == 0:
                    _tag_module(agent, sub, f"layer_{i}/{name.replace('.', '_')}")

        else:
            for name, sub in layer.named_modules():
                ranges = [r for r in map(_param_range, sub.parameters(recurse=False)) if r]

                n = len(ranges)
                ptrs = (ctypes.c_uint64 * n)(*(a for a, _ in ranges))
                sizes = (ctypes.c_uint64 * n)(*(b for _, b in ranges))

                def _sched_prefetch_ranges_hook(m, args, _p=ptrs, _s=sizes, _n=n):
                    agent.sched_prefetch_ranges(_p, _s, _n, torch.cuda.current_stream().cuda_stream)

                sub.register_forward_pre_hook(_sched_prefetch_ranges_hook)

    model._sched_registered = True
    return n_layers, n_tensors


# ---------------------------------------------------------------------------------------------------------------------
# Long-running use (online serving). The agent holds timing rows in memory until sched_flush_timing(). By default we
# flush exactly once, at shutdown (sched_worker.SchedWorker.shutdown, reached on Ctrl+C / SIGTERM). A timer is opt-in
# (SCHED_FLUSH_INTERVAL_S > 0): sched_flush_timing appends to the TSV, writes the header once, and releases the rows it
# wrote, so repeated calls are safe, but each call also prints the agent's summary line, which is noise while idle.
# ---------------------------------------------------------------------------------------------------------------------
_flush_lock = threading.Lock()
_flush_stop = threading.Event()
_flush_thread = None
_trigger_stop = threading.Event()
_trigger_thread = None
_idle_stop = threading.Event()
_idle_thread = None
_activity = [0]                               # bumped once per forward pass (see _activity_hook)
_shutdown_lock = threading.Lock()
_shutdown_done = False
_exit_hooks_registered = False


def _say(msg):
    try:
        print(f"[sched] pid={os.getpid()}: {msg}", file=sys.stderr, flush=True)
    except Exception:                         # stderr can already be closed while the interpreter is exiting
        pass
    try:                                      # the same line in a plain file: it survives lost or scrolled terminal output
        path = os.path.join(os.environ.get("SCHED_LOG_DIR") or ".", "sched_trace.log")
        with open(path, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} pid={os.getpid()} {msg}\n")
    except Exception:
        pass


def _log_level():
    """SCHED_LOG, the one variable the daemon, the agent and this module share: full (default) | decisions | off. Only \"full\" makes the
    agent record CUDA-event timing, so only \"full\" has a timing log to write. Returns 2 (full), 1 (decisions) or 0 (off)."""
    v = (os.environ.get("SCHED_LOG") or "full").strip().lower()
    if v in ("off", "0", "none"):
        return 0
    if v in ("decisions", "1"):
        return 1
    return 2


def flush_timing():
    if _log_level() < 2:                      # nothing was recorded (see _log_level)
        return
    with _flush_lock:                         # one flush at a time (timer vs trigger vs shutdown)
        agent.sched_flush_timing()


def start_periodic_flush(interval_s):
    """Flush the timing rows every interval_s seconds on a daemon thread (the ctypes call releases the GIL, so the
    engine thread is not held up). interval_s <= 0 does nothing."""
    global _flush_thread
    if interval_s <= 0 or _flush_thread is not None:
        return

    def loop():
        while not _flush_stop.wait(interval_s):
            try:
                flush_timing()
            except Exception as e:            # never let a logging problem take the server down
                _say(f"periodic flush failed: {e!r}")

    _flush_stop.clear()
    _flush_thread = threading.Thread(target=loop, name="sched-flush", daemon=True)
    _flush_thread.start()


def stop_periodic_flush():
    global _flush_thread
    _flush_stop.set()
    if _flush_thread is not None:
        _flush_thread.join(timeout=5)
        _flush_thread = None


def _activity_hook(module, args):
    _activity[0] += 1                         # returns None: the module's inputs are untouched


def start_idle_flush(idle_s, poll_s=None):
    """Write the timing log once, idle_s seconds after the engine goes quiet following activity (activity = forward
    passes, counted by _activity_hook). Nothing is printed or written while the engine sits idle, and one write follows
    each burst of work, so you get the log without having to stop the server. idle_s <= 0 does nothing."""
    global _idle_thread
    if idle_s <= 0 or _idle_thread is not None:
        return
    poll = poll_s if poll_s is not None else min(1.0, max(0.05, idle_s / 5.0))

    def loop():
        last_seen, last_change, dirty = _activity[0], time.monotonic(), False
        while not _idle_stop.wait(poll):
            cur, now = _activity[0], time.monotonic()
            if cur != last_seen:
                last_seen, last_change, dirty = cur, now, True
            elif dirty and now - last_change >= idle_s:
                dirty = False
                try:
                    _say(f"engine idle for {idle_s:g}s: writing the timing log")
                    flush_timing()
                except Exception as e:
                    _say(f"idle flush failed: {e!r}")

    _idle_stop.clear()
    _idle_thread = threading.Thread(target=loop, name="sched-idle-flush", daemon=True)
    _idle_thread.start()


def stop_idle_flush():
    global _idle_thread
    _idle_stop.set()
    if _idle_thread is not None:
        _idle_thread.join(timeout=5)
        _idle_thread = None


def start_flush_trigger(path, poll_s=0.5):
    """On-demand flush that does not depend on how the server is stopped: when the file `path` appears it is removed
    and the timing log is written right away. Polled on a daemon thread (a stat every poll_s seconds)."""
    global _trigger_thread
    if not path or _trigger_thread is not None:
        return

    def loop():
        while not _trigger_stop.wait(poll_s):
            try:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        continue              # another process took it
                    _say(f"flush requested ({path})")
                    flush_timing()
            except Exception as e:
                _say(f"triggered flush failed: {e!r}")

    _trigger_stop.clear()
    _trigger_thread = threading.Thread(target=loop, name="sched-flush-trigger", daemon=True)
    _trigger_thread.start()


def stop_flush_trigger():
    global _trigger_thread
    _trigger_stop.set()
    if _trigger_thread is not None:
        _trigger_thread.join(timeout=5)
        _trigger_thread = None


def shutdown(reason="shutdown"):
    """Write the timing log, ONCE. It is called from several places (vLLM's worker shutdown, multiprocessing's exit
    finalizers, atexit): the first caller does the work and says why, later calls return at once. The agent has no
    exit-time flush of its own, so without this call the timing rows are lost."""
    global _shutdown_done
    if _log_level() < 2:                      # nothing to flush, nothing was started
        return
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True
    _say(f"writing the timing log ({reason})")
    stop_periodic_flush()
    stop_idle_flush()
    stop_flush_trigger()
    try:
        flush_timing()
    except Exception as e:
        _say(f"timing flush failed: {e!r}")


def _register_exit_hooks():
    """Make sure shutdown() runs however this process ends, short of being killed outright. vLLM's engine process is
    FORKED by default, and a forked multiprocessing child leaves through os._exit(), which skips atexit handlers;
    multiprocessing runs its own exit finalizers first, so register one there too. atexit covers spawned children and
    ordinary interpreters."""
    global _exit_hooks_registered
    if _exit_hooks_registered:
        return
    _exit_hooks_registered = True
    atexit.register(shutdown, "atexit")
    try:
        multiprocessing.util.Finalize(None, shutdown, args=("process exit",), exitpriority=100)
    except Exception as e:                    # an internal multiprocessing API: never let it break startup
        _say(f"could not register the multiprocessing exit finalizer: {e!r}")


def install_flush(activity_module, what):
    """The timing-log half of install(): the idle watcher (it counts forward passes of `activity_module`), the optional
    timer, the on-demand trigger file and the exit hooks. It registers no weights and hooks no layers. For a model the tag
    and prefetch hooks do not cover (Whisper), kernel interception comes from the preloaded agent alone, and this is all
    the worker has to add: without it the timing rows would never be written."""
    if _log_level() < 2:                      # no timing rows will exist: no idle watcher, timer, trigger file or exit flush, no counter hook
        _say(f"{what}; timing log OFF (SCHED_LOG={os.environ.get('SCHED_LOG')})")
        if _env_flag("SCHED_L2_PERSIST"):
            _say("WARNING: SCHED_L2_PERSIST is set but this agent has no L2 persistence support; ignored.")
        return
    if not getattr(activity_module, "_sched_activity", False):    # one cheap counter bump per forward pass
        activity_module.register_forward_pre_hook(_activity_hook)
        activity_module._sched_activity = True
    interval = float(os.environ.get("SCHED_FLUSH_INTERVAL_S", "0"))
    start_periodic_flush(interval)
    idle = float(os.environ.get("SCHED_FLUSH_IDLE_S", "5"))
    start_idle_flush(idle)
    trigger = os.environ.get("SCHED_FLUSH_TRIGGER", "sched_flush_now")
    trigger_path = os.path.abspath(trigger) if trigger not in ("", "0") else None
    start_flush_trigger(trigger_path)
    _register_exit_hooks()
    parts = []
    if idle > 0:
        parts.append(f"{idle:g}s after the engine goes idle")
    if interval > 0:
        parts.append(f"every {interval:g}s")
    parts.append("at shutdown")
    if trigger_path:
        parts.append(f"on demand (touch {trigger_path})")
    _say(f"{what}; timing log written: {', '.join(parts)}")
    if _env_flag("SCHED_L2_PERSIST"):       # a leftover export from an earlier experiment would otherwise be a silent no-op
        _say("WARNING: SCHED_L2_PERSIST is set but this agent has no L2 persistence support; ignored.")


def install(model):
    """Online entry point for the LLM: call once per worker process after the model is loaded."""
    n_layers, n_tensors = register_model(model)
    mode = "prefetch pre-hooks (SCHED_NO_CUPTI)" if _env_flag("SCHED_NO_CUPTI") else "tag hooks (CUPTI path, scheduler decides)"
    install_flush(_find_layers(model)[0], f"{mode}; {n_layers} layers, {n_tensors} tensors registered")