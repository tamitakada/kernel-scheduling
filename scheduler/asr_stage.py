"""
Speech-to-text stage for online_client.py: LibriSpeech utterances -> Whisper -> transcript text, which the client then
sends to the LLM as the request. Whisper runs in its OWN process, started by the client, on the same GPU as the vLLM
server.

Why a separate process (and not a thread in the client):
  * the client is an open-loop load generator: its timestamps (arrivals, TTFT, TPOT) must not be disturbed by Whisper
    competing for the Python GIL, so the client process does no model work at all;
  * only the Whisper process needs the scheduler's agent.so preloaded. It is attached exactly as serve_online.sh
    attaches vLLM (LD_PRELOAD, CUPTI mode), so Whisper's kernel launches go through the same scheduler_daemon and
    are serialized with the LLM's. The client process itself is NOT preloaded and is not a scheduled GPU process.

The worker also does the dataset work: it samples the utterances (so hundreds of MB of audio never travel through a pipe
to it) and loads the model, then reports ready. There is no warm-up: first-use costs land in the first transcriptions.

Protocol (multiprocessing queues, 'spawn' start method):
  job_q  : (job_id, sample_index) per transcription, None = stop          (client -> worker)
  res_q  : ("ready", wid, info) | ("fatal", wid, msg) | ("res", job_id, wid, text, service_s, err) | ("bye", wid)
"""
import asyncio
import ctypes
import glob
import io
import itertools
import multiprocessing as mp
import os
import queue
import random
import signal
import sys
import threading
import time

SAMPLE_RATE = 16000                                  # Whisper's input rate, and LibriSpeech's
DATASET, CONFIG, SPLIT = "openslr/librispeech_asr", "clean", "test"      # test-clean: 2620 utterances
STARTUP_TIMEOUT_S = 1800                             # dataset + model download and load; a dead worker is noticed much sooner


# ---------------------------------------------------------------- agent helpers (used in the worker, and by the client)
def resolve_agent(agent_so_arg, no_agent, script_dir):
    """Which agent.so to preload into the Whisper process, or None for 'unmanaged'. Mirrors serve_online.sh: no preload
    with --no-agent, or with SCHED_NO_CUPTI=1 (hook mode: no kernel interception anywhere). Fails early, with the
    reason, instead of letting the worker die or hang later."""
    if no_agent:
        return None
    if os.environ.get("SCHED_NO_CUPTI", "0") not in ("0", ""):
        print("[asr] SCHED_NO_CUPTI is set: Whisper runs WITHOUT the agent (no kernel interception), like the server.",
              file=sys.stderr)
        return None
    path = os.path.abspath(agent_so_arg or os.environ.get("SCHED_AGENT_SO") or os.path.join(script_dir, "agent.so"))
    if not os.path.isfile(path):
        raise SystemExit(f"agent.so not found at {path}: build it, pass --agent-so / set SCHED_AGENT_SO, "
                         f"or pass --no-agent to run Whisper unmanaged.")
    if not glob.glob("/dev/shm/kernel_sched_shm*"):
        raise SystemExit("no scheduler shared-memory segment in /dev/shm: start ./scheduler_daemon first. (The agent "
                         "exits at once if the segment does not exist, so the Whisper process could not start.)")
    return path


def agent_loaded(agent_so):
    """True if agent_so is mapped into THIS process (that is, LD_PRELOAD took effect)."""
    if not agent_so:
        return False
    try:
        real = os.path.realpath(agent_so)
        with open("/proc/self/maps") as f:
            return any(real in line for line in f)
    except OSError:
        return False


def flush_agent_timing(agent_so):
    """Write this process's kernel-timing rows (agent_<pid>_timing.tsv). The agent has no exit-time flush of its own.
    Only done when the agent is already mapped: CDLL on a file that is NOT loaded would load it late and run its
    constructor, which is the one thing that must never happen here."""
    if not agent_loaded(agent_so):
        return False
    try:
        lib = ctypes.CDLL(agent_so)                      # the same instance the preload loaded
        lib.sched_flush_timing.argtypes = []
        lib.sched_flush_timing.restype = None
        lib.sched_flush_timing()
        return True
    except Exception as e:
        print(f"[asr pid={os.getpid()}] could not flush the agent timing log: {e!r}", file=sys.stderr)
        return False


# ---------------------------------------------------------------- the inputs: LibriSpeech
def sample_librispeech(n, seed):
    """n utterances from LibriSpeech test-clean, chosen with `seed`: (ids, float32 mono 16 kHz arrays). Without replacement
    when n fits in the split, with replacement otherwise. The audio is read as raw FLAC bytes and decoded with soundfile,
    so this does not depend on which decoding backend the installed `datasets` version uses."""
    import soundfile as sf
    from datasets import Audio, load_dataset
    ds = load_dataset(DATASET, CONFIG, split=SPLIT).cast_column("audio", Audio(decode=False))
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n) if n <= len(ds) else [rng.randrange(len(ds)) for _ in range(n)]
    ids, arrays = [], []
    for ex in ds.select(idx):
        a = ex["audio"]
        x, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"], dtype="float32")   # bytes, or a path (older script-based datasets)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"expected {SAMPLE_RATE} Hz audio, got {sr} Hz")
        ids.append(ex["id"])
        arrays.append(x)
    return ids, arrays


# ---------------------------------------------------------------- the model
class WhisperEngine:
    """Hugging Face transformers Whisper on cuda, batch size 1, greedy decoding, English. Short-form: the processor
    truncates an utterance to Whisper's 30 s window (a few LibriSpeech utterances are slightly longer)."""

    def __init__(self, cfg):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch, self.device = torch, "cuda"
        self.dtype = getattr(torch, cfg["dtype"])
        name = cfg["model"]
        self.processor = WhisperProcessor.from_pretrained(name)
        try:
            model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=self.dtype)
        except TypeError:                                # newer transformers renamed torch_dtype -> dtype
            model = WhisperForConditionalGeneration.from_pretrained(name, dtype=self.dtype)
        self.model = model.to(self.device).eval()
        self.gen = dict(max_new_tokens=cfg["max_new_tokens"], do_sample=False, num_beams=1)
        if ".en" not in name.lower():                    # English-only checkpoints reject language/task
            self.gen.update(task="transcribe", language="en")

    def transcribe(self, audio):
        feats = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        feats = feats.to(self.device, dtype=self.dtype)
        with self.torch.inference_mode():
            ids = self.model.generate(feats, **self.gen)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------- the worker process
def asr_worker_main(wid, job_q, res_q, cfg):
    signal.signal(signal.SIGINT, signal.SIG_IGN)         # Ctrl+C belongs to the client, which stops us in order (and we flush)
    parent = mp.parent_process()
    t_start = time.time()
    try:
        ids, audio = sample_librispeech(cfg["n"], cfg["seed"])
        engine = WhisperEngine(cfg)
        res_q.put(("ready", wid, {"ids": ids, "durations": [len(a) / SAMPLE_RATE for a in audio], "pid": os.getpid(),
                                  "agent_loaded": agent_loaded(cfg["agent_so"]), "startup_s": time.time() - t_start}))
    except Exception as e:
        res_q.put(("fatal", wid, f"{type(e).__name__}: {e}"))
        flush_agent_timing(cfg["agent_so"])
        return
    try:
        while True:
            try:
                job = job_q.get(timeout=1.0)
            except queue.Empty:
                if not parent.is_alive():                # the client is gone: do not linger on the GPU
                    break
                continue
            if job is None:
                break
            job_id, k = job
            t = time.perf_counter()
            try:
                text, err = engine.transcribe(audio[k]), None
            except Exception as e:
                text, err = None, f"{type(e).__name__}: {e}"
            res_q.put(("res", job_id, wid, text, time.perf_counter() - t, err))
    finally:
        flush_agent_timing(cfg["agent_so"])
        res_q.put(("bye", wid))


# ---------------------------------------------------------------- client side
class AsrClient:
    def __init__(self, n, seed, model, dtype, max_new_tokens, workers, agent_so):
        self.cfg = dict(n=n, seed=seed, model=model, dtype=dtype, max_new_tokens=max_new_tokens, agent_so=agent_so)
        self.n_workers, self.agent_so = workers, agent_so
        self.procs, self.ids, self.durations = [], [], []
        self._pending, self._ids, self._stop = {}, itertools.count(), threading.Event()
        self._dead, self._reader, self._loop = None, None, None

    # -- start
    def _spawn(self):
        ctx = mp.get_context("spawn")
        # The queues come first: creating them starts multiprocessing's resource-tracker helper process, and that helper
        # must NOT inherit the preload (only the Whisper worker is meant to be a scheduled process).
        self.job_q, self.res_q = ctx.Queue(), ctx.Queue()
        old = os.environ.get("LD_PRELOAD")
        if self.agent_so:
            os.environ["LD_PRELOAD"] = self.agent_so + (":" + old if old else "")
        try:
            for w in range(self.n_workers):
                p = ctx.Process(target=asr_worker_main, args=(w, self.job_q, self.res_q, self.cfg),
                                daemon=True, name=f"asr-worker-{w}")
                p.start()
                self.procs.append(p)
        finally:                                          # the client's own environment goes back to what it was
            if old is None:
                os.environ.pop("LD_PRELOAD", None)
            else:
                os.environ["LD_PRELOAD"] = old

    def _hint(self):
        if self.agent_so:
            return (" The agent exits at once if scheduler_daemon is not running, and a stale /dev/shm/kernel_sched_shm_* "
                    "left by a crashed daemon makes the first kernel hang: rm -f /dev/shm/kernel_sched_shm_*, restart "
                    "the daemon, and start the server before the client.")
        return ""

    def _wait_ready(self):
        deadline, infos = time.time() + STARTUP_TIMEOUT_S, []
        while len(infos) < self.n_workers:
            try:
                msg = self.res_q.get(timeout=0.5)
            except queue.Empty:
                dead = [p for p in self.procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"the ASR worker exited during startup (exit code {dead[0].exitcode})." + self._hint())
                if time.time() > deadline:
                    raise RuntimeError(f"the ASR worker was not ready after {STARTUP_TIMEOUT_S} s "
                                       f"(dataset or model download, or a hang)." + self._hint())
                continue
            if msg[0] == "fatal":
                raise RuntimeError(f"the ASR worker failed to start: {msg[2]}")
            if msg[0] == "ready":
                infos.append(msg[2])
        return infos

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._spawn()
        infos = await self._loop.run_in_executor(None, self._wait_ready)
        self.ids, self.durations = infos[0]["ids"], infos[0]["durations"]
        if self.agent_so and not all(i["agent_loaded"] for i in infos):
            print("[asr] WARNING: agent.so is not mapped in the Whisper process (LD_PRELOAD ignored?): its kernels are "
                  "NOT going through the scheduler.", file=sys.stderr)
        print(f"[asr] {len(infos)} worker(s) ready (pid {', '.join(str(i['pid']) for i in infos)}), {len(self.ids)} "
              f"LibriSpeech utterance(s), {sum(self.durations):.0f} s of audio, agent "
              f"{'attached' if self.agent_so else 'NOT attached'}, startup {max(i['startup_s'] for i in infos):.1f} s",
              file=sys.stderr)
        self._reader = threading.Thread(target=self._read_results, name="asr-results", daemon=True)
        self._reader.start()

    # -- results
    def _read_results(self):
        while not self._stop.is_set():
            try:
                msg = self.res_q.get(timeout=0.5)
            except queue.Empty:
                if self._pending and self.procs and not any(p.is_alive() for p in self.procs):
                    self._loop.call_soon_threadsafe(self._fail_all, f"all ASR workers died (exit code {self.procs[0].exitcode})")
                continue
            if msg[0] == "res":
                _, jid, wid, text, service_s, err = msg
                self._loop.call_soon_threadsafe(self._resolve, jid, wid, text, service_s, err, time.perf_counter())

    def _resolve(self, jid, wid, text, service_s, err, t_now):
        item = self._pending.pop(jid, None)
        if item is None:
            return
        fut, t_submit = item
        total = t_now - t_submit
        if not fut.done():
            fut.set_result({"text": text, "service_s": service_s, "wait_s": max(0.0, total - service_s),
                            "total_s": total, "worker": wid, "error": err})

    def _fail_all(self, why):
        self._dead = why
        for jid in list(self._pending):
            fut, t_submit = self._pending.pop(jid)
            if not fut.done():
                fut.set_result({"text": None, "service_s": 0.0, "wait_s": 0.0, "total_s": time.perf_counter() - t_submit,
                                "worker": None, "error": why})

    async def transcribe(self, k):
        """Transcribe sample k. Never raises: failures come back in result['error']."""
        t = time.perf_counter()
        if self._dead:
            return {"text": None, "service_s": 0.0, "wait_s": 0.0, "total_s": 0.0, "worker": None, "error": self._dead}
        jid = next(self._ids)
        fut = self._loop.create_future()
        self._pending[jid] = (fut, t)
        self.job_q.put((jid, k))
        return await fut

    # -- stop
    def close(self, timeout=120.0):
        """Ask the workers to stop (they flush their kernel timing log on the way out) and wait for them."""
        for _ in self.procs:
            try:
                self.job_q.put(None)
            except Exception:
                pass
        deadline = time.time() + timeout
        for p in self.procs:
            p.join(max(0.1, deadline - time.time()))
            if p.is_alive():
                print(f"[asr] worker pid {p.pid} did not stop in {timeout:.0f} s: terminating (its kernel timing log is lost)",
                      file=sys.stderr)
                p.terminate()
        self._stop.set()
        if self._reader is not None:
            self._reader.join(2.0)
