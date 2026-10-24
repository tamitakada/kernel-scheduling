#!/usr/bin/env bash
# Start `vllm serve` with the prefetch / kernel-scheduler infrastructure, for online (request-arrival) experiments.
#
#   ./serve_online.sh <model> [extra `vllm serve` args...]
#
# Same switches as the offline script:
#   SCHED_NO_CUPTI=1        hook-triggered prefetch, no kernel interception, no daemon decisions
#   SCHED_PREFETCH=0        hooks and scheduler stay, nothing is prefetched
#   SCHED_PREFETCH_MAX_MB / SCHED_PREFETCH_MIN_KB / SCHED_LOG_DIR as before
#   SCHED_FLUSH_INTERVAL_S  seconds between writes of the timing TSV (default 0: written once, when the server stops)
#   SCHED_FLUSH_IDLE_S      write the timing log this many seconds after the engine goes idle (default 5; 0 = off)
#   sched_trace.log         every important event is also appended to this file (in SCHED_LOG_DIR or the current dir)
#   SCHED_FLUSH_TRIGGER     `touch` this file (default ./sched_flush_now) to write the timing log NOW, without stopping
#                           the server; set to "" to disable.  Stop the server with ONE Ctrl+C and wait for it to finish.
#   SCHED_MODEL             which kernel table the scheduler daemon searches for this server's kernels (default: qwen)
#   PORT                    default 8000
#
# Start the scheduler daemon FIRST (in another terminal, after `rm -f /dev/shm/kernel_sched_shm_*`):
#   ./scheduler_daemon
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?usage: serve_online.sh <model> [vllm serve args...]}"
shift

export SCHED_MODEL="${SCHED_MODEL:-qwen}"                           # the daemon searches the qwen table for this server's kernels
export SCHED_AGENT_SO="${SCHED_AGENT_SO:-$HERE/agent.so}"           # absolute: the engine core is a separate process
[ -f "$SCHED_AGENT_SO" ] || { echo "agent.so not found at $SCHED_AGENT_SO (build it, or set SCHED_AGENT_SO)" >&2; exit 1; }
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"                # the engine core imports sched_worker / sched_hooks by name

if ! compgen -G "/dev/shm/kernel_sched_shm*" > /dev/null; then      # no pipeline: ls | grep -q + pipefail gives false "missing"
    echo "WARNING: no scheduler shared-memory segment in /dev/shm. Start ./scheduler_daemon first: without it the agent" \
         "exits at once and the engine process dies at startup. (A stale segment left by a crashed daemon is worse: the" \
         "first kernel hangs. rm -f /dev/shm/kernel_sched_shm_* and restart the daemon.)" >&2
fi

# CUPTI mode: preload the agent so it is loaded from the start in every vLLM process (as in your multi-process runs).
# SCHED_NO_CUPTI=1: no preload; sched_hooks loads it inside the engine's worker only, and it never touches CUPTI.
case "${SCHED_NO_CUPTI:-0}" in
    0|"") export LD_PRELOAD="$SCHED_AGENT_SO${LD_PRELOAD:+:$LD_PRELOAD}" ;;
esac

# --enforce-eager is required (CUDA graphs bypass the hooks and the launch interception); sched_worker enforces it.
exec vllm serve "$MODEL" \
    --worker-cls sched_worker.SchedWorker \
    --enforce-eager \
    --no-enable-prefix-caching \
    --max-num-seqs 1 \
    --port "${PORT:-8000}" \
    "$@"
