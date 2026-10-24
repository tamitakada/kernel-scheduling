#!/usr/bin/env bash
# Start Whisper under vLLM as a SECOND server on the same GPU as the LLM, under the same kernel scheduler.
#
#   ./serve_whisper.sh [model] [extra `vllm serve` args...]      default model: openai/whisper-large-v3, port 8001 (PORT=...)
#
# Differences from serve_online.sh: no prefetch (SCHED_PREFETCH=0 here: Whisper has no registered weights, its kernels are
# only intercepted and timed) and a worker class that installs just the timing-log flush. Its own trigger file, so
# `touch sched_flush_now_whisper` flushes this server and `touch sched_flush_now` the LLM's.
#
# Order: scheduler_daemon, then the LLM server, then this one, one after the other (not together). vLLM counts the memory the
# first server already holds when it sizes the second, so the later server's --gpu-memory-utilization has to cover that too
# (for example LLM 0.6, Whisper 0.8); raise it if the second server says it has no memory for the cache blocks.
# Needs vLLM's audio extras on the server side: pip install "vllm[audio]".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="openai/whisper-large-v3"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then MODEL="$1"; shift; fi

export SCHED_MODEL=whisper                                          # the daemon searches the whisper table for this server's kernels
export SCHED_AGENT_SO="${SCHED_AGENT_SO:-$HERE/agent.so}"           # absolute: the engine core is a separate process
[ -f "$SCHED_AGENT_SO" ] || { echo "agent.so not found at $SCHED_AGENT_SO (build it, or set SCHED_AGENT_SO)" >&2; exit 1; }
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"                # the engine core imports sched_worker / sched_hooks by name

if ! compgen -G "/dev/shm/kernel_sched_shm*" > /dev/null; then
    echo "WARNING: no scheduler shared-memory segment in /dev/shm. Start ./scheduler_daemon first: without it the agent" \
         "exits at once and the engine process dies at startup. (A stale segment left by a crashed daemon is worse: the" \
         "first kernel hangs. rm -f /dev/shm/kernel_sched_shm_* and restart the daemon.)" >&2
fi

export SCHED_PREFETCH=0                                             # never inject prefetch into Whisper
export SCHED_FLUSH_TRIGGER="${SCHED_FLUSH_TRIGGER:-sched_flush_now_whisper}"
case "${SCHED_NO_CUPTI:-0}" in                                      # as serve_online.sh: preload the agent unless hook mode
    0|"") export LD_PRELOAD="$SCHED_AGENT_SO${LD_PRELOAD:+:$LD_PRELOAD}" ;;
esac

# --enforce-eager is required: CUDA graphs bypass the launch interception (the worker class refuses to start without it).
exec vllm serve "$MODEL" \
    --worker-cls sched_worker.SchedFlushWorker \
    --enforce-eager \
    --no-enable-prefix-caching \
    --max-num-seqs 1 \
    --max-num-batched-tokens 1500 \
    --port "${PORT:-8001}" \
    "$@"
