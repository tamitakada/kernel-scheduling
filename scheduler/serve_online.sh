# start vLLM online server for model passed as arg
export PYTHONPATH=$(pwd):$PYTHONPATH
vllm serve "$1" --worker-cls sched_worker.SchedWorker --enforce-eager --port 8000