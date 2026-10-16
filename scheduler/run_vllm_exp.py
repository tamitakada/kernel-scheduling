import os
import sys
import pandas as pd
import ctypes
import torch

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams


agent = ctypes.CDLL("./agent.so")
agent.register_weight_region.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
agent.register_weight_region.restype = None
agent.sched_push_tag.argtypes = [ctypes.c_char_p]
agent.sched_push_tag.restype = None
agent.sched_pop_tag.argtypes = []
agent.sched_pop_tag.restype = None
agent.sched_flush_timing.argtypes = []
agent.sched_flush_timing.restype = None
agent.sched_prefetch.argtypes = [ctypes.c_char_p, ctypes.c_longlong]
agent.sched_prefetch.restype = ctypes.c_longlong
agent.sched_prefetch_ranges.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                                        ctypes.c_int, ctypes.c_longlong]
agent.sched_prefetch_ranges.restype = ctypes.c_longlong


def _env_flag(name):
    v = os.environ.get(name, "")
    return v not in ("", "0")


def _find_layers(model) -> list:
    """Returns list of all decoder layers, i.e. model.model.layers for 
    Llama/Qwen-style models."""

    if model.model.layers is None:
        raise RuntimeError("Bad model")
    return model.model.layers


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
    Call once, after the weights are on the GPU and in the process that launches the kernels."""

    if should_tag_hooks is None:
        should_tag_hooks = not _env_flag("SCHED_NO_CUPTI")

    layers = _find_layers(model)

    for i, layer in enumerate(layers):
        for name, param in layer.named_parameters():
            tag = f"layer{i}_{name}".replace(".", "_").encode("utf-8")
            nbytes = param.numel() * param.element_size()
            agent.register_weight_region(
                tag, 
                ctypes.c_void_p(param.data_ptr()), 
                ctypes.c_size_t(nbytes))

        if should_tag_hooks: # register prefetch tags before/after modules
            # tag the layer then any sublayers
            _tag_module(agent, layer, f"layer_{i}")

            # TODO: confirm layer.named_modules() walks ALL nodes in tree
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


def run_batch_infer(model_name, prompts, enable_pc=False):    
    sampling_params = SamplingParams(
        temperature=0.8, 
        seed=42,
        top_p=0.95, 
        max_tokens=2048)

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        enable_prefix_caching=enable_pc,
        gpu_memory_utilization=0.9,
        disable_log_stats=False
    )

    print("Loaded llm, begin register...")
    
    llm.apply_model(register_model)

    print("Finished register, begin warmup...")

    warmup_prompts = [[{"role": "user", "content": "Recite the alphabet."}]]
    _ = llm.chat(warmup_prompts, sampling_params, chat_template_kwargs={"enable_thinking": False})
    
    print("Finished warmup, begin profiling...")

    # torch.cuda.cudart().cudaProfilerStart()

    outputs = llm.chat(
        [[{"role": "user", "content": p}] for p in prompts],
        sampling_params,
        chat_template_kwargs={"enable_thinking": False})
    
    # torch.cuda.cudart().cudaProfilerStop()

    # for o in outputs:
    #     ttft = o.metrics.first_token_latency
    #     tpot = (o.metrics.last_token_ts - o.metrics.first_token_ts) / (o.metrics.num_generation_tokens - 1)
    #     print("TTFT: ", ttft, ", TPOT: ", tpot)
    
    return [o.outputs[0].text for o in outputs]


if __name__ == "__main__":
    model_name = sys.argv[1]
    
    dataset_df = pd.read_csv("hf://datasets/fka/prompts.chat/prompts.csv")
    prompts = dataset_df["prompt"].sample(n=1000, random_state=42).tolist()
    
    # prompts = [
    #     "Imagine you are an experienced Ethereum developer tasked with creating a smart contract for a blockchain messenger. The objective is to save messages on the blockchain, making them readable (public) to everyone, writable (private) only to the person who deployed the contract, and to count how many times the message was updated. Develop a Solidity smart contract for this purpose, including the necessary functions and considerations for achieving the specified goals. Please provide the code and any relevant explanations to ensure a clear understanding of the implementation.",
    #     "I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets {like this}. my first command is pwd",
    #     "I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations. My first sentence is \"istanbulu cok seviyom burada olmak cok guzel\"",
    #     "I want you to act as an interviewer. I will be the candidate and you will ask me the interview questions for the ${Position:Software Developer} position. I want you to only reply as the interviewer. Do not write all the conversation at once. I want you to only do the interview with me. Ask me the questions and wait for my answers. Do not write explanations. Ask me the questions one by one like an interviewer does and wait for my answers. My first sentence is \"Hi\"",
    #     "I want you to act as a javascript console. I will type commands and you will reply with what the javascript console should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets {like this}. my first command is console.log(\"Hello World\");"
    # ]

    print(f"Received {model_name}, starting load...")
    
    data = run_batch_infer(model_name, prompts)

    with open("out.txt", "w") as f:
        f.write("\n\n".join(data))
        f.close()

    agent.sched_flush_timing.argtypes = []
    agent.sched_flush_timing()