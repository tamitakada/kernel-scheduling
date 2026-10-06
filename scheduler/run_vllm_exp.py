import os
import sys
import time
import numpy as np

import ctypes
import torch
import torch.cuda.nvtx as nvtx

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams


# "HuggingFaceTB/SmolLM2-135M-Instruct"

agent = ctypes.CDLL("./agent.so")
agent.register_weight_region.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
agent.register_weight_region.restype = None
agent.sched_push_tag.argtypes = [ctypes.c_char_p]
agent.sched_push_tag.restype = None
agent.sched_pop_tag.argtypes = []
agent.sched_pop_tag.restype = None
agent.sched_flush_timing.argtypes = []
agent.sched_flush_timing.restype = None

def _make_push(name):
    b = name.encode()
    def hook(module, args):
        agent.sched_push_tag(b)
    return hook

def _pop(module, args, output):
    agent.sched_pop_tag()

def _find_layers(model):
    """The list of decoder layers: model.model.layers for Llama/Qwen-style
    models, else the first module list named '...layers'."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is not None:
        return list(layers)
    for name, mod in model.named_modules():
        if name.split(".")[-1] == "layers" and hasattr(mod, "__len__"):
            return list(mod)
    raise RuntimeError("could not find the decoder layers; print(model) and adjust _find_layers")


def _tag_module(agent, module, tag):
    """Wraps `module`'s forward in a push/pop of `tag` on the agent's tag stack."""
    tag_bytes = tag.encode("utf-8")

    def pre_hook(mod, args):
        agent.sched_push_tag(tag_bytes)    # returns None -> inputs left untouched

    def post_hook(mod, args, output):
        agent.sched_pop_tag()              # returns None -> output left untouched

    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)


def register_model(model, so_path=None, verbose=True):
    """Registers every decoder-layer parameter and tags every leaf submodule.
    Call once, after the weights are on the GPU and in the process that
    launches the kernels. Returns (tensors registered, modules tagged)."""
    layers = _find_layers(model)

    n_tensors = n_modules = 0
    for i, layer in enumerate(layers):
        for name, param in layer.named_parameters():
            tag = f"layer{i}_{name}".replace(".", "_").encode("utf-8")
            nbytes = param.numel() * param.element_size()
            agent.register_weight_region(tag, ctypes.c_void_p(param.data_ptr()), ctypes.c_size_t(nbytes))
            n_tensors += 1

        _tag_module(agent, layer, f"layer_{i}")
        n_modules += 1
        for name, sub in layer.named_modules():
            is_leaf = not any(True for _ in sub.children())
            if name and is_leaf:
                _tag_module(agent, sub, f"layer_{i}/{name.replace('.', '_')}")
                n_modules += 1

    if verbose:
        print(f"[scheduler_setup] {len(layers)} layers: registered {n_tensors} tensors, "
              f"tagged {n_modules} modules")
        if layers:
            leaves = [n.replace(".", "_") for n, m in layers[0].named_modules()
                      if n and not any(True for _ in m.children())]
            print(f"[scheduler_setup] layer 0 leaf modules tagged: {', '.join(leaves)}")
    return n_tensors, n_modules

def run_batch_infer(model_name, prompts, enable_pc=True, so_path="./agent.so"):    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

    llm = LLM(model=model_name,
              enforce_eager=True,
              enable_prefix_caching=enable_pc) #gpu_memory_utilization=0.5)

    llm.apply_model(register_model)

    warmup_prompts = [[{"role": "user", "content": "Recite the alphabet."}]]
    _ = llm.chat(warmup_prompts, sampling_params, chat_template_kwargs={
        "enable_thinking": False})
    
    # start = time.time()
    # torch.cuda.cudart().cudaProfilerStart()
    outputs = llm.chat(
        [[{"role": "user", "content": p}] for p in prompts],
        sampling_params,
        chat_template_kwargs={"enable_thinking": False})
    # torch.cuda.cudart().cudaProfilerStop()
    # total_time = time.time() - start
    return [o.outputs[0].text for o in outputs]


if __name__ == "__main__":
    model_name = sys.argv[1]
    # prompts = sys.argv[2:]
    
    prompts = [
        # "What was the number one best-selling album of the 1980s?",
        "Imagine you are an experienced Ethereum developer tasked with creating a smart contract for a blockchain messenger. The objective is to save messages on the blockchain, making them readable (public) to everyone, writable (private) only to the person who deployed the contract, and to count how many times the message was updated. Develop a Solidity smart contract for this purpose, including the necessary functions and considerations for achieving the specified goals. Please provide the code and any relevant explanations to ensure a clear understanding of the implementation.",
        "I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets {like this}. my first command is pwd",
        "I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations. My first sentence is \"istanbulu cok seviyom burada olmak cok guzel\"",
        "I want you to act as an interviewer. I will be the candidate and you will ask me the interview questions for the ${Position:Software Developer} position. I want you to only reply as the interviewer. Do not write all the conversation at once. I want you to only do the interview with me. Ask me the questions and wait for my answers. Do not write explanations. Ask me the questions one by one like an interviewer does and wait for my answers. My first sentence is \"Hi\"",
        "I want you to act as a javascript console. I will type commands and you will reply with what the javascript console should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets {like this}. my first command is console.log(\"Hello World\");"
    ]
    
    data = run_batch_infer(model_name, prompts)
    print(data) 

    agent.sched_flush_timing.argtypes = []
    agent.sched_flush_timing()

    # with open("responses.txt", "a") as f:
    #     f.write(f"{data}\n")

