"""
Pairs with agent.so (built from agent.cpp). Run your vLLM script with:

    LD_PRELOAD=./agent.so python your_script.py

...and call setup_scheduler(llm) once, right after `LLM(...)` construction,
before serving any requests.
"""

import ctypes
import torch.cuda.nvtx as nvtx


def setup_scheduler(llm, so_path="./agent.so"):
    agent = ctypes.CDLL(so_path)
    agent.register_weight_region.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    agent.register_weight_region.restype = None

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    # path varies by vLLM version / architecture -- confirm with:
    #   print(model)
    layers = model.model.layers

    for i, layer in enumerate(layers):
        # register every weight tensor in this layer under "layer{i}_{param_name}"
        for name, param in layer.named_parameters():
            tag = f"layer{i}_{name}".replace(".", "_").encode("utf-8")
            addr = param.data_ptr()
            nbytes = param.numel() * param.element_size()
            agent.register_weight_region(tag, ctypes.c_void_p(addr), ctypes.c_size_t(nbytes))

        # tag every kernel launched during this layer's forward() with "layer_{i}"
        # (matches the "layer_" prefix parsing in policy_choose_prefetch in scheduler_daemon.cpp)
        layer.register_forward_pre_hook(lambda m, inp, idx=i: nvtx.range_push(f"layer_{idx}"))
        layer.register_forward_hook(lambda m, inp, out, idx=i: nvtx.range_pop())

    print(f"[scheduler_setup] registered {len(layers)} layers")
