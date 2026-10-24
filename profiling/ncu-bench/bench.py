import io
import os
import socket
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from vllm import LLM, SamplingParams

SEED = 42
QWEN_MAX_TOKENS     = 2048
WHISPER_MAX_TOKENS  = 2048
COMMON = dict(enforce_eager=True, enable_prefix_caching=False, gpu_memory_utilization=0.45)
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"


def librispeech(n):
    """The first n LibriSpeech test-clean utterances, as online_client.py picks them: ([(samples, rate)], [transcript])."""
    rows = list(load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
                .cast_column("audio", Audio(decode=False)).take(n))
    return [sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32") for r in rows], [r["text"] for r in rows]


def alpaca(n):
    rows = load_dataset("tatsu-lab/alpaca", split="train", streaming=True).take(n)
    return [r["instruction"] + ("\n\n" + r["input"] if r["input"] else "") for r in rows]

def main(which, n):
    if which == "qwen":
        llm = LLM("Qwen/Qwen3-4B", max_model_len=2048, **COMMON)
        chats = [[{"role": "user", "content": t}] for t in alpaca(n)]
        params = SamplingParams(temperature=0,
                                seed=SEED,
                                max_tokens=QWEN_MAX_TOKENS,
                                ignore_eos=True)
        call = lambda: llm.chat(chats,
                                params,
                                chat_template_kwargs={"enable_thinking": False},
                                use_tqdm=False) # turn off progbar
    else:
        llm = LLM("openai/whisper-large-v3",
                  limit_mm_per_prompt={"audio": 1}, **COMMON)
        
        clips = librispeech(n)[0]
        assert(all(len(a) / sr <= 30 for a, sr in clips))
        
        prompts = [{"encoder_prompt": {"prompt": "",
                                       "multi_modal_data": {"audio": clip}},
                    "decoder_prompt": WHISPER_PROMPT}
                    for clip in clips]
        params = SamplingParams(temperature=0,
                                seed=SEED,
                                max_tokens=WHISPER_MAX_TOKENS)
        call = lambda: llm.generate(prompts, params, use_tqdm=False)

    call() # warm-up (no profiling)
    
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    
    call()
    
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    
    print(f"{which}: batch of {n} done")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
