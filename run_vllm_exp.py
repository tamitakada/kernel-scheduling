import os
import sys
import time
import numpy as np

from vllm import LLM, SamplingParams


# "HuggingFaceTB/SmolLM2-135M-Instruct"

def run_batch_infer(model_name, prompts, enable_pc=True):
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model=model_name,
              enforce_eager=True,
              enable_prefix_caching=enable_pc,) #gpu_memory_utilization=0.5)

    warmup_prompts = [[{"role": "user", "content": "Recite the alphabet."}]]
    _ = llm.chat(warmup_prompts, sampling_params, chat_template_kwargs={
        "enable_thinking": False})
    
    start = time.time()
    
    outputs = llm.chat(
        [[{"role": "user", "content": p}] for p in prompts],
        sampling_params,
        chat_template_kwargs={"enable_thinking": False})
        
    total_time = time.time() - start
    
    return (total_time, [len(o.outputs[0].text) for o in outputs])


if __name__ == "__main__":
    model_name = sys.argv[1]
    prompts = sys.argv[2:]
    
    """
    prompts = [
        "What is the best selling album of the 1980s?",
        "Recommend the best dishes to try in Italy.",
        "My favorite book is 'Gone with the Wind.' What else should I read?",
        "How much has Google grown since its founding?",
        "Rank the ten most popular AI chatbots.",
        "Who was the longest living president of the US?"
    ]
    """

    print("Load model ", model_name)
    
    data = run_batch_infer(model_name, prompts)
    
    print(data)
    #with open("results.txt", "a") as f:
    #    f.write(f"{data}\n")

