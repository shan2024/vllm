import gradio as gr
import torch
from datasets import load_dataset, get_dataset_config_names
import time
from math import ceil
from vllm import LLM, SamplingParams, AsyncLLMEngine
from vllm.config import ModelConfig
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio
import random

# Enable torch profiler
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set this to the GPU num that of a GPU not in use
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Global settings
DATASET = "esb/diagnostic-dataset"
SAMPLING_RATE = 16000

# Initialize the Whisper model via vLLM
llm = None
# llm = LLM(
#     model="openai/whisper-large-v3",
#     max_model_len=448,
#     max_num_seqs=400,
#     limit_mm_per_prompt={"audio": 1},
#     kv_cache_dtype="fp8",
# )



# async_engine_args = AsyncEngineArgs(
#     model="openai/whisper-large-v3",
#     max_model_len=448,
#     max_num_seqs=16,
#     limit_mm_per_prompt={"audio": 1},
#     kv_cache_dtype="fp8",
# )

# llm = AsyncLLMEngine.from_engine_args(async_engine_args)

processor = None
# processor = WhisperProcessor.from_pretrained("openai/whisper-large")

# Get the list of dataset configurations (we assume there are at least 8)
config_names = list(get_dataset_config_names(DATASET))[:8]


# Helper function to reorder the dataset based on the length
def reorder_dataset(dataset, reverse=False):
    samples_with_lengths = []
    for audio_sample in dataset["clean"]:
        ground_truth = audio_sample["ortho_transcript"]
        tokenized_gt = processor.tokenizer(ground_truth, return_tensors="pt")
        decode_length = tokenized_gt.input_ids.shape[1]  # Number of tokens
        samples_with_lengths.append((audio_sample, decode_length))

    sorted_dataset = sorted(samples_with_lengths, key=lambda x: x[1], reverse=reverse)
    final_dataset = [item[0] for item in sorted_dataset]
    return final_dataset

def shuffle_dataset(dataset):
    shuffled_dataset = []
    for audio_sample in dataset["clean"]:
        shuffled_dataset.append(audio_sample)

    random.shuffle(shuffled_dataset)
    return shuffled_dataset

########################################################################
# Stat computation helper functions
########################################################################
def save_plot(list1, list2, xlab, ylab):
    """
    Plots two lists on a graph.
    list1: X-axis values
    list2: Y-axis values
    xlab: The label for the X-axis (e.g. "Batch Size")
    ylab: The label for the Y-axis (e.g. "99% Tail Latency")
    """
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(list1, list2, marker='o', linestyle='-', color='b', label='Data Line')
    
    # Adding titles and labels
    plt.title(f"{ylab} vs. {xlab}")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    # Show the legend
    plt.legend()
    plt.savefig(f"assets/latency_histogram_{ylab}_vs_{xlab}.png", dpi=600)
    plt.show()

########################################################################
# This generator function loads the chosen dataset and runs the model
# in batches. It yields a tuple of (profiling text, progress fraction)
########################################################################
async def run_whisper(selected_dataset, num_samples, batch_size, temperature, top_p, max_tokens, inference_mode, request_rate, dataset_order):
    if not selected_dataset:
        yield "Error: No dataset selected.", 0
        return
    if not inference_mode:
        yield "Error: No inference mode selected.", 0
        return

    profiling_output = ""
    # (Re)load the dataset using the selected configuration name.
    dataset = load_dataset(DATASET, selected_dataset)
    prompts = []
    count = 0

    # Reorder dataset if 'batch reordering' option selected
    if dataset_order == 'forward':
        dataset = reorder_dataset(dataset, reverse=False)
    elif dataset_order == 'reverse':
        dataset = reorder_dataset(dataset, reverse=True)
    elif dataset_order == 'shuffle':
        dataset = shuffle_dataset(dataset)
    else:
        dataset = dataset["clean"]

    # Here we use the "clean" split and build a list of prompts.
    for audio_sample in dataset:
        audio_array = audio_sample["audio"]['array']
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio_array, SAMPLING_RATE),
                },
            },
            "decoder_prompt": "<|startoftranscript|>",
        }
        prompts.append(prompt)
        count += 1
        if count >= num_samples:
            break
    overall_start = time.time()

    # Create the sampling parameters using user-defined values.
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    prompts = prompts

    if inference_mode == "async":
        async def generate_async(prompt, idx, latencies):
            start_time = time.time()

            results_generator = llm.generate(prompt, sampling_params, request_id=f"{idx}")
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            end_time = time.time()
            assert final_output is not None
            prompt = final_output.prompt
            assert prompt is not None
            text = [prompt + output.text for output in final_output.outputs]
            latency = end_time - start_time
            latencies.append(latency)

            return f"Sample #{idx}: {text}\n"
        
        latencies = []

        tasks = [generate_async(prompt, idx, latencies) for idx, prompt in enumerate(prompts)]
        output = await asyncio.gather(*tasks)  # Runs all at once

        average_latency = sum(latencies) / len(latencies)
        profiling_output = "".join(output)
        profiling_output += f"\n Average Latency: {average_latency:.2f} seconds\n"
        p99_latency = np.percentile(latencies, 99)
        yield p99_latency
        return
    elif inference_mode == "offline":
        # llm.start_profile()
        generate_start_time = time.time()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        generate_end_time = time.time()
        generate_time = generate_end_time - generate_start_time

        for i, output in enumerate(outputs):
            transcription = output.outputs[0].text
            profiling_output += f"Sample {i}: {transcription}\n"

        profiling_output += f"\nGenerate Overall Duration: {generate_time:.2f} seconds\n"
        profiling_output += f"Generate RPS: {len(prompts) / generate_time:.2f}\n"
        
        # llm.stop_profile()
        # time.sleep(10)
    elif inference_mode == "online":
        async def generate_async(prompt, idx, latencies, arrival_time, sim_start_time):
            results_generator = llm.generate(prompt, sampling_params, request_id=f"{idx}")
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            end_time = time.time()
            assert final_output is not None
            prompt = final_output.prompt
            assert prompt is not None
            text = [prompt + output.text for output in final_output.outputs]
            latency = end_time - sim_start_time - arrival_time
            latencies.append(latency)

            return f"Sample #{idx}: {text}\n"
        latencies = []

        # Online: Simulate realistic arrival times.
        # 1. Sample inter-arrival delays (in seconds) from an exponential distribution.
        exp_dist = torch.distributions.Exponential(request_rate)
        request_delay = exp_dist.sample((num_samples,))
        # 2. Compute cumulative arrival times (relative to simulation start).
        delay_prefixes = torch.cumsum(request_delay, dim=0)

        # For batching, maintain a queue of arrived prompts along with their scheduled arrival times.
        # Each entry is a tuple: (prompt, scheduled_arrival_time)
        arrived_prompts = []
        next_arrival_index = 0
        processed_count = 0  # Counter for processed requests.
        norm_latencies = []  # List to store normalized latencies (in sec/token)
        simulation_end_time = delay_prefixes[-1].item()  # Total simulation duration (relative)
        sim_start_time = time.time()  # Record the simulation start (absolute time)

        # Continue simulation until all requests have arrived and been processed.
        index = 1
        while next_arrival_index < num_samples or arrived_prompts:
            current_sim_time = time.time() - sim_start_time  # current simulation time (in sec)

            # Add requests that have "arrived" (i.e., their scheduled arrival time has passed).
            while next_arrival_index < num_samples and delay_prefixes[next_arrival_index].item() <= current_sim_time:
                arrived_prompts.append((prompts[next_arrival_index], delay_prefixes[next_arrival_index].item()))
                next_arrival_index += 1

            # If enough requests have accumulated for a batch, or if no more are coming, process a batch.
            if len(arrived_prompts) >= 1 or (next_arrival_index >= num_samples and arrived_prompts):
                # Extract batch data.
                prompt = arrived_prompts[0][0]
                arrival_time = arrived_prompts[0][1]
                arrived_prompts = arrived_prompts[1:]

                generate_async(prompt, index, latencies, arrival_time, sim_start_time)
                index += 1

            # Sleep briefly to avoid busy waiting.
            time.sleep(0.01)

        average_latency = sum(latencies) / len(latencies)
        profiling_output = "".join(output)
        profiling_output += f"\n Average Latency: {average_latency:.2f} seconds\n"

        if norm_latencies:
            avg_norm_latency = sum(norm_latencies) / len(norm_latencies)
        else:
            avg_norm_latency = 0.0
        # Convert to ms/token for display.
        profiling_output += f"Average Normalized Latency: {avg_norm_latency:.2f} sec/token\n"

    # After all requests have been processed, calculate overall metrics.
    overall_duration = time.time() - overall_start
    profiling_output += f"\nOverall Duration: {overall_duration:.2f} seconds\n"
    profiling_output += f"RPS: {len(prompts) / overall_duration:.2f}\n"

    yield profiling_output, 1.0

dataset_state = "ami"
num_samples_input = 10000
batch_size_input = 10
temperature_input = 0
top_p_input = 1.0
max_tokens_input = 200
inference_mode = "async"
request_rate = 10
dataset_order = "forward"

# Batch size vs 99th tail latency
async def result1():
  global processor
  global llm
  
  batch_sizes = [16, 32, 64, 128]
  tail_latencies = []
  for bs in batch_sizes:
    # Reset args based on batch size
    async_engine_args = AsyncEngineArgs(
        model="openai/whisper-large-v3",
        max_model_len=448,
        max_num_seqs=bs,
        limit_mm_per_prompt={"audio": 1},
        kv_cache_dtype="fp8",
    )
    llm = AsyncLLMEngine.from_engine_args(async_engine_args)
    processor = WhisperProcessor.from_pretrained("openai/whisper-large")

    print("============DONE SETUP===========")

    tail_latency = None
    async for res in run_whisper(dataset_state, num_samples_input, batch_size_input, temperature_input, top_p_input, max_tokens_input, inference_mode, request_rate, dataset_order):
      tail_latency = res

    tail_latencies.append(tail_latency)
    llm.shutdown_background_loop()
    del async_engine_args, llm, processor
    torch.cuda.empty_cache()
    time.sleep(5)

    print(f"============DONE RUN==============\nbatch_size: {bs}; tail_latency: {tail_latency}")

  save_plot(batch_sizes, tail_latencies, "Batch Size", "99th Tail Latency")


# Dataset vs 99th tail latency

asyncio.run(result1())


