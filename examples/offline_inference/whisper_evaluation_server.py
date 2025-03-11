import gradio as gr
import torch
from datasets import load_dataset, get_dataset_config_names
import time
from math import ceil
from vllm import LLM, SamplingParams, AsyncLLMEngine
from vllm.config import ModelConfig, SchedulerConfig

import numpy as np
import os
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio
import random 
from vad import estimate_speech_duration


# Enable torch profiler
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

# Global settings
DATASET = "esb/diagnostic-dataset"
SAMPLING_RATE = 16000

async_engine_args = AsyncEngineArgs(
    model="openai/whisper-large-v3",
    max_model_len=448,
    max_num_seqs=16,
    limit_mm_per_prompt={"audio": 1},
    kv_cache_dtype="fp8",
    scheduling_policy='priority',
)

llm = AsyncLLMEngine.from_engine_args(async_engine_args)

processor = WhisperProcessor.from_pretrained("openai/whisper-large")

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

def reorder_dataset_by_audio_length(dataset):
    samples_with_lengths = []
    for audio_sample in dataset["clean"]:
        duration = estimate_speech_duration(audio_sample["audio"]['array'])
        samples_with_lengths.append((audio_sample, duration))

    sorted_dataset = sorted(samples_with_lengths, key=lambda x: x[1])
    final_dataset = [item[0] for item in sorted_dataset]
    return final_dataset


########################################################################
# This generator function loads the chosen dataset and runs the model
# in batches. It yields a tuple of (profiling text, progress fraction)
########################################################################
async def run_whisper(selected_dataset, num_samples, batch_size, temperature, top_p, max_tokens, inference_mode, request_rate, dataset_order, scheduling_option):
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
    elif dataset_order == 'audio_length':
        dataset = reorder_dataset_by_audio_length(dataset)
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
    num_samples = count
    overall_start = time.time()

    # Create the sampling parameters using user-defined values.
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    prompts = prompts

    latencies = []

    if inference_mode == "async":
        async def generate_async(prompt, idx, latencies):
            if scheduling_option == 'priority':
                priority = estimate_speech_duration(prompt["encoder_prompt"]["multi_modal_data"]["audio"][0])
            elif scheduling_option == 'reverse_priority':
                priority = -estimate_speech_duration(prompt["encoder_prompt"]["multi_modal_data"]["audio"][0])
            elif scheduling_option == 'fifo':
                priority = 0

            start_time = time.time()
            results_generator = llm.generate(prompt, sampling_params, request_id=f"{idx}", priority=priority)
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
        
        tasks = [generate_async(prompt, idx, latencies) for idx, prompt in enumerate(prompts)]
        output = await asyncio.gather(*tasks)  # Runs all at once

        average_latency = sum(latencies) / len(latencies)
        profiling_output = "".join(output)
        profiling_output += f"\n Average Latency: {average_latency:.2f} seconds\n"
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
        async def generate_async(prompt, idx, latencies, absolute_start_time):
            results_generator = llm.generate(prompt, sampling_params, request_id=f"{idx}")
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            end_time = time.time()
            assert final_output is not None
            prompt = final_output.prompt
            assert prompt is not None
            text = [prompt + output.text for output in final_output.outputs]
            latency = end_time - absolute_start_time
            latencies.append(latency)

            return f"Sample #{idx}: {text}\n"
        
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

        sim_start_time = time.time()  # Record the simulation start (absolute time)
        # Continue simulation until all requests have arrived and been processed.
        index = 1
        tasks = []

        while next_arrival_index < num_samples or arrived_prompts:
            current_sim_time = time.time() - sim_start_time  # current simulation time (in sec)

            # Add requests that have "arrived" (i.e., their scheduled arrival time has passed).
            while next_arrival_index < num_samples and delay_prefixes[next_arrival_index].item() <= current_sim_time:
                arrived_prompts.append((prompts[next_arrival_index], delay_prefixes[next_arrival_index].item()))
                next_arrival_index += 1

            if len(arrived_prompts) >= 1:
                prompt, arrival_time = arrived_prompts.pop(0)

                task = asyncio.create_task(generate_async(prompt, index, latencies, sim_start_time + arrival_time))
                tasks.append(task)
                index += 1

            # # Sleep briefly to avoid busy waiting.
            # time.sleep(0.01)
            await asyncio.sleep(0.01)

        results = await asyncio.gather(*tasks)
        profiling_output = "".join(results)
        average_latency = sum(latencies) / len(latencies)

        profiling_output += f"\n Average Latency: {average_latency:.2f} seconds\n"

        # if norm_latencies:
        #     avg_norm_latency = sum(norm_latencies) / len(norm_latencies)
        # else:
        #     avg_norm_latency = 0.0
        # # Convert to ms/token for display.
        # profiling_output += f"Average Normalized Latency: {avg_norm_latency:.2f} sec/token\n"

    # After all requests have been processed, calculate overall metrics.
    overall_duration = time.time() - overall_start
    profiling_output += f"\nOverall Duration: {overall_duration:.2f} seconds\n"
    profiling_output += f"RPS: {len(prompts) / overall_duration:.2f}\n"
    average_latency = sum(latencies) / len(latencies)
    yield latencies, average_latency

########################################################################
# Build the Gradio Interface
########################################################################
with gr.Blocks() as demo:
    gr.Markdown("# Whisper Model Profiling Interface")
    gr.Markdown(
        "Select one of the 8 datasets by clicking its button, then set the parameters "
        "and click **Run Evaluation** to process the samples. The evaluation results "
        "and a progress bar (below) will be updated as processing proceeds."
    )

    # --- Dataset Selection ---
    with gr.Row():
        # A hidden state to hold the selected dataset configuration.
        dataset_state = gr.State(value="")
        # A textbox to show which dataset was selected.
        dataset_display = gr.Textbox(label="Selected Dataset", interactive=False)

    with gr.Row():
        # Create eight buttons (one per dataset config).
        btn0 = gr.Button(config_names[0])
        btn1 = gr.Button(config_names[1])
        btn2 = gr.Button(config_names[2])
        btn3 = gr.Button(config_names[3])
        btn4 = gr.Button(config_names[4])
        btn5 = gr.Button(config_names[5])
        btn6 = gr.Button(config_names[6])
        btn7 = gr.Button(config_names[7])


    # Each button click sets the hidden state and updates the display.
    btn0.click(lambda: (config_names[0], f"Selected dataset: {config_names[0]}"),
               None, [dataset_state, dataset_display])
    btn1.click(lambda: (config_names[1], f"Selected dataset: {config_names[1]}"),
               None, [dataset_state, dataset_display])
    btn2.click(lambda: (config_names[2], f"Selected dataset: {config_names[2]}"),
               None, [dataset_state, dataset_display])
    btn3.click(lambda: (config_names[3], f"Selected dataset: {config_names[3]}"),
               None, [dataset_state, dataset_display])
    btn4.click(lambda: (config_names[4], f"Selected dataset: {config_names[4]}"),
               None, [dataset_state, dataset_display])
    btn5.click(lambda: (config_names[5], f"Selected dataset: {config_names[5]}"),
               None, [dataset_state, dataset_display])
    btn6.click(lambda: (config_names[6], f"Selected dataset: {config_names[6]}"),
               None, [dataset_state, dataset_display])
    btn7.click(lambda: (config_names[7], f"Selected dataset: {config_names[7]}"),
               None, [dataset_state, dataset_display])

    # --- Parameter Inputs ---
    gr.Markdown("### Set Parameters")
    with gr.Row():
        num_samples_input = gr.Number(value=100, label="Number of Samples", precision=0)
        batch_size_input = gr.Number(value=10, label="Batch Size", precision=0)
    with gr.Row():
        temperature_input = gr.Slider(0, 1, value=0, step=0.01, label="Temperature")
        top_p_input = gr.Slider(0, 1, value=1.0, step=0.01, label="Top-p")
        max_tokens_input = gr.Number(value=200, label="Max Tokens", precision=0)

    # --- Online Inference Parameters ---
    gr.Markdown("### Set Online Inference Parameters")
    with gr.Row():
        request_rate = gr.Number(value=10, label="Request Rate", precision=0)

    # --- Online/Offline Inference Selection  ---
    with gr.Row():
        # A hidden state to hold the selected dataset configuration.
        inference_mode = gr.State(value="")
        # A textbox to show which dataset was selected.
        inference_display = gr.Textbox(label="Selected Inference Mode", interactive=False)

    with gr.Row():
        offline_btn = gr.Button("Offline Inference")
        online_btn = gr.Button("Online Inference")
        async_btn = gr.Button("Async Inference")

    offline_btn.click(lambda: ("offline", f"Selected mode: Offline Inference"),
               None, [inference_mode, inference_display])
    
    online_btn.click(lambda: ("online", f"Selected mode: Online Inference"),
               None, [inference_mode, inference_display])
    
    async_btn.click(lambda: ("async", f"Selected mode: Async Inference"),
               None, [inference_mode, inference_display])
    
    # Optimizations
    with gr.Row():
        # A hidden state to hold the selected dataset configuration.
        dataset_order = gr.State(value="")
        # A textbox to show which dataset was selected.
        dataset_order_display = gr.Textbox(label="Dataset Order", interactive=False)

    with gr.Row():
        dataset_shuffle_btn = gr.Button("Dataset Shuffle")
        dataset_forward_btn = gr.Button("Dataset Forward")
        dataset_reverse_btn = gr.Button("Dataset Reverse")
        dataset_audio_btn = gr.Button("Dataset Audio Length Forward")

    dataset_shuffle_btn.click(lambda: ("shuffle", f"Selected mode: Dataset Shuffle"),
               None, [dataset_order, dataset_order_display])

    dataset_forward_btn.click(lambda: ("forward", f"Selected mode: Dataset Forward"),
               None, [dataset_order, dataset_order_display])
    
    dataset_reverse_btn.click(lambda: ("reverse", f"Selected mode: Dataset Reverse"),
               None, [dataset_order, dataset_order_display])
    
    dataset_audio_btn.click(lambda: ("audio_length", f"Selected mode: Dataset Audio Length Forward"),
               None, [dataset_order, dataset_order_display])
    
    # Display options to use priority scheduling or not
    with gr.Row():
        scheduling_option = gr.State(value="fifo")
        scheduling_option_display = gr.Textbox(label="Scheduling", interactive=False)

    with gr.Row():
        priority_btn = gr.Button("Priority Scheduling")
        reverse_priority_btn =  gr.Button("Reverse Priority Scheduling")
        fifo_btn = gr.Button("FIFO Scheduling")

    priority_btn.click(lambda: ("priority", f"Selected mode: Priority Scheduling"),
               None, [scheduling_option, scheduling_option_display])

    reverse_priority_btn.click(lambda: ("reverse_priority", f"Selected mode: Reverse Priority Scheduling"),
               None, [scheduling_option, scheduling_option_display])
    
    fifo_btn.click(lambda: ("fifo", f"Selected mode: FIFO Scheduling"),
               None, [scheduling_option, scheduling_option_display])

    # --- Run Button and Outputs ---
    run_button = gr.Button("Run Evaluation")

    # A textbox to display the accumulated evaluation results.
    profiling_output_box = gr.Textbox(label="Evaluation Results", lines=20)
    # A slider (read-only) to show the progress (0 to 1).
    progress_bar = gr.Slider(minimum=0, maximum=1, value=0, label="Progress", interactive=False)

    # When the "Run Evaluation" button is clicked, the run_whisper generator is called.
    # Using show_progress=True will also display Gradio's built-in progress indicator.
    run_button.click(
        fn=run_whisper,
        inputs=[dataset_state, num_samples_input, batch_size_input, temperature_input, top_p_input, max_tokens_input, inference_mode, request_rate, dataset_order, scheduling_option],
        outputs=[profiling_output_box, progress_bar],
        show_progress=True,
    )

# demo.launch(server_port=3333)

