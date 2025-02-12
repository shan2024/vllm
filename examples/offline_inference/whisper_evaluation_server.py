import gradio as gr
import torch
from datasets import load_dataset, get_dataset_config_names
import time
from math import ceil
from vllm import LLM, SamplingParams
import numpy as np

# Global settings
DATASET = "esb/diagnostic-dataset"
SAMPLING_RATE = 16000

# Initialize the Whisper model via vLLM
llm = LLM(
    model="openai/whisper-large-v3",
    max_model_len=448,
    max_num_seqs=400,
    limit_mm_per_prompt={"audio": 1},
    kv_cache_dtype="fp8",
)

# Get the list of dataset configurations (we assume there are at least 8)
config_names = list(get_dataset_config_names(DATASET))[:8]

########################################################################
# This generator function loads the chosen dataset and runs the model
# in batches. It yields a tuple of (profiling text, progress fraction)
########################################################################
def run_whisper(selected_dataset, num_samples, batch_size, temperature, top_p, max_tokens, inference_mode, request_rate):
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
    # Here we use the "clean" split and build a list of prompts.
    for audio_sample in dataset["clean"]:
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

    total_batches = ceil(len(prompts) / batch_size)
    overall_start = time.time()

    # Create the sampling parameters using user-defined values.
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    if inference_mode == "offline":
        # Process the prompts in batches.
        for batch_idx in range(total_batches):
            batch_prompts = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_start = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            batch_duration = time.time() - batch_start

            # Append results for each sample in the batch.
            for i, output in enumerate(outputs):
                transcription = output.outputs[0].text
                sample_index = batch_idx * batch_size + i + 1
                profiling_output += f"Sample {sample_index}: {transcription}\n"

            profiling_output += f"Batch {batch_idx+1}/{total_batches} processed in {batch_duration:.2f} seconds\n\n"

            # Calculate progress as a fraction (0 to 1) and yield intermediate results.
            progress_fraction = (batch_idx + 1) / total_batches
            yield profiling_output, progress_fraction
    elif inference_mode == "online":
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
        while next_arrival_index < num_samples or arrived_prompts:
            current_sim_time = time.time() - sim_start_time  # current simulation time (in sec)

            # Add requests that have "arrived" (i.e., their scheduled arrival time has passed).
            while next_arrival_index < num_samples and delay_prefixes[next_arrival_index].item() <= current_sim_time:
                arrived_prompts.append((prompts[next_arrival_index], delay_prefixes[next_arrival_index].item()))
                next_arrival_index += 1

            # If enough requests have accumulated for a batch, or if no more are coming, process a batch.
            if len(arrived_prompts) >= batch_size or (next_arrival_index >= num_samples and arrived_prompts):
                # Extract batch data.
                batch_data = arrived_prompts[:batch_size]
                arrived_prompts = arrived_prompts[batch_size:]
                batch_prompts = [item[0] for item in batch_data]
                batch_arrival_times = [item[1] for item in batch_data]  # these are relative times

                batch_start = time.time()
                outputs = llm.generate(batch_prompts, sampling_params)
                batch_finish_time = time.time()
                # Get simulation time at batch finish:
                sim_time_now = batch_finish_time - sim_start_time

                # Process each output in the batch.
                for i, output in enumerate(outputs):
                    transcription = output.outputs[0].text
                    # Calculate latency as the difference between the current simulation time and the scheduled arrival.
                    req_latency = sim_time_now - batch_arrival_times[i]
                    # Count tokens (using a simple whitespace split).
                    token_count = len(transcription.split())
                    if token_count == 0:
                        token_count = 1  # Avoid division by zero.
                    # Normalized latency: seconds per token.
                    norm_latency = req_latency / token_count
                    norm_latencies.append(norm_latency)

                    sample_index = processed_count + i + 1
                    profiling_output += f"Sample {sample_index}: {transcription}\n"
                    profiling_output += f"  Latency: {req_latency:.2f} s, Tokens: {token_count}, Norm Latency: {norm_latency:.2f} sec/token\n"
                processed_count += len(batch_prompts)
                profiling_output += f"Processed a batch of {len(batch_prompts)} requests in {(batch_finish_time - batch_start):.2f} seconds\n\n"

            # Report progress based on elapsed simulation time.
            progress_fraction = min(current_sim_time / simulation_end_time, 1.0)
            yield profiling_output, progress_fraction

            # Sleep briefly to avoid busy waiting.
            time.sleep(0.01)
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

    offline_btn.click(lambda: ("offline", f"Selected mode: Offline Inference"),
               None, [inference_mode, inference_display])
    
    online_btn.click(lambda: ("online", f"Selected mode: Online Inference"),
               None, [inference_mode, inference_display])
    
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
        inputs=[dataset_state, num_samples_input, batch_size_input, temperature_input, top_p_input, max_tokens_input, inference_mode, request_rate],
        outputs=[profiling_output_box, progress_bar],
        show_progress=True,
    )

demo.launch()

