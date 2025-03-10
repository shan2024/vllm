import gradio as gr
import torch
from datasets import load_dataset, get_dataset_config_names
import time
from math import ceil
from vllm import LLM, SamplingParams, AsyncLLMEngine
from vllm.config import ModelConfig, SchedulerConfig

import numpy as np
import matplotlib.pyplot as plt
import os
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio
import random
from vad import estimate_speech_duration


# Enable torch profiler
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set this to the GPU num that of a GPU not in use
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Global settings
DATASET = "esb/diagnostic-dataset"
SAMPLING_RATE = 16000

async_engine_args = AsyncEngineArgs(
    model="openai/whisper-large-v3",
    max_model_len=448,
    max_num_seqs=16,
    limit_mm_per_prompt={"audio": 1},
    kv_cache_dtype="fp8",
    scheduling_policy='priority'
)

llm = AsyncLLMEngine.from_engine_args(async_engine_args)

processor = WhisperProcessor.from_pretrained("openai/whisper-large")

# Get the list of dataset configurations (we assume there are at least 8)
config_names = list(get_dataset_config_names(DATASET))[:8]


########################################################################
# Stat computation helper functions
########################################################################

def analyze_token_lengths(lengths, dataset=""):
    """
    lengths: List of output token decode lengths
    """
    # Compute statistics
    median_latency = np.median(lengths)
    mean_latency = np.mean(lengths)
    std_latency = np.std(lengths)

    output = f"Median Length: {median_latency:.3f} tokens\n" \
        + f"Mean Length: {mean_latency:.3f} tokens\n" \
        + f"Standard Deviation Length: {std_latency:.3f} tokens"

    # Plot histogram of latencies
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=100, alpha=0.7, color="blue", edgecolor="black")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title("Token Length Distribution")
    plt.savefig(f"assets/dataset_distribution_{dataset}.png", dpi=600)
    plt.show()

    return output


# Helper function to get dataset token length distribution
def get_token_length_distribution(dataset, selected_dataset):
    lengths = []
    for audio_sample in dataset["clean"]:
        ground_truth = audio_sample["ortho_transcript"]
        tokenized_gt = processor.tokenizer(ground_truth, return_tensors="pt")
        decode_length = tokenized_gt.input_ids.shape[1]  # Number of tokens
        lengths.append(decode_length)

    return analyze_token_lengths(lengths, selected_dataset)

########################################################################
# This generator function loads the chosen dataset and runs the model
# in batches. It yields a tuple of (profiling text, progress fraction)
########################################################################
async def run_whisper(selected_dataset):
    if not selected_dataset:
        yield "Error: No dataset selected.", 0
        return

    dataset = load_dataset(DATASET, selected_dataset)
    output = get_token_length_distribution(dataset, selected_dataset)
    yield output, 1.0

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
        inputs=[dataset_state],
        outputs=[profiling_output_box, progress_bar],
        show_progress=True,
    )

demo.launch(share=True)

