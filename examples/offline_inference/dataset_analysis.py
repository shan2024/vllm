import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, get_dataset_config_names
from vad import estimate_speech_duration
import numpy as np

# Load Whisper model and tokenizer
model_name = "openai/whisper-large"
processor = WhisperProcessor.from_pretrained(model_name)

DATASET = "esb/diagnostic-dataset"  # dataset to load from
config_names = ['ami', 'earnings22', 'librispeech', 'voxpopuli']

global_prefill_lengths = []
global_decode_lengths = []

sns.set_style("whitegrid")
sns.set_context("paper")
plt.rcParams.update({"font.family": "serif", "font.size": 12})

for config in config_names:
    dataset = load_dataset(DATASET, config)
    local_decode_lengths = []
    gt_decode_lengths = []
    for audio_sample in dataset["clean"]:
        audio_array = audio_sample["audio"]["array"]
        ground_truth = audio_sample["ortho_transcript"]

        # Process text
        tokenized_gt = processor.tokenizer(ground_truth, return_tensors="pt")
        decode_length = tokenized_gt.input_ids.shape[1]  # Number of tokens
        gt_decode_lengths.append(decode_length)

        decode_length = estimate_speech_duration(audio_array)
        local_decode_lengths.append(decode_length)
        global_decode_lengths.append(decode_length)

    print(np.corrcoef(local_decode_lengths, gt_decode_lengths)[0,1])
    
    # Plot the distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(local_decode_lengths, bins=30, kde=False, color="royalblue", edgecolor='black')
    plt.xlabel("Audio Duration (sec)", fontsize=14)
    plt.ylabel("Number of Samples", fontsize=14)
    plt.title(f"{config} Audio Duration Distribution", fontsize=18, weight='bold', color='black')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"/home/paulh27/vllm/plots/{config}_decode_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

print("Plots saved successfully.")