import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, get_dataset_config_names

# Load Whisper model and tokenizer
model_name = "openai/whisper-large"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Example dataset (list of audio files)
# dataset = ["path/to/audio1.wav", "path/to/audio2.wav"]  # Replace with actual dataset

global_prefill_lengths = []
global_decode_lengths = []

DATASET = "esb/diagnostic-dataset"  # dataset to load from
config_names = get_dataset_config_names(DATASET)

for config in config_names:
    dataset = load_dataset(DATASET, config, download_mode="force_redownload")
    
    local_prefill_lengths = []
    local_decode_lengths = []  

     # Store samples with their decode lengths
    samples_with_lengths = []

    for audio_sample in dataset["clean"]:
        audio_array = audio_sample["audio"]['array']
        ground_truth = audio_sample["ortho_transcript"]

        # Process audio
        audio_input = processor(audio_array, return_tensors="pt", sampling_rate=16000)
        
        # Get encoder token length (Prefill)
        encoder_input = audio_input["input_features"]  # Shape: (1, 80, T)
        print(encoder_input)
        exit()
        local_prefill_lengths.append(encoder_input.shape[-1])  # Time-frame length
        global_prefill_lengths.append(encoder_input.shape[-1])  # Time-frame length

        tokenized_gt = processor.tokenizer(ground_truth, return_tensors="pt")
        local_decode_lengths.append(tokenized_gt.input_ids.shape[1])  # Number of tokens
        global_decode_lengths.append(tokenized_gt.input_ids.shape[1])  # Number of tokens

        decode_length = tokenized_gt.input_ids.shape[1]  # Number of tokens
        samples_with_lengths.append((audio_sample, decode_length))

    sorted_dataset = sorted(samples_with_lengths, key=lambda x: x[1])

    for audio_sample, decode_length in sorted_dataset:
        print(audio_sample["ortho_transcript"])
    break

    avg_prefill_length = sum(local_prefill_lengths) / len(local_prefill_lengths)
    avg_decode_length = sum(local_decode_lengths) / len(local_decode_lengths)

    print(f"{config} Average Prefill Length: {avg_prefill_length}")
    print(f"{config} Average Decode Length: {avg_decode_length}")

# Compute averages
avg_prefill_length = sum(global_prefill_lengths) / len(global_prefill_lengths)
avg_decode_length = sum(global_decode_lengths) / len(global_decode_lengths)

print(f"Global Average Prefill Length: {avg_prefill_length}")
print(f"Global Average Decode Length: {avg_decode_length}")