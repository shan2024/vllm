import numpy as np
import librosa
import webrtcvad
import struct
from datasets import load_dataset, get_dataset_config_names
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from scipy.signal import butter, filtfilt
import time 
import asyncio

def high_pass_filter(audio, sr, cutoff=100):
    """Apply a Butterworth high-pass filter to remove low-frequency noise."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(6, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, audio)

processor = WhisperProcessor.from_pretrained("openai/whisper-large")
DATASET = "esb/diagnostic-dataset"
SAMPLING_RATE = 16000
dataset = load_dataset(DATASET, "ami")


def tensor_to_audio(tensor, sr=16000):
    """Convert tensor to NumPy array and normalize."""
    audio = tensor.numpy() if hasattr(tensor, 'numpy') else tensor
    return librosa.util.normalize(audio), sr

# def high_pass_filter(audio, sr, cutoff=100):
#     """Apply a high-pass filter to remove low-frequency noise."""
#     sos = librosa.filters.get_window(('butter', 6), len(audio))
#     return librosa.effects.preemphasis(audio, coef=0.97)

def apply_vad(audio, sr, vad_mode=3):
    """Use WebRTC VAD to detect voiced frames."""
    vad = webrtcvad.Vad(vad_mode)
    frame_length = 30  # 30ms per frame
    frame_size = int(sr * frame_length / 1000)  # Convert to samples

    speech_segments = []
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size]
        if len(frame) < frame_size:
            continue
        pcm_frame = struct.pack(f"{len(frame)}h", *(np.int16(frame * 32768)))
        if vad.is_speech(pcm_frame, sr):
            speech_segments.append(frame_length / 1000)

    return sum(speech_segments)  # Estimated speech duration

def estimate_speech_duration(tensor, sr=16000):
    """Estimate the duration of transcribed speech from a raw audio tensor."""
    audio, sr = tensor_to_audio(tensor, sr)
    filtered_audio = high_pass_filter(audio, sr)
    speech_duration = apply_vad(filtered_audio, sr)
    return speech_duration

# start = time.time()
# for audio_sample in dataset["clean"]:
#     # ground_truth = audio_sample["ortho_transcript"]
#     # tokenized_gt = processor.tokenizer(ground_truth, return_tensors="pt")
#     # decode_length = tokenized_gt.input_ids.shape[1]  # Number of tokens
#     # print(f"Decode Length: {decode_length}")

#     duration = estimate_speech_duration(audio_sample["audio"]['array'])

#     print(f"Estimated Duration: {duration}")
# end = time.time()

# print(f"Total duration: {end - start}")
