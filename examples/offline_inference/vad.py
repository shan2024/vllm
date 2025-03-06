import torch
import numpy as np
import librosa
import webrtcvad
import struct
from datasets import load_dataset
from transformers import WhisperProcessor
from scipy.signal import butter, filtfilt
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def high_pass_filter(audio, sr, cutoff=100):
    """Apply a Butterworth high-pass filter to remove low-frequency noise."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(6, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, audio)

processor = WhisperProcessor.from_pretrained("openai/whisper-large").to(device)  # Move processor to GPU
DATASET = "esb/diagnostic-dataset"
SAMPLING_RATE = 16000
dataset = load_dataset(DATASET, "ami")

def tensor_to_audio(tensor, sr=16000):
    """Convert tensor to NumPy array and normalize, and move to GPU."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.to(device)  # Move to GPU
    audio = tensor.cpu().numpy() if hasattr(tensor, 'numpy') else tensor
    return librosa.util.normalize(audio), sr

@torch.jit.script
def apply_vad(audio: torch.Tensor, sr: int, vad_mode: int = 3) -> float:
    """Use WebRTC VAD to detect voiced frames."""
    vad = webrtcvad.Vad(vad_mode)
    frame_length = 30  # 30ms per frame
    frame_size = int(sr * frame_length / 1000)  # Convert to samples

    speech_segments = []
    audio_np = audio.cpu().numpy()  # Move audio back to CPU for WebRTC processing
    for i in range(0, len(audio_np) - frame_size, frame_size):
        frame = audio_np[i:i + frame_size]
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
    
    # Convert to Torch Tensor and Move to GPU
    filtered_audio_tensor = torch.tensor(filtered_audio, dtype=torch.float32, device=device)
    
    # Apply VAD (runs on CPU because of WebRTC limitation)
    speech_duration = apply_vad(filtered_audio_tensor, sr)
    return speech_duration

start = time.time()
for audio_sample in dataset["clean"]:
    ground_truth = audio_sample["ortho_transcript"]
    
    # Move text processing to GPU
    tokenized_gt = processor.tokenizer(ground_truth, return_tensors="pt").to(device)
    
    decode_length = tokenized_gt.input_ids.shape[1]  # Number of tokens
    print(f"Decode Length: {decode_length}")

    # Move audio tensor to GPU
    audio_tensor = torch.tensor(audio_sample["audio"]['array'], dtype=torch.float32, device=device)

    duration = estimate_speech_duration(audio_tensor)

    print(f"Estimated Duration: {duration}")
end = time.time()

print(f"Total duration: {end - start} seconds")

