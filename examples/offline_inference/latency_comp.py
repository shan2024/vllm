import json
import matplotlib.pyplot as plt

# Load the JSON file
with open("whisper_output.json", "r") as f:
    output = json.load(f)

dataset = "ami"  # Assuming only one dataset
orders = ["shuffle", "audio_length"]
request_rates = sorted(output[dataset]["shuffle"].keys(), key=int)  # Sort request rates

# Extract average latencies
shuffle_latencies = [output[dataset]["shuffle"][str(rate)]['avg_latency'] for rate in request_rates]
audio_length_latencies = [output[dataset]["audio_length"][str(rate)]['avg_latency'] for rate in request_rates]

# Convert request rates to integers for plotting
request_rates = list(map(int, request_rates))

# Plot the data
plt.figure(figsize=(7, 5))
plt.plot(request_rates, shuffle_latencies, marker='o', linestyle='-', color='royalblue', label='Shuffle')
plt.plot(request_rates, audio_length_latencies, marker='s', linestyle='-', color='darkorange', label='Audio Length')

# Labels and title
plt.xlabel("Request Rate", fontsize=14)
plt.ylabel("Average Latency (ms)", fontsize=14)
plt.title("Latency Comparison: Shuffle vs. Audio Length", fontsize=16, weight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot
plt.tight_layout()
plt.savefig(f"/home/paulh27/vllm/plots/{dataset}_latency_comparison.png", dpi=300)
plt.show()
