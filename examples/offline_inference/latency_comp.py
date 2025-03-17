import json
import matplotlib.pyplot as plt

with open(f"librispeech_whisper_output.json", "r") as f:
    output = json.load(f)

for dataset_key in output.keys():
    request_rates = sorted(output[dataset_key]["fifo"].keys(), key=int)  # Sort request rates
    fifo_latencies = [output[dataset_key]["fifo"][str(rate)]['avg_latency'] for rate in request_rates]
    priority_latencies = [output[dataset_key]["priority"][str(rate)]['avg_latency'] for rate in request_rates]

    request_rates = list(map(int, request_rates))

    plt.figure(figsize=(7, 5))
    plt.plot(request_rates, fifo_latencies, marker='o', linestyle='-', color='royalblue', label='FIFO')
    plt.plot(request_rates, priority_latencies, marker='s', linestyle='-', color='darkorange', label='Priority')

    # Labels and title
    plt.xlabel("Request Rate (req/s)", fontsize=14)
    plt.ylabel("Average Latency (s)", fontsize=14)
    plt.title(f"{dataset_key} Latency Comparison", fontsize=16, weight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"/home/paulh27/vllm/plots/final_{dataset_key}_latency_comparison.png", dpi=300)