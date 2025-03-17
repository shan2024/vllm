import json
import matplotlib.pyplot as plt

# Load the JSON file
with open("ami_tail_latencies7.json", "r") as f:
    data = json.load(f)

# Extract latencies for each request rate
request_rates = data["ami"]["shuffle"].keys()
aging_factors = data["ami"]["shuffle"]["15"].keys()

for rate in request_rates:
    for aging_factor in aging_factors:
        
        
      latencies = data["ami"]["shuffle"][rate][aging_factor]["latencies"]
      avg_latency = data["ami"]["shuffle"][rate][aging_factor]["avg_latency"]
      max_latency = max(latencies)
      print(f"Aging factor: {aging_factor} has max: {max_latency} and avg: {avg_latency}")

      # Create a new figure for each request rate
      plt.figure(figsize=(8, 5))
      n, bins, patches = plt.hist(latencies, bins=30, edgecolor='black', alpha=0.7)

      plt.text(0.95, 0.95, 
                 f"Avg Latency  = {avg_latency:.2f} s\nMax Latency  = {max_latency:.2f} s",
                 transform=plt.gca().transAxes, fontsize=10, ha='right', va='top',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
                 multialignment='right')
      
      # Add labels on top of each bar
      for count, bin_edge in zip(n, bins[:-1]):
          if count > 0:  # Only label bars with samples
              plt.text(bin_edge + (bins[1] - bins[0]) / 2, count + 1, f"{int(count)}",
                      ha='center', fontsize=9, color='black')

      # Formatting
      plt.xlabel("Latency (s)")
      plt.ylabel("Frequency")
      plt.title(f"Request Latency Distribution (γ={aging_factor})")
      plt.grid(True, linestyle='--', alpha=0.6)
      plt.legend()
      
      # Save the plot as a PNG file
      plt.savefig(f"/home/paulh27/vllm/plots/ami13_latency_distribution_rate={rate}_age={aging_factor}.png")
      plt.close()

