import asyncio
from whisper_evaluation_server import run_whisper
import json

datasets = ['ami', 'earnings22', 'librispeech', 'voxpopuli']
num_samples = 1000
request_rates = [16,18,20,22,24,26,28]
scheduling_options = ['fifo', 'priority']
output = {}

async def main():
    for dataset in datasets:
        if dataset not in output:
            output[dataset] = {}

        for option in scheduling_options:
            if option not in output[dataset]:
                output[dataset][option] = {}

            for request_rate in request_rates:
                if request_rate not in output[dataset][option]:
                    output[dataset][option][request_rate] = {}

                async for latencies, avg_latency in run_whisper(
                        selected_dataset=dataset,
                        num_samples=num_samples,
                        batch_size=0,
                        temperature=0,
                        top_p=1,
                        max_tokens=500,
                        inference_mode='online',
                        request_rate=request_rate,
                        dataset_order='shuffle',
                        scheduling_option=option,
                        min_queue_len=10
                ):
                    output[dataset][option][request_rate]['latencies'] = latencies
                    output[dataset][option][request_rate]['avg_latency'] = avg_latency

    with open("final_part2_whisper_output.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Results saved to whisper_output.json")
asyncio.run(main())
