import asyncio
from whisper_evaluation_server import run_whisper
import json

datasets = ['earnings22']
num_samples = 1000
request_rates = [5,10,15,20,25,30,35,40,45]
orders = ['shuffle', 'audio_length']
output = {}

async def main():
    for dataset in datasets:
        if dataset not in output:
            output[dataset] = {}

        for order in orders:
            if order not in output[dataset]:
                output[dataset][order] = {}

            for request_rate in request_rates:
                if request_rate not in output[dataset][order]:
                    output[dataset][order][request_rate] = {}

                async for latencies, avg_latency in run_whisper(
                        selected_dataset=dataset,
                        num_samples=num_samples,
                        batch_size=0,
                        temperature=0,
                        top_p=1,
                        max_tokens=500,
                        inference_mode='online',
                        request_rate=request_rate,
                        dataset_order=order,
                        scheduling_option='fifo'
                ):
                    output[dataset][order][request_rate]['latencies'] = latencies
                    output[dataset][order][request_rate]['avg_latency'] = avg_latency

    with open("earnings22_whisper_output.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Results saved to whisper_output.json")

asyncio.run(main())
