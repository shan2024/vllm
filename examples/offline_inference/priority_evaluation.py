import asyncio
from whisper_evaluation_server import run_whisper
import json

datasets = ['ami']
num_samples = 1000
request_rates = [15]
orders = ['shuffle']
aging_factors = [0.0, 0.0001 ,0.001,0.01, 0.1]
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

                for aging_factor in aging_factors:
                    if aging_factor not in output[dataset][order][request_rate]:
                      output[dataset][order][request_rate][aging_factor] = {}

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
                            scheduling_option='priority',
                            bootstrap=False,
                            min_queue_len=10,
                            aging_factor=aging_factor
                    ):
                        output[dataset][order][request_rate][aging_factor]['latencies'] = latencies
                        output[dataset][order][request_rate][aging_factor]['avg_latency'] = avg_latency

    with open(f"ami_tail_latencies7.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Results saved to whisper_output.json")
asyncio.run(main())