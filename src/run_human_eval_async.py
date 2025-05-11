import time
import asyncio
import aiohttp
from tqdm import tqdm
from utils import (
    DEFAULT_PORT,
    DEFAULT_MODEL_NAME,
    init_tokenizer,
    load_humaneval_data,
    prepare_request_data,
    save_results,
)


async def async_send_request(session, url, data):
    """Send a request asynchronously to the model API and return the text response."""
    async with session.post(url, json=data) as response:
        response_data = await response.json()
        return response_data["text"]


async def async_process_sample(session, sample, url):
    """Process a single sample asynchronously."""
    prompt = sample["prompt"]
    data = prepare_request_data(prompt)
    response = await async_send_request(session, url, data)

    return {
        "task_id": sample["task_id"],
        "completion": response,
        "test": sample["test"],
        "entry_point": sample["entry_point"],
    }


async def main():
    # Initialize configuration
    port = DEFAULT_PORT
    url = f"http://localhost:{port}/generate"
    model_name = DEFAULT_MODEL_NAME

    # Initialize the tokenizer
    init_tokenizer(model_name)

    # Load HumanEval dataset
    humaneval_data = load_humaneval_data()

    # Process samples using asyncio
    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [
            async_process_sample(session, sample, url) for sample in humaneval_data
        ]

        # Use tqdm to show progress with asyncio.as_completed
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await task
            results.append(result)

    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")

    # Save the results
    save_results(results, output_file="../results/humaneval_results_async.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
