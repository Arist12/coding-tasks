import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils import (
    DEFAULT_PORT,
    DEFAULT_MODEL_NAME,
    init_tokenizer,
    load_humaneval_data,
    process_sample,
    save_results,
)


def main():
    # Initialize configuration
    port = DEFAULT_PORT
    url = f"http://localhost:{port}/generate"
    model_name = DEFAULT_MODEL_NAME
    max_workers = 164

    # Initialize the tokenizer
    init_tokenizer(model_name)

    # Load HumanEval dataset
    humaneval_data = load_humaneval_data()

    # Process samples using ThreadPoolExecutor
    results = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_sample, sample, url) for sample in humaneval_data
        ]

        # Collect results as they complete
        for future in tqdm(futures, total=len(futures)):
            results.append(future.result())

    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")

    # Save the results
    save_results(results, output_file="../results/humaneval_results_multi_thread.jsonl")


if __name__ == "__main__":
    main()
