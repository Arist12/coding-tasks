import time
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
    port = DEFAULT_PORT  # 19999
    url = f"http://localhost:{port}/generate"
    model_name = DEFAULT_MODEL_NAME  # "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    # Initialize the tokenizer
    init_tokenizer(model_name)

    # Load HumanEval dataset
    humaneval_data = load_humaneval_data()

    # Process each sample sequentially
    results = []
    start_time = time.time()
    for sample in tqdm(humaneval_data):
        result = process_sample(sample, url)
        results.append(result)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # Save the results
    save_results(results, output_file="../results/humaneval_results_seq.jsonl")


if __name__ == "__main__":
    main()
