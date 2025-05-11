import os
import json
import requests
from transformers import AutoTokenizer
from typing import List, Dict, Any

# Default configs
DEFAULT_PORT = 19999
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = None


def init_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> None:
    """Initialize the tokenizer for the specified model."""
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_tokenizer() -> AutoTokenizer:
    """Get the tokenizer, initializing it if needed."""
    global tokenizer
    if tokenizer is None:
        init_tokenizer()
    return tokenizer


def load_humaneval_data(
    file_path: str = "../datasets/humaneval/HumanEval.jsonl",
) -> List[Dict[str, Any]]:
    """Load HumanEval dataset from a jsonl file."""
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def prepare_request_data(prompt: str) -> str:
    """Format a prompt with the chat template, and set sampling params."""
    global tokenizer
    if tokenizer is None:
        tokenizer = get_tokenizer()

    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return {
        "text": formatted_prompt,
        "sampling_params": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "frequency_penalty": 1.05,
            "stop_token_ids": [151645, 151643],
        },
    }


def save_results(
    results: List[Dict[str, str]],
    output_file: str = "../results/humaneval_results.jsonl",
) -> None:
    """Save results to a specified jsonl file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def process_sample(sample: Dict[str, Any], url: str) -> Dict[str, str]:
    """Process a single sample by sending it to the model server and return formatted result."""
    prompt = sample["prompt"]
    data = prepare_request_data(prompt)
    response = requests.post(url, json=data).json()["text"]

    return {
        "task_id": sample["task_id"],
        "completion": response,
    }
