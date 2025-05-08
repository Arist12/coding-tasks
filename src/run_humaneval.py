import os
import json
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

# configs
port = 19999
url = f"http://localhost:{port}/generate"
model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


# load HumanEval dataset
with open("../datasets/humaneval/HumanEval.jsonl", "r") as f:
    humaneval_data = [json.loads(line) for line in f]

# load Qwen2.5-Coder-0.5B-Instruct tokenizer to encode the prompt
tokenizer = AutoTokenizer.from_pretrained(model_name)


# save the results to `./results/humaneval_results.jsonl` (create the folder if it doesn't exist)
os.makedirs("../results", exist_ok=True)

with open("../results/humaneval_results.jsonl", "w") as f:
    for sample in tqdm(humaneval_data):
        prompt = sample["prompt"]
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

        data = {
            "text": formatted_prompt,
            "sampling_params": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "frequency_penalty": 1.05,
            },
        }

        response = requests.post(url, json=data).json()["text"]

        completed_task = {
            "task_id": sample["task_id"],
            "completion": response,
        }

        f.write(json.dumps(completed_task) + "\n")
