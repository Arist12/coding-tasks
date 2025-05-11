# Serving Qwen2.5-Coder with SGLang

This repo includes Python scripts to serve, inference, and evaluate `Qwen/Qwen2.5-Coder-0.5B-Instruct` with Docker containers.

## 1. Serve the model with SGLang

The `./src/cli.py` script automates the setup of a Docker container that launches an SGLang server to serve a user-specified model (e.g., for this repo, we default to serve `Qwen2.5-Coder-0.5B-Instruct`).

### Running the script

To serve the model, navigate to the `./src` directory and run the script:

```bash
python cli.py [OPTIONS]
```

#### Options:

The script accepts several command-line arguments to customize its behavior:

-   `--container_name`: Name of the Docker container (default: `sglang_yikai`).
-   `--model_path`: Path or Hugging Face repo ID of the model to serve (default: `Qwen/Qwen2.5-Coder-0.5B-Instruct`).
-   `--port`: Port to serve the model on (default: `19998`).
-   `--sglang_branch`: Branch of the SgLang repository to clone (default: `v0.4.6.post2` - the latest version at the time of implementation).
-   `--gpu_id`: GPU ID to use for serving the model (default: `1`).
-   `--proxy_host`: HTTP/HTTPS proxy to use inside the container (default: `172.17.0.1:1081`).

For example, to run with default settings:

```bash
python cli.py
```

To specify a different GPU and port:

```bash
python cli.py --gpu_id 0 --port 20000
```

This will pull the necessary Docker image, set up the SGLang environment, and start the model server on the specified GPU and port.

Now, we can greet the model by sending a request to the model server on the specified port:

```python
import requests

port = 19999 # we will use this port by default in this repo
url = f"http://localhost:{port}/generate"
data = {"text": "Hi, how are you?"}

response = requests.post(url, json=data)
print(response.json())
```

### Comments

You might find it weird that extra steps are taken to install Python 3.10 and set up a virtual environment before serving the model.

However, this is an indispendable step because the official SGLang Docker doesn't have SGLang dependencies correctly installed (during the development process, I found that both `lmsysorg/sglang:dev` and `lmsysorg/sglang:latest` will report missing `torchvision` error if directly executing `sglang.launch_server` command).

```bash
export http_proxy=http://{args.proxy_host}
export https_proxy=http://{args.proxy_host}

cd /sgl-workspace &&
rm -rf sglang &&
git clone -b {args.sglang_branch} https://github.com/sgl-project/sglang.git &&
cd sglang &&

apt update &&
apt install -y python3.10 python3.10-venv &&

python3 -m venv ~/.python/sglang &&
source ~/.python/sglang/bin/activate &&

pip install uv &&
uv pip install --upgrade pip &&
uv pip install -e 'python[all]' &&

CUDA_VISIBLE_DEVICES={args.gpu_id} python3 -m sglang.launch_server \\
    --model-path {args.model_path} \\
    --host 0.0.0.0 \\
    --port {args.port}
```

## 2. Model Inference on HumanEval

HumanEval is a code generation task that consists of 164 manually written programming tasks, each providing a Python function signature and a docstring as input to the model.

The `./src/run_humaneval.py` script is used to inference the model on the HumanEval dataset. The output will be saved to `./results/humaneval_results.jsonl` by default.

### Running the script

We follow the [official hugging face settings](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) for `Qwen/Qwen2.5-Coder-0.5B-Instruct` to set chat template and sampling parameters during the inference.

We set `max_new_tokens` to 512 following the [Qwen2.5-Coder GitHub repo](https://github.com/QwenLM/Qwen2.5-Coder).

Here's a brief example of how the chat template and sampling parameters are set:

```python
messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    },
    {
        "role": "user",
        "content": prompt
    },
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
    },
}

response = requests.post(url, json=data).json()["text"]
```

To run the script, navigate to the `./src` directory and run the sequential inference script:

```bash
python run_humaneval_seq.py
```

The results will be saved to `./results/humaneval_results_seq.jsonl` by default.

### Updates

We modularized the code to store utility functions for the whole model inference process in `./src/utils.py`.

To maximize evaluation throughput, we also implemented parallel versions of the inference script, which will be detailed in section 4.

## 3. Evaluation with `evalplus`

First, install the `evalplus` package with the default `vllm` backend:

```bash
pip install "evalplus[vllm]" --upgrade
```

Now, we can sanitize the results and evaluate the model with the `evalplus` package:

```bash
evalplus.sanitize --samples ./results/humaneval_results.jsonl
```

With the sanitized results, we can evaluate the model with the `evalplus` package:

```bash
evalplus.evaluate --dataset humaneval --samples ./results/humaneval_results-sanitized.jsonl
```

If you are not in a Docker container, you can also evaluate the model with the `evalplus` package in a docker container by running the following command:

```bash
docker run --rm --pull=always -v $(pwd):/app/local_data ganler/evalplus:latest \
    evalplus.evaluate --dataset humaneval \
    --samples /app/local_data/results/humaneval_results-sanitized.jsonl
```

| Model         | Our Result | Reported Result |
|---------------|------------|-----------------|
| Qwen2.5-Coder | 23.8       | 61.6            |

## 4. Performance Improvement

Original Speed: 136.7s
