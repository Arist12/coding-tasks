import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Launch SGLang to serve LLMs")
    parser.add_argument("--container_name", default="sglang_yikai", help="Name of the Docker container")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Model to serve")
    parser.add_argument("--port", type=int, default=19999, help="Port to serve the model on")
    parser.add_argument("--proxy_host", default="172.17.0.1:1081", help="HTTP/HTTPS proxy to use in container")
    parser.add_argument("--sglang_branch", default="v0.4.6.post2", help="Branch of sglang to clone")
    parser.add_argument("--gpu_id", default="1", help="GPU id to use")

    args = parser.parse_args()

    docker_command = f'''
    docker run \
        -it \
        --shm-size 32g \
        --gpus all \
        -v /models/shared/.cache:/root/.cache \
        --ipc=host \
        --network=host \
        --privileged \
        --name {args.container_name} \
        lmsysorg/sglang:dev \
        /bin/bash -c "
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
        "
    '''

    subprocess.run(docker_command, shell=True, check=True)

if __name__ == "__main__":
    main()
