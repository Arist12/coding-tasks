# ---- Stage 1: build image ----
FROM python:3.10-slim

# Install exactly what you need (tqdm and nothing else)
RUN --mount=type=cache,target=/root/.cache \
    pip install --no-cache-dir tqdm

# Copy the evaluation script into the image
WORKDIR /app
COPY src/eval_humaneval.py .

# Do not run as root
RUN adduser --disabled-password --home /home/eval runner
USER runner

# Default command
ENTRYPOINT ["python", "eval_humaneval.py"]