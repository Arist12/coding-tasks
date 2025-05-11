import json
import os
import tempfile
import subprocess
import docker
from typing import Dict, Any
from tqdm import tqdm


class DockerCodeEvaluator:
    def __init__(self, timeout: float = 3.0):
        self.timeout = timeout
        self.client = docker.from_env()

        # Create a minimal Python Docker image
        self.setup_docker_image()

    def setup_docker_image(self):
        """Set up a minimal Docker image for code execution"""
        dockerfile_content = """
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Remove any unnecessary programs and files for security
RUN pip install --no-cache-dir pytest

# Create non-root user
RUN useradd -m -s /bin/bash coderunner

# Switch to non-root user
USER coderunner

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
"""

        # Check if our image already exists
        try:
            self.client.images.get("humaneval:safe")
            print("Docker image 'humaneval:safe' already exists")
        except:
            print("Building Docker image...")
            # Create temporary Dockerfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix="Dockerfile", delete=False
            ) as f:
                f.write(dockerfile_content)
                dockerfile_path = f.name

            # Build the image
            self.client.images.build(
                path=os.path.dirname(dockerfile_path),
                dockerfile=os.path.basename(dockerfile_path),
                tag="humaneval:safe",
                rm=True,
            )
            os.unlink(dockerfile_path)
            print("Docker image built successfully")

    def run_code_in_container(self, code: str) -> Dict[str, Any]:
        """Run code in a Docker container and return results"""
        try:
            # Create test program
            test_program = f"""
import sys
import json

try:
    # Execute the code
    exec('''
{code}
''')
    result = {{"status": "PASSED", "error": null}}
except Exception as e:
    result = {{"status": "FAILED", "error": str(e)}}

# Output result as JSON
print(json.dumps(result))
"""

            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(test_program)
                temp_file = f.name

            # Mount the file in container and run it
            container = self.client.containers.run(
                "humaneval:safe",
                f"python /app/test.py",
                volumes={temp_file: {"bind": "/app/test.py", "mode": "ro"}},
                remove=True,
                stdout=True,
                stderr=True,
                working_dir="/app",
                network_disabled=True,  # Disable network access
                mem_limit="256m",  # Limit memory
                timeout=self.timeout,
                detach=False,
            )

            # Parse output
            output = container.decode("utf-8")
            try:
                result = json.loads(output.strip())
            except:
                result = {"status": "FAILED", "error": "Invalid output format"}

            os.unlink(temp_file)
            return result

        except subprocess.TimeoutExpired:
            os.unlink(temp_file)
            return {"status": "TIMEOUT", "error": "Code execution timed out"}
        except Exception as e:
            if "temp_file" in locals():
                os.unlink(temp_file)
            return {"status": "FAILED", "error": str(e)}

    def check_correctness(self, problem: Dict, completion: str) -> Dict:
        """Check if a completion passes the tests"""
        # Construct the complete program to execute
        check_program = (
            completion
            + "\n"
            + problem["test"]
            + "\n"
            + f"check({problem['entry_point']})"
        )

        # Run in container
        result = self.run_code_in_container(check_program)

        # Convert to expected format
        return {
            "task_id": problem["task_id"],
            "passed": result["status"] == "PASSED",
            "result": result["status"].lower(),
            "error": result.get("error"),
        }

    def evaluate_pass_at_1(self, results_file: str) -> float:
        """Evaluate pass@1 score from results file"""
        # Load results
        with open(results_file, "r") as f:
            data = [json.loads(line) for line in f]

        passed_count = 0
        total_count = 0

        print(f"Evaluating {len(data)} problems in Docker containers...")

        for problem in tqdm(data, desc="Testing completions"):
            # Extract code from markdown if present
            try:
                if "```python" in problem["completion"]:
                    completion = (
                        problem["completion"].split("```python")[1].split("```")[0]
                    )
                else:
                    completion = problem["completion"]
            except:
                continue

            # Test the completion
            result = self.check_correctness(problem, completion)

            if result["passed"]:
                passed_count += 1

            total_count += 1

        # Calculate pass@1
        pass_at_1 = passed_count / total_count if total_count > 0 else 0

        print(f"\nResults:")
        print(f"Total problems: {total_count}")
        print(f"Passed: {passed_count}")
        print(f"Pass@1: {pass_at_1:.4f}")

        return pass_at_1

    def cleanup(self):
        """Clean up Docker resources"""
        try:
            self.client.close()
        except:
            pass


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = DockerCodeEvaluator(timeout=3.0)

    try:
        # Run evaluation
        pass_at_1 = evaluator.evaluate_pass_at_1("../results/he_results.jsonl")
        print(f"Final Pass@1 score: {pass_at_1:.4f}")
    finally:
        evaluator.cleanup()
