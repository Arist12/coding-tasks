import io
import json
import signal
import contextlib
import multiprocessing
from typing import Dict
from tqdm import tqdm


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    """
    Context manager to enforce time limit on code execution.
    """

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    """
    Context manager to capture stdout/stderr
    """
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            yield


def unsafe_execute(code: str, timeout: float):
    """
    Executes code and returns result.
    WARNING: This function executes untrusted code.
    """
    try:
        with swallow_io():
            with time_limit(timeout):
                # Create namespace for execution
                exec_globals = {}
                exec(code, exec_globals)
        return "passed"
    except TimeoutException:
        return "timed out"
    except Exception as e:
        return f"failed: {e}"


def check_correctness(
    problem: Dict, completion: str, timeout: float = 3.0, completion_id: int = 0
) -> Dict:
    """
    Evaluates the functional correctness of a completion by running tests.
    """

    # Construct the complete program to execute
    check_program = (
        completion + "\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"
    )

    def target(result):
        """Execute code in a separate process"""
        try:
            # Basic setup to prevent destructive actions
            import builtins

            builtins.exit = None
            builtins.quit = None

            # Execute with timeout
            result.append(unsafe_execute(check_program, timeout))
        except Exception as e:
            result.append(f"failed: {e}")

    # Create a manager for inter-process communication
    manager = multiprocessing.Manager()
    result = manager.list()

    # Run in separate process to isolate execution
    p = multiprocessing.Process(target=target, args=(result,))
    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.kill()
        result.append("timed out")

    # Parse result
    if not result:
        result.append("timed out")

    execution_result = result[0]
    passed = execution_result == "passed"

    return {
        "task_id": problem["task_id"],
        "passed": passed,
        "result": execution_result,
        "completion_id": completion_id,
    }


# Example usage
if __name__ == "__main__":
    # Example problem

    with open("../results/humaneval_results_multi_thread.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    pass_at_1 = 0
    for problem in tqdm(data):
        try:
            completion = problem["completion"].split("```python")[1].split("```")[0]
        except Exception as e:
            continue

        # Test the completion
        result = check_correctness(problem, completion)
        if result["passed"]:
            pass_at_1 += 1

    print(f"Pass@1: {pass_at_1 / len(data)}")
