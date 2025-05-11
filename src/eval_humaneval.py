import io, json, signal, contextlib, multiprocessing, os
from typing import Dict
from tqdm import tqdm


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def _handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        yield


def unsafe_execute(code: str, timeout: float):
    try:
        with swallow_io(), time_limit(timeout):
            exec_globals = {}
            exec(code, exec_globals)
        return "passed"
    except TimeoutException:
        return "timed out"
    except Exception as e:
        return f"failed: {e}"


def check_correctness(
    problem: Dict, completion: str, timeout: float = 3.0, completion_id: int = 0
):
    check_program = (
        completion + "\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"
    )

    def target(result):
        import builtins

        builtins.exit = None
        builtins.quit = None
        result.append(unsafe_execute(check_program, timeout))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=target, args=(result,))
    p.start()
    p.join(timeout + 1)
    if p.is_alive():
        p.kill()
        result.append("timed out")
    if not result:
        result.append("timed out")
    execution_result = result[0]
    return {
        "task_id": problem["task_id"],
        "passed": execution_result == "passed",
        "result": execution_result,
        "completion_id": completion_id,
    }


if __name__ == "__main__":
    path = os.getenv(
        "HUMANEVAL_JSON", "../results/humaneval_results_multi_thread.jsonl"
    )

    with open(path, "r") as f:
        data = [json.loads(line) for line in f]

    passed = sum(
        check_correctness(
            item, item["completion"].split("```python")[1].split("```")[0]
        )["passed"]
        for item in tqdm(data)
        if "completion" in item and "```python" in item["completion"]
    )

    print(f"Pass@1: {passed / len(data):.4f}")
