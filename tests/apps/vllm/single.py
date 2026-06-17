import subprocess
from os import environ, getpid
from time import perf_counter

environ.update({
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NO_USAGE_STATS": "1",
})

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

MODEL = "/path/to/model"
PROMPT = "placeholder"
MAX_TOKENS = 320
CLI = "/workspace/target/release/server"
PID = getpid()

ENV = environ.copy()
ENV.pop("LD_PRELOAD")
ENV.pop("LD_LIBRARY_PATH", None)


def main():
    init_process_group()

    llm = LLM(
        MODEL,
        enable_sleep_mode=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        disable_log_stats=False,
        max_model_len=-1,
    )

    llm.sleep()
    run_checked("detach", "--client-pid", str(PID))

    start = perf_counter()
    run_checked("attach", "--client-pid", str(PID))
    llm.wake_up()
    print(f"Attach + wake up: {perf_counter() - start:.6f} s")

    outputs: list[RequestOutput] = llm.generate(PROMPT, SamplingParams(max_tokens=MAX_TOKENS))

    llm.llm_engine.engine_core.shutdown()


def init_process_group():
    """Prevent vLLM from using tcp:// init method."""

    import torch.distributed as dist

    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)


def run_checked(*args: str):
    subprocess.run((CLI, *args), env=ENV, check=True)


if __name__ == "__main__":
    main()
