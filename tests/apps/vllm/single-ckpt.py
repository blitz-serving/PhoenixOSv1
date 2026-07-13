import subprocess
from os import environ, getpid

environ.update({
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NO_USAGE_STATS": "1",
})

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

MODEL = "/models/Qwen3-0.6B"
PROMPT = "Hello, I'm a language model,"
MAX_TOKENS = 320
CLI = "/workspace/target/release/server"
PID = getpid()
CKPT_DIR = "/workspace/ckpt"

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

    run("checkpoint", "--ckpt-dir", CKPT_DIR)

    llm.wake_up()

    outputs: list[RequestOutput] = llm.generate(PROMPT, SamplingParams(max_tokens=MAX_TOKENS))
    for output in outputs:
        print(f"Prompt: {output.prompt!r}\n")
        print(f"Generated text: {output.outputs[0].text}")

    llm.llm_engine.engine_core.shutdown()


def init_process_group():
    """Prevent vLLM from using tcp:// init method."""

    import torch.distributed as dist

    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)


def run(*args: str):
    subprocess.run((CLI, *args), env=ENV)


if __name__ == "__main__":
    main()
