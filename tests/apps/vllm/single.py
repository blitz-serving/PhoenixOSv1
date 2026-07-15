import subprocess
from os import environ, getpid
from time import perf_counter

environ.update({
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NO_USAGE_STATS": "1",
})

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

MODEL = "/models/Qwen3-0.6B"
PROMPT = "Hello, I'm a language model,"
MAX_TOKENS = 64
CLI = "/workspace/target/release/server"
PID = getpid()

ENV = environ.copy()
ENV.pop("LD_PRELOAD")
ENV.pop("LD_LIBRARY_PATH", None)

# Greedy so the before/after-restore outputs are directly comparable.
SAMPLING = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)


def generate(llm, tag):
    outputs: list[RequestOutput] = llm.generate(PROMPT, SAMPLING)
    gen = outputs[0].outputs[0]
    print(f"[{tag}] #TOKENS={len(gen.token_ids)} finish={gen.finish_reason} OUTPUT={gen.text!r}")
    return gen.text


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

    # 1) inference BEFORE sleep/detach
    before = generate(llm, "before-sleep")

    # 2) sleep -> detach (frees the GPU worker) -> attach (new worker) -> wake
    llm.sleep()
    run_checked("detach", "--client-pid", str(PID))

    start = perf_counter()
    run_checked("attach", "--client-pid", str(PID))
    llm.wake_up()
    print(f"Attach + wake up: {perf_counter() - start:.6f} s")

    # 3) inference AFTER restore
    after = generate(llm, "after-restore")

    # 4) verify restore preserved the model (greedy => must be identical)
    print("\n==================== PhOSv1 VERIFY ====================")
    print(f"MATCH (before == after): {before == after}")
    if before != after:
        print("WARNING: output changed after restore -- state not preserved!")
    print("======================================================\n")

    llm.llm_engine.engine_core.shutdown()


def init_process_group():
    """Prevent vLLM from using tcp:// init method."""
    import torch.distributed as dist
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)


def run_checked(*args: str):
    subprocess.run((CLI, *args), env=ENV, check=True)


if __name__ == "__main__":
    main()
