from os import environ
from time import perf_counter

environ.update({
    "VLLM_NO_USAGE_STATS": "1",
})

from vllm import LLM

MODEL = "/path/to/model"


def main():
    llm = LLM(
        MODEL,
        enable_sleep_mode=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        disable_log_stats=False,
        max_model_len=-1,
    )

    llm.sleep()
    start = perf_counter()
    llm.wake_up()
    print(f"Wake up: {perf_counter() - start:.6f} s")

    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
