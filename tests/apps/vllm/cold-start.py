from os import environ
from time import perf_counter

environ.update({
    "VLLM_NO_USAGE_STATS": "1",
})

from vllm import LLM, SamplingParams

MODEL = "/models/Qwen3-0.6B"

# Prompts to sanity-check that the model emits sensible tokens
PROMPTS = [
    "The capital of France is",
    "Q: What is 2 + 2? A:",
]
MAX_TOKENS = 64


def main():
    start = perf_counter()
    llm = LLM(
        MODEL,
        tensor_parallel_size=1,
        enforce_eager=True,
        disable_log_stats=False,
        max_model_len=-1,
    )
    print(f"Cold start: {perf_counter() - start:.6f} s")

    # Greedy decode so output is deterministic and easy to eyeball
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    gen_start = perf_counter()
    outputs = llm.generate(PROMPTS, sampling_params)
    gen_dt = perf_counter() - gen_start

    print("\n==================== GENERATION ====================")
    for out in outputs:
        gen = out.outputs[0]
        print(f"PROMPT : {out.prompt!r}")
        print(f"OUTPUT : {gen.text!r}")
        print(f"#TOKENS: {len(gen.token_ids)}  finish={gen.finish_reason}")
        print("----------------------------------------------------")
    print(f"Generation: {gen_dt:.3f} s")
    print("====================================================\n")

    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
