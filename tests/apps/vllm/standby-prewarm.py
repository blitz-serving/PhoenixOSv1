"""Pre-warm phase for the PhOS context service (run this FIRST, keep it running).

Loads a model, generates once, then parks it: sleep -> detach -> standby.
The process then stays alive (holding the model's CPU weights and its client
identity) and waits for `standby-attach.py` to attach. Once attached it wakes
the model up, generates again, and checks the output is unchanged.

Run in terminal 1:   ./run.sh ./vllm/standby-prewarm.py
Then in terminal 2:  python3 ./vllm/standby-attach.py <PID printed here>
"""
import os
from os import environ, getpid
from time import perf_counter, sleep
import subprocess

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
SIGNAL = "/workspace/.phos_attach_done"

ENV = environ.copy()
ENV.pop("LD_PRELOAD")
ENV.pop("LD_LIBRARY_PATH", None)

SAMPLING = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)


def generate(llm, tag):
    outputs: list[RequestOutput] = llm.generate(PROMPT, SAMPLING)
    gen = outputs[0].outputs[0]
    print(f"[{tag}] #TOKENS={len(gen.token_ids)} finish={gen.finish_reason} OUTPUT={gen.text!r}")
    return gen.text


def run_checked(*args):
    subprocess.run((CLI, *args), env=ENV, check=True)


def init_process_group():
    import torch.distributed as dist
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)


def main():
    init_process_group()
    if os.path.exists(SIGNAL):
        os.remove(SIGNAL)

    llm = LLM(MODEL, enable_sleep_mode=True, tensor_parallel_size=1,
              enforce_eager=True, disable_log_stats=False, max_model_len=-1)

    before = generate(llm, "before-sleep")

    # Park the model and pre-warm the server's context service.
    llm.sleep()
    run_checked("detach", "--client-pid", str(PID))
    run_checked("standby", "--client-pids", str(PID))

    print(f"[prewarm] PID={PID}", flush=True)
    print(f"[prewarm] model parked, server pre-warmed. In another terminal run:", flush=True)
    print(f"[prewarm]     python3 ./vllm/standby-attach.py {PID}", flush=True)
    print(f"[prewarm] waiting for attach ...", flush=True)

    while not os.path.exists(SIGNAL):
        sleep(0.02)
    os.remove(SIGNAL)

    # standby-attach.py has completed the `attach`; bring the weights back.
    t = perf_counter()
    llm.wake_up()
    print(f"[prewarm] wake_up (weights H2D): {perf_counter() - t:.6f} s", flush=True)

    after = generate(llm, "after-restore")
    print(f"\n[prewarm] MATCH (before == after): {before == after}\n", flush=True)

    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
