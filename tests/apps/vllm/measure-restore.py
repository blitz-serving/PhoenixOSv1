"""Measure PhOSv1 service startup (restore) performance.

Phases per iteration:
  detach  -> kills the GPU worker (0 GPU memory)
  attach  -> PhOS service restores a GPU worker / rebuilds context   <-- "start from PhOS"
  wake_up -> vLLM copies weights CPU->GPU
  gen     -> first generation after restore (liveness / first-token-ish)

Reports min/avg/median over ITERS iterations, plus the cold-start baseline.
"""
import subprocess
from os import environ, getpid
from time import perf_counter

environ.update({
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NO_USAGE_STATS": "1",
})

from vllm import LLM, SamplingParams

MODEL = "/models/Qwen3-0.6B"
PROMPT = "Hello, I'm a language model,"
MAX_TOKENS = 32
ITERS = 5
CLI = "/workspace/target/release/server"
PID = getpid()

ENV = environ.copy()
ENV.pop("LD_PRELOAD")
ENV.pop("LD_LIBRARY_PATH", None)

SAMPLING = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)


def run_checked(*args):
    subprocess.run((CLI, *args), env=ENV, check=True)


def init_process_group():
    import torch.distributed as dist
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)


def stats(xs):
    s = sorted(xs)
    n = len(s)
    return min(s) * 1000, sum(s) / n * 1000, s[n // 2] * 1000


def main():
    init_process_group()

    t0 = perf_counter()
    llm = LLM(MODEL, enable_sleep_mode=True, tensor_parallel_size=1,
              enforce_eager=True, disable_log_stats=False, max_model_len=-1)
    cold = perf_counter() - t0
    print(f"[cold-start] full load+init: {cold * 1000:.1f} ms")

    llm.generate(PROMPT, SAMPLING)  # warm up

    attach_t, wake_t, gen_t = [], [], []
    for i in range(ITERS):
        llm.sleep()
        run_checked("detach", "--client-pid", str(PID))

        t = perf_counter(); run_checked("attach", "--client-pid", str(PID)); a = perf_counter() - t
        t = perf_counter(); llm.wake_up(); w = perf_counter() - t
        t = perf_counter(); llm.generate(PROMPT, SAMPLING); g = perf_counter() - t

        attach_t.append(a); wake_t.append(w); gen_t.append(g)
        print(f"[iter {i}] attach={a*1000:8.1f}  wake_up={w*1000:8.1f}  "
              f"restore={ (a+w)*1000:8.1f}  first-gen={g*1000:8.1f}  (ms)")

    print("\n==================== RESTORE PERF (ms) ====================")
    print(f"{'phase':24s} {'min':>9s} {'avg':>9s} {'median':>9s}")
    for name, xs in [
        ("attach (PhOS restore)", attach_t),
        ("wake_up (weights H2D)", wake_t),
        ("restore = attach+wake", [a + w for a, w in zip(attach_t, wake_t)]),
        ("first-gen after restore", gen_t),
    ]:
        mn, avg, md = stats(xs)
        print(f"{name:24s} {mn:9.1f} {avg:9.1f} {md:9.1f}")
    print(f"\ncold-start baseline: {cold*1000:.1f} ms  "
          f"(speedup vs restore-median ~ {cold*1000 / stats([a+w for a,w in zip(attach_t, wake_t)])[2]:.1f}x)")
    print("==========================================================\n")

    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
