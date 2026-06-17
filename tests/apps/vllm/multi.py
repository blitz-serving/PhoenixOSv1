import subprocess
from os import environ
from typing import cast

environ.update({
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
})

from vllm import LLM, SamplingParams
from vllm.v1.engine.core_client import SyncMPClient
from vllm.v1.engine.utils import CoreEngineProcManager

MODELS = [
    "/path/to/model-1",
    "/path/to/model-2",
]
PROMPT = "placeholder"
MAX_TOKENS = 320
CLI = "/workspace/target/release/server"

ENV = environ.copy()
ENV.pop("LD_PRELOAD")
ENV.pop("LD_LIBRARY_PATH", None)


def main():
    llms: dict[int, LLM] = {}

    for model in MODELS:
        llm = LLM(
            model,
            enable_sleep_mode=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            disable_log_stats=False,
            max_model_len=-1,
        )

        sleep(llm)
        pid = get_pid(llm)
        run_checked("detach", "--client-pid", str(pid))
        llms[pid] = llm

    # Standby for faster attaching; comment out to create new worker for each attach.
    run_checked("standby", "--client-pids", ",".join(str(pid) for pid in llms))

    for pid, llm in llms.items():
        run_checked("attach", "--client-pid", str(pid))
        llm.wake_up()
        llm.generate(PROMPT, SamplingParams(max_tokens=MAX_TOKENS))
        sleep(llm)
        run_checked("detach", "--client-pid", str(pid))

    for llm in llms.values():
        llm.llm_engine.engine_core.shutdown()


def get_pid(llm: LLM) -> int:
    core_client = llm.llm_engine.engine_core
    assert isinstance(core_client, SyncMPClient)
    engine_manager = cast(CoreEngineProcManager, core_client.resources.engine_manager)
    [process] = engine_manager.processes
    assert (pid := process.pid)
    return pid


def sleep(llm: LLM, level: int = 1):
    llm.sleep(level)


def run_checked(*args: str):
    subprocess.run((CLI, *args), env=ENV, check=True)


if __name__ == "__main__":
    main()
