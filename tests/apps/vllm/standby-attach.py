"""Attach phase for the PhOS context service (run this SECOND).

Given the PID printed by `standby-prewarm.py`, this measures how long it takes
to `attach` to the pre-warmed context service, then signals the prewarm process
to wake the model up and serve.

Usage:  python3 ./vllm/standby-attach.py <PID>

This script only talks to the PhOS server CLI (no CUDA), so it does not need
`run.sh` / libclient -- run it directly with python3.
"""
import subprocess
import sys
from os import environ
from time import perf_counter

environ.update({"VLLM_NO_USAGE_STATS": "1"})

CLI = "/workspace/target/release/server"
SIGNAL = "/workspace/.phos_attach_done"

ENV = environ.copy()
ENV.pop("LD_PRELOAD", None)
ENV.pop("LD_LIBRARY_PATH", None)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 standby-attach.py <PID-from-prewarm>")
    pid = sys.argv[1]

    t = perf_counter()
    subprocess.run((CLI, "attach", "--client-pid", pid), env=ENV, check=True)
    dt = perf_counter() - t
    print(f"[attach] attach --client-pid {pid}: {dt:.6f} s")

    # Tell the (waiting) prewarm process that attach is done -> it will wake_up.
    with open(SIGNAL, "w"):
        pass
    print("[attach] signaled prewarm process to wake up and generate")


if __name__ == "__main__":
    main()
