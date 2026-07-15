# vLLM v0.19.1 Examples

You need to run these examples in the provided `environment/vllm` Docker Compose environment.

Follow the steps in [environment/README.md](../../../environment/README.md) to start a container and spawn a terminal in it.

Make sure you check out the `compose.yaml` file and tweak the config according to the comments if:
- You want to run whole-process checkpoint and restore with CRIU.
- You want to download model weights from ModelScope instead of using local weights/download from Hugging Face.

Before running any Python script in this directory, edit it to point the model variable to a local directory with model weights, or a model identifier on Hugging Face or ModelScope to let vLLM download it.

```python
MODEL = "/path/to/model"  # edit this line
```

## Cold start and vanilla sleep

- `cold-start.py`: starts vLLM and measures startup time.
- `vanilla-sleep.py`: starts vLLM in sleep mode, sleeps it, then measures wake-up time.

Run these scripts directly with Python:

```bash
cd tests/apps/vllm
CUDA_VISIBLE_DEVICES=0 python3 cold-start.py
CUDA_VISIBLE_DEVICES=0 python3 vanilla-sleep.py
```

## Single or multiple model startup with PhOSv1 service

PhOSv1 checkpoints and restores a running vLLM sleep process without killing it. In the CLI, the two operations are:

- **checkpoint** = `detach`: serializes the engine's GPU state — loaded kernel modules, handles, and GPU memory — into a shared-memory snapshot, then tears the GPU worker down. The parked process holds **no GPU context and no GPU memory**.
- **restore** = `attach`: brings a GPU worker back, replays the snapshot, and the engine can `wake_up()` and serve again.

In this mode only the **GPU worker** is torn down and rebuilt — the host (CPU-side) vLLM process stays alive the entire time, holding all of its Python and host-memory state. This is deliberate, and it is the key difference from the [CRIU-based flow](#whole-process-checkpoint-and-restore-with-criu) below, which checkpoints and restores the CPU process as well. CRIU-restoring a complex engine like vLLM is slow, while host CPU and memory are relatively abundant on modern GPU servers, so there is no reason to kill the CPU process — freeing the scarce **GPU** is what matters.

Restores can be sped up by a shared **context service**, started with the `standby` command. "Standby" here means the service stays running with the checkpointed models' kernel modules already loaded into a live CUDA context on the GPU — warm and waiting for a later `attach`. It also deduplicates modules that several models have in common, so shared kernels are loaded only once. Because the kernels are already warm, `attach` reuses them instead of reloading and JIT-compiling them, so much of the GPU state is **not rebuilt at restore time** and restores are markedly faster.

From the repository root, copy the default config:

```bash
cd /workspace
cp config.example.toml config.toml
```

With the default config, the daemon binds to port 8001.

Build and start the server:

```bash
cargo build --release
./target/release/server
```

In another terminal, run the script you want with `run.sh` to use the PhOSv1 service:

- `single.py`: checkpoint then restore one vLLM engine — sleep → detach → attach → wake up — and measure the restore + wake-up time.
- `multi.py`: the same flow across several models, optionally behind a shared context service (`standby`) that preloads GPU kernels so restores are faster.
- `standby-prewarm.py` + `standby-attach.py`: the same context-service restore split into two scripts, so you can pre-warm (park) the context service in one and attach from another.

```bash
# another terminal
cd tests/apps
export CUDA_VISIBLE_DEVICES=0
./run.sh ./vllm/multi.py
```

### Why not just sleep?

For a **single** model, checkpoint/restore behaves just like `vanilla-sleep.py`: vLLM's sleep already offloads the model, so `detach`/`attach` buys you nothing extra and the timings are comparable.

The win shows up with **multiple** models. A vanilla-sleeping vLLM process keeps its CUDA context and GPU worker resident, so N sleeping models pin N contexts' worth of GPU memory at once. PhOSv1 `detach` frees the GPU worker entirely, so any number of checkpointed models can sit in host memory using **zero GPU memory** — you only pay for the GPU context of the model you `attach` and wake up. That lets you park far more models on a single GPU than sleep alone.

The key parts in `single.py`:

```python
# Sleep the model
llm.sleep()

# Disconnect the process from the PhOS worker and shut it down by running:
# /workspace/target/release/server detach --client-pid <PID>
run_checked("detach", "--client-pid", str(PID))

# (No GPU memory usage)

# Start a new PhOS GPU worker process and rebuild GPU context by running:
# /workspace/target/release/server attach --client-pid <PID>
run_checked("attach", "--client-pid", str(PID))

# Wake the model up
llm.wake_up()
```

The key parts in `multi.py`:

```python
sleep(llm)
run_checked("detach", "--client-pid", str(pid))

# Spin up a context service and preloads the GPU kernels for multiple processes:
# /workspace/target/release/server standby --client-pids <PID-1>,<PID-2>,...
run_checked("standby", "--client-pids", ",".join(str(pid) for pid in llms))

run_checked("attach", "--client-pid", str(pid))
llm.wake_up()
```

To restore GPU context without a preloaded context service, comment out the standby command call.

It is possible to manually call the `detach`/`attach`/`standby` commands in a terminal, but you need to insert `time.sleep()` into the script so you can find the exact time spot to detach the process.



## Whole-process checkpoint and restore with CRIU

Unlike the PhOSv1 service above — which keeps the host process alive and only cycles the GPU worker — this flow checkpoints and restores the **whole process, CPU side included**. PhOS snapshots the GPU state while [CRIU](https://criu.org/) checkpoints the host (CPU) process, so the original process can be killed entirely and later brought back from disk. This frees host CPU and memory too, at the cost of a slower restore for a complex engine like vLLM.

You can checkpoint and restore one vLLM process this way.

You need to turn off `io_uring` as CRIU [does not support it](https://github.com/checkpoint-restore/criu/issues/2131):

```bash
sysctl kernel.io_uring_disabled=2
```

Caution! This disables `io_uring` on the entire host machine, causing performance hit on other running applications. To turn it back on:

```bash
sysctl kernel.io_uring_disabled=0
```

Follow the same steps to build PhOS and start a server, then run the `single-ckpt.py` script:

```bash
./target/release/server

# the second terminal
cd tests/apps
export CUDA_VISIBLE_DEVICES=0
./run.sh ./vllm/single-ckpt.py
```

The default checkpoint directory is `/workspace/ckpt`, which can be edited in the script:

```python
CKPT_DIR = "/workspace/ckpt"  # edit this line
```

After the checkpoint is generated and the original process is killed, press Ctrl+C to terminate the original server and start a new one:

```bash
# after terminating the original server by Ctrl+C
./target/release/server
```

Then in the second terminal:

```bash
# the second terminal
export CUDA_VISIBLE_DEVICES=0
/workspace/target/release/server restore --ckpt-dir <CKPT-DIR>
```

If you're using the default checkpoint directory, you can omit the `--ckpt-dir` option.

The original process will restore in the terminal where the server is running.

After testing, don't forget to enable `io_uring`.

You cannot restore the same image after you rebuild the PhOS binaries due to CRIU restrictions.

Remove the checkpointed files before reusing the directory, otherwise the files will be mixed up.
