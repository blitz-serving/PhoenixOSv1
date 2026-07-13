# vLLM v0.19.1 Examples

You need to run these examples in the provided `environment/vllm` Docker Compose environment.

Follow the steps in [environment/README.md](../../../environment/README.md) to start a container and spawn a terminal in it.

Make sure you check out the `compose.yaml` file and tweak the config according to the comments if:
- You want to run PhOSv0-style checkpoint and restore with CRIU.
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

## Single or multiple model inference with PhOSv1 service

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

In another terminal, run the script you want with `run.sh` to use PhOSv1 service:

- `single.py`: measures restore (attach) and wake-up time for one in-process vLLM engine.
- `multi.py`: starts a context service for multiple models, then measures restore and wake-up time for each engine.

```bash
# another terminal
cd tests/apps
export CUDA_VISIBLE_DEVICES=0
./run.sh ./vllm/multi.py
```

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

## PhOSv0-style checkpoint and restore with CRIU

You can checkpoint one vLLM process with PhOSv0-style checkpoint and restore with CRIU.

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
