# vLLM Examples

Run these examples in the [provided environment](../../../environment/vllm/compose.example.yaml).

Edit the model paths and other variables in the selected script before running it.

## Cold Start and Vanilla Sleep

- `cold-start.py`: starts vLLM and measures startup time.
- `vanilla-sleep.py`: starts vLLM in sleep mode, sleeps it, then measures wake-up time.

Run these scripts directly with Python:

```bash
cd tests/apps/vllm
CUDA_VISIBLE_DEVICES=0 python3 cold-start.py
CUDA_VISIBLE_DEVICES=0 python3 vanilla-sleep.py
```

## Single and Multi

- `single.py`: measures attach and wake-up time for one in-process vLLM engine.
- `multi.py`: puts multiple vLLM engines into standby, then measures attach and wake-up time for each engine.

Start the server in one terminal from the repository root:

```bash
cargo run --release
```

Run the selected script in another terminal:

```bash
cd tests/apps
export CUDA_VISIBLE_DEVICES=0
./run.sh ./vllm/multi.py
```

To use `single.py` for PhOSv0-style checkpoint and restore, or to manually attach and detach, run the CLI from the repository root:

```bash
./target/release/server --help
```
