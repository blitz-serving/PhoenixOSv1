# PhoenixOS v1

PhoenixOS (PhOS) checkpoints and restores a running GPU process — its live CUDA context, loaded kernels, and GPU memory — without killing the process. This lets you park an idle GPU application off the GPU and later bring it back in place, so a single GPU can be time-shared across far more workloads than physically fit at once.

PhOS v1 is a preview refactor of [SJTU-IPADS/phoenixos](https://github.com/SJTU-IPADS/phoenixos) that focuses on ease of use and adds checkpoint-and-restore support for vLLM. It offers two modes:

- **GPU-only checkpoint/restore** (`detach`/`attach`): only the GPU worker is torn down while the host (CPU-side) process stays alive holding its state. A shared **context service** (`standby`) keeps kernels warm across processes so restores skip reloading them. For LLM serving this means many models can be parked using **zero GPU memory** and packed onto one GPU — where plain sleep would pin each model's CUDA context resident.
- **Whole-process checkpoint/restore** (with CRIU): PhOS snapshots the GPU state while CRIU checkpoints the host process, so the entire process — CPU side included — can be killed and later restored from disk.

See the [vLLM guide and examples](tests/apps/vllm/README.md) for a hands-on walkthrough of both.

## Docs

- Remoting component and `config.toml` docs: [docs/remoting.md](docs/remoting.md)
- Environment docs: [environment/README.md](environment/README.md)
- vLLM examples: [tests/apps/vllm/README.md](tests/apps/vllm/README.md)
