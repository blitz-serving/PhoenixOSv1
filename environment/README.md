## Using Docker Compose

For any environment, copy the `compose.example.yaml` file to `compose.yaml` and update `container_name`.

Edit `volumes` in `compose.yaml` to include directories like model weight storage.

Then run:

```shell
cd environment/<env_name>
docker compose up -d
```

Then you can use VS Code to attach to the container, or just:

```shell
docker exec -it <container_name> bash
```

To stop the container and start it later:

```shell
docker stop <container_name>
docker start <container_name>
```

To remove the container:

```shell
cd environment/<env_name>
docker compose down
```

## Appendix

### Installing requirements (legacy CUDA 11 environment, not needed for `vllm`)

It is recommended to mount a `docker-root` folder as the home directory so you don't need to install Rust in every container. To get the basic configuration like `.bashrc`, run:

```shell
cp -r /etc/skel/. /path/to/docker-root
```

- Note that we need Rust **nightly** toolchain to build the project, which is not installed in the docker image.

Mirrors:
- https://help.mirrors.cernet.edu.cn/rustup/
- https://help.mirrors.cernet.edu.cn/crates.io-index

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly
. "$HOME/.cargo/env"
```

- Version checklist
  - Rust: `nightly`
  - CMake: at least `3.22.1` (if you want to run some of the remoting tests)
  - Clang: at least `6.0.0-1ubuntu2`

You need to install Clang manually:

```shell
apt update
apt install clang --no-install-recommends
```

You probably also need to upgrade NCCL:

```shell
apt install libnccl2 libnccl-dev
```

If you need RDMA support:

```shell
apt install librdmacm-dev ibverbs-utils --no-install-recommends
```

### Build the (legacy PyTorch 1.13 + CUDA 11) docker image

Please refer to the [link](https://x8csr71rzs.feishu.cn/docx/DdXFdGSYOo8cktxgj8hcYh12nHf), and use the Dockerfile in this directory.
