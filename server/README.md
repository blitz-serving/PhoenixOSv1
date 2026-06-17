# Server Architecture

This document describes the tarpc-based server control plane and worker lifecycle.

## PhOS CLI

PhOS checkpoint/restore CLI is integrated into the `server` binary.

Usage:

```bash
cargo run -p server -- checkpoint \
  --job-name demo \
  --ckpt-dir /tmp/phos-ckpt
```

```bash
cargo run -p server -- restore \
  --ckpt-dir /tmp/phos-ckpt
```

Socket resolution:

- default: read `client.network.daemon_socket` from `NETWORK_CONFIG` (or default `config.toml`)
- override: pass `--daemon-socket <ADDR>`

Misuse handling:

- If there are no extra args, server continues the normal daemon/worker flow.
- `cargo run -p server -- server` is accepted as a deprecated compatibility subcommand and continues normal flow.
- If args cannot be parsed as PhOS CLI, clap exits with an error.

## Control Plane

- Client -> daemon control RPC (OOB) uses tarpc over TCP (`daemon_socket`).
- Daemon -> worker process control RPC uses tarpc over UNIX socket (runtime-generated path).
- Worker processes are started by spawning the server binary itself and bootstrapped by:
  - `XPUREMOTING_SERVER_WORKER_SOCKET`

## Goals

- Use tarpc over TCP for client-daemon OOB.
- Use tarpc over UNIX socket for daemon-worker control.
- Worker creation with `spawn` + socket bootstrap.
- Keep two daemon behaviors:
  - default mode (no checkpoint)
  - `phos` mode (checkpoint/restore support)

## Interfaces

- `network/src/oob.rs`: OOB tarpc service, exposed as daemon public API.
- `server/src/pipe.rs`: internal daemon-worker tarpc service:
  - `create_thread(id, client_pid)`
  - `checkpoint_process(request)` (`phos` only)
  - `restore_process(request)` (`phos` only)

## Worker Bootstrap

- Daemon creates a unique UNIX socket path under `/tmp`.
- Daemon spawns the server binary and sets:
  - `XPUREMOTING_SERVER_WORKER_SOCKET=<unix-socket-path>`
- Server process enters worker mode only when `XPUREMOTING_SERVER_WORKER_SOCKET` is present and non-empty.
- Worker connects back to daemon over UNIX socket and serves control RPCs.

## Runtime Notes

- Thread launch path is shared between default and `phos` daemon behavior.
- `phos` checkpoint/restore daemon behavior is implemented in `server/src/phos/oob.rs`.
