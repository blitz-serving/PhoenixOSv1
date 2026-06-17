use std::env;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use network::config::Config;
use network::oob::{self, CheckpointMode, CheckpointRequest};
use tarpc::context::Context;

#[derive(Debug, Parser)]
#[clap(name = "server", about = "PhOS server & checkpoint/restore CLI")]
struct Cli {
    #[clap(subcommand)]
    command: Option<Command>,
    #[clap(long, global = true)]
    daemon_socket: Option<String>,
}

#[derive(Debug, Subcommand)]
enum Command {
    Checkpoint(CheckpointArgs),
    Restore(RestoreArgs),
    Detach(PidArgs),
    Attach(PidArgs),
    Standby(StandbyArgs),
    #[clap(about = "DEPRECATED: no-op compatibility subcommand; runs normal server flow")]
    Server,
}

#[derive(Debug, Args)]
struct CheckpointArgs {
    #[clap(long, default_value = "default")]
    job_name: String,
    #[clap(long, default_value = "/workspace/ckpt")]
    ckpt_dir: String,
    #[clap(long)]
    leave_running: bool,
}

#[derive(Debug, Args)]
struct RestoreArgs {
    #[clap(long, default_value = "/workspace/ckpt")]
    ckpt_dir: String,
}

#[derive(Debug, Args)]
struct PidArgs {
    #[clap(long)]
    client_pid: u32,
}

#[derive(Debug, Args)]
struct StandbyArgs {
    #[clap(long, required = true, value_delimiter = ',')]
    client_pids: Vec<u32>,
}

fn daemon_socket(override_addr: Option<String>) -> String {
    override_addr.unwrap_or_else(|| Config::read_env().client.network.daemon_socket)
}

fn long_context() -> Context {
    let mut context = Context::current();
    context.deadline = Instant::now() + Duration::from_secs(120);
    context
}

pub async fn run() -> bool {
    let cli = Cli::parse();
    let command = match cli.command {
        None | Some(Command::Server) => return false,
        Some(command) => command,
    };
    let addr = daemon_socket(cli.daemon_socket);
    match command {
        Command::Checkpoint(args) => {
            let oob = oob::connect(&addr).await;
            oob.checkpoint_job(
                long_context(),
                CheckpointRequest {
                    job_name: args.job_name,
                    ckpt_dir: args.ckpt_dir,
                    mode: if args.leave_running {
                        CheckpointMode::LeaveRunning
                    } else {
                        CheckpointMode::Kill
                    },
                },
            )
            .await
            .unwrap_or_else(|e| panic!("failed to call checkpoint_job via daemon {addr}: {e}"));
            log::info!("checkpoint request sent");
        }
        Command::Restore(args) => {
            let oob = oob::connect(&addr).await;
            oob.restore_job(
                long_context(),
                args.ckpt_dir,
                env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(),
            )
            .await
            .unwrap_or_else(|e| panic!("failed to call restore_job via daemon {addr}: {e}"));
            log::info!("restore request sent");
        }
        Command::Detach(args) => {
            let oob = oob::connect(&addr).await;
            oob.detach(long_context(), args.client_pid)
                .await
                .unwrap_or_else(|e| panic!("failed to call detach via daemon {addr}: {e}"));
            log::info!("detach request sent for client_pid {}", args.client_pid);
        }
        Command::Attach(args) => {
            let oob = oob::connect(&addr).await;
            oob.attach(long_context(), args.client_pid)
                .await
                .unwrap_or_else(|e| panic!("failed to call attach via daemon {addr}: {e}"));
            log::info!("attach completed for client_pid {}", args.client_pid);
        }
        Command::Standby(args) => {
            let oob = oob::connect(&addr).await;
            oob.standby(long_context(), args.client_pids.clone())
                .await
                .unwrap_or_else(|e| panic!("failed to call standby via daemon {addr}: {e}"));
            log::info!("standby completed for client_pids {:?}", args.client_pids);
        }
        Command::Server => unreachable!(),
    }
    true
}
