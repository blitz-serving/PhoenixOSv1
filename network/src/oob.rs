use serde::{Deserialize, Serialize};
use tarpc::context::Context;
use tarpc::serde_transport;
use tarpc::tokio_serde::formats::Bincode;

use crate::config::CommonConfig;

#[derive(Debug, Serialize, Deserialize)]
pub struct NewClientThreadRequest {
    pub pid: u32,
    pub job_name: String,
    pub cuda_visible_devices: String,
    pub config: CommonConfig,
    pub is_phos: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CheckpointMode {
    Kill,
    LeaveRunning,
    Detach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointRequest {
    pub job_name: String,
    pub ckpt_dir: String,
    pub mode: CheckpointMode,
}

#[tarpc::service]
pub trait Oob {
    async fn new_client_thread(request: NewClientThreadRequest) -> i32;

    async fn checkpoint_job(request: CheckpointRequest);

    async fn restore_job(ckpt_dir: String, cuda_visible_devices: String);

    async fn detach(client_pid: u32);

    async fn attach(client_pid: u32);

    async fn standby(client_pids: Vec<u32>);
}

pub async fn connect(addr: &str) -> OobClient {
    let mut transport = serde_transport::tcp::connect(addr, Bincode::default);
    transport.config_mut().max_frame_length(usize::MAX);
    let transport = transport.await.unwrap();
    OobClient::new(Default::default(), transport).spawn()
}

#[inline]
pub fn current_context() -> Context {
    Context::current()
}
