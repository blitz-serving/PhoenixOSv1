use network::config::Config;
use network::oob::CheckpointRequest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct RestoreProcessRequest {
    pub job_name: String,
    pub ckpt_dir: String,
    pub client_pid: u32,
    pub ids: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandbyClient {
    pub client_pid: u32,
    pub id: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandbyRequest {
    pub clients: Vec<StandbyClient>,
    pub cuda_visible_devices: String,
}

#[tarpc::service]
pub trait Control {
    async fn assert_same_config(config: Config);

    async fn create_thread(id: i32, client_pid: u32);

    async fn checkpoint_process(request: CheckpointRequest);

    async fn restore_process(request: RestoreProcessRequest);

    async fn attach(id: i32, client_pid: u32);

    async fn standby(request: StandbyRequest);

    async fn detach() -> Option<Vec<u32>>;
}
