use std::path::{Path, PathBuf};
use std::{env, fs};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub common: CommonConfig,
    pub client: ClientConfig,
    pub server: ServerConfig,
}

impl Config {
    pub fn read_env() -> Self {
        check_log_level();
        let path = match env::var_os("NETWORK_CONFIG") {
            Some(path) => PathBuf::from(path),
            None => {
                let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                path.set_file_name("config.toml");
                path
            }
        };
        Self::read_from_file(&path)
    }

    pub fn read_from_file(path: &Path) -> Self {
        let content = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
        toml::from_str(&content)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()))
    }

    pub fn write_to_file(&self, path: &Path) {
        let content = toml::to_string(self).unwrap();
        fs::write(path, content)
            .unwrap_or_else(|e| panic!("Failed to write {}: {e}", path.display()));
    }

    pub fn assert_same_server(&self, other: &Self) {
        assert_eq!(self.common, other.common);
        assert_eq!(self.server, other.server);
    }
}

fn check_log_level() {
    if log::max_level() > log::STATIC_MAX_LEVEL {
        log::error!(
            "Max log level ({}) is higher than compiled ({}) \n\
            Try turning off the 'rdma' feature if log is required \n\
            See: https://doc.rust-lang.org/cargo/reference/features.html#feature-unification \n\
            https://github.com/SJTU-IPADS/krcore-artifacts/blob/develop/rdma-shim/Cargo.toml#L12",
            log::max_level(),
            log::STATIC_MAX_LEVEL,
        );
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommonConfig {
    pub comm_type: String,
    pub stoc_channel_name: String,
    pub ctos_channel_name: String,
    pub buf_size: usize,
    pub emulator: bool,
    pub rtt: f64,
    pub bandwidth: f64,
    pub opt_async_api: bool,
    pub opt_shadow_desc: bool,
}

#[derive(Debug)]
pub enum CommType {
    Shm {
        emulator: bool,
    },
    Tcp,
    #[cfg(feature = "rdma")]
    Rdma,
}

impl CommonConfig {
    pub fn checked_comm_type(&self) -> CommType {
        if self.emulator {
            assert_eq!(self.comm_type, "shm");
            return CommType::Shm { emulator: true };
        }
        match self.comm_type.as_str() {
            "shm" => CommType::Shm { emulator: false },
            "tcp" => CommType::Tcp,
            #[cfg(feature = "rdma")]
            "rdma" => CommType::Rdma,
            comm_type => panic!("Unsupported communication type in config: {comm_type}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    pub opt_local: bool,
    pub job_name: String,
    pub network: NetworkConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerConfig {
    pub network: NetworkConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub daemon_socket: String,
    pub handshake_socket: String,
    pub device_name: String,
    pub device_port: u8,
}
