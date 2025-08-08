use std::sync::atomic::Ordering;
use std::{fs, thread};

use network::ringbufferchannel::{SHM_FLAG_IDLE, SHM_FLAG_IN_USE};
use network::type_impl::send_slice;
use network::Channel;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    job_name: String,
    #[serde(default)]
    log_path: String,
    #[serde(default)]
    daemon_addr: String,
}

pub fn read_job_name() -> String {
    let file = fs::read_to_string("./pos.yaml").expect("failed to read ./pos.yaml");
    let Config { job_name, log_path, daemon_addr } =
        serde_yaml::from_str(&file).expect("failed to parse ./pos.yaml");
    match job_name.len() {
        1..=256 => {} // see kMaxJobNameLen in PhOS
        _ => panic!("job_name is empty or too long"),
    }
    if !log_path.is_empty() || !daemon_addr.is_empty() {
        log::warn!("remoting client ignores log_path and daemon_addr in ./pos.yaml");
    }
    log::info!("job_name: {job_name}");
    job_name
}

pub fn send_job_name(job_name: &str, channel_sender: &Channel) {
    send_slice(job_name.as_bytes(), channel_sender).unwrap()
}

pub fn set_client_flag_blocking(channel_sender: &Channel) {
    let flag = channel_sender.flag().unwrap();
    while let Err(_) =
        flag.compare_exchange(SHM_FLAG_IDLE, SHM_FLAG_IN_USE, Ordering::SeqCst, Ordering::SeqCst)
    {
        thread::yield_now();
    }
}

pub fn clear_client_flag(channel_sender: &Channel) {
    let flag = channel_sender.flag().unwrap();
    flag.fetch_and(!SHM_FLAG_IN_USE, Ordering::SeqCst);
}
