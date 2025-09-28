use std::sync::atomic::Ordering;

use network::ringbufferchannel::SHM_FLAG_WRITE_DISABLED;
use network::{Channel, NetworkConfig};

pub fn checkpoint() {
    todo!("checkpoint")
}

pub fn clear_flag(channel_receiver: &Channel) {
    let flag = channel_receiver.flag().unwrap();
    flag.fetch_and(!SHM_FLAG_WRITE_DISABLED, Ordering::SeqCst);
}

pub fn check_config(config: &NetworkConfig) {
    assert_eq!(config.comm_type, "shm", "PhOS only supports SHM communication");
    assert!(!config.emulator, "PhOS does not support emulator");
    assert!(config.opt_shadow_desc, "PhOS requires shadow (proxied) handles")
}
