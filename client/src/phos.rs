use std::sync::atomic::Ordering;
use std::{fs, thread};

use network::Channel;
use network::ringbufferchannel::{SHM_FLAG_IDLE, SHM_FLAG_IN_USE};

pub fn set_client_flag_blocking(channel_sender: &Channel) {
    let flag = channel_sender.flag().unwrap();
    while flag
        .compare_exchange(SHM_FLAG_IDLE, SHM_FLAG_IN_USE, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        thread::yield_now();
    }
}

pub fn clear_client_flag(channel_sender: &Channel) {
    let flag = channel_sender.flag().unwrap();
    flag.fetch_and(!SHM_FLAG_IN_USE, Ordering::SeqCst);
}
