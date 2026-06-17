use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use network::channel::Sender;
use network::ringbufferchannel::{SHM_FLAG_IDLE, SHM_FLAG_IN_USE};

pub fn set_client_flag_blocking(channel: &Sender) {
    let Sender::Shm(channel) = channel else { panic!("SHM channel expected") };
    let flag = unsafe { AtomicU64::from_ptr(channel.flag_ptr()) };
    while flag
        .compare_exchange(SHM_FLAG_IDLE, SHM_FLAG_IN_USE, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        thread::yield_now();
    }
}

pub fn clear_client_flag(channel: &Sender) {
    let Sender::Shm(channel) = channel else { panic!("SHM channel expected") };
    let flag = unsafe { AtomicU64::from_ptr(channel.flag_ptr()) };
    flag.fetch_and(!SHM_FLAG_IN_USE, Ordering::SeqCst);
}
