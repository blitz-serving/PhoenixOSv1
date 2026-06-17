use std::sync::OnceLock;
use std::{env, ptr};

use cudasys::nvml::*;

use super::ByteSize;

const ENABLE: bool = true;

static NVML_DEVICE: OnceLock<usize> = OnceLock::new();

pub fn log_memory(stage: &str) {
    if !ENABLE {
        return;
    }

    let device = *NVML_DEVICE.get_or_init(init_device) as nvmlDevice_t;

    let pid = std::process::id();
    let total_bytes = query_used_bytes_for_pid(device, pid);
    log::info!("{stage}: pid {pid} used_gpu_memory={}", ByteSize(total_bytes));
}

fn init_device() -> usize {
    match unsafe { nvmlInit_v2() } {
        nvmlReturn_t::NVML_SUCCESS => {}
        nvmlReturn_t::NVML_ERROR_ALREADY_INITIALIZED => {}
        err => panic!("nvmlInit_v2 failed: {err:?}"),
    }

    let device_index = env::var("CUDA_VISIBLE_DEVICES").unwrap().parse().unwrap();
    let mut device = ptr::null_mut();
    match unsafe { nvmlDeviceGetHandleByIndex_v2(device_index, &mut device) } {
        nvmlReturn_t::NVML_SUCCESS => {}
        err => panic!("nvmlDeviceGetHandleByIndex_v2({device_index}) failed: {err:?}"),
    }

    device as usize
}

fn query_used_bytes_for_pid(device: nvmlDevice_t, pid: u32) -> u64 {
    let mut count = 0;
    match unsafe { nvmlDeviceGetComputeRunningProcesses_v3(device, &mut count, ptr::null_mut()) } {
        nvmlReturn_t::NVML_SUCCESS => return 0,
        nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE => {}
        err => panic!("nvmlDeviceGetComputeRunningProcesses_v3(count) failed: {err:?}"),
    }

    let mut infos = Box::<[nvmlProcessInfo_t]>::new_uninit_slice(count as usize * 2 + 5);
    match unsafe {
        nvmlDeviceGetComputeRunningProcesses_v3(device, &mut count, infos.as_mut_ptr().cast_init())
    } {
        nvmlReturn_t::NVML_SUCCESS => {}
        err => panic!("nvmlDeviceGetComputeRunningProcesses_v3(data) failed: {err:?}"),
    }
    let infos = unsafe { infos[..count as usize].assume_init_ref() };

    let not_available = (NVML_VALUE_NOT_AVAILABLE as i64).cast_unsigned();
    infos
        .iter()
        .filter(|info| info.pid == pid)
        .map(|info| {
            if info.usedGpuMemory == not_available {
                0
            } else {
                info.usedGpuMemory
            }
        })
        .sum()
}
