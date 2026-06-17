use crate::types::nvml::*;
use codegen::cuda_hook;
use std::os::raw::*;

#[cuda_hook(proc_id = 991000, async_api = false)]
fn nvmlInit_v2() -> nvmlReturn_t;

#[cuda_hook(proc_id = 991001)]
fn nvmlDeviceGetCount_v2(deviceCount: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991002, async_api = false)]
fn nvmlInitWithFlags(flags: c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991003, async_api = false)]
fn nvmlShutdown() -> nvmlReturn_t;

#[cuda_hook(proc_id = 991004)]
fn nvmlDeviceGetHandleByIndex_v2(index: c_uint, device: *mut nvmlDevice_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991005)]
fn nvmlDeviceGetName(
    device: nvmlDevice_t,
    #[host(output, len = length)] name: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991006)]
fn nvmlDeviceGetCudaComputeCapability(
    device: nvmlDevice_t,
    major: *mut c_int,
    minor: *mut c_int,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991007)]
fn nvmlDeviceGetP2PStatus(
    device1: nvmlDevice_t,
    device2: nvmlDevice_t,
    p2pIndex: nvmlGpuP2PCapsIndex_t,
    p2pStatus: *mut nvmlGpuP2PStatus_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991008)]
fn nvmlDeviceGetMemoryInfo(device: nvmlDevice_t, memory: *mut nvmlMemory_t) -> nvmlReturn_t;
