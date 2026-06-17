mod ckpt;
mod cuda;
mod manager;

use std::ffi::c_void;
use std::mem;

pub use ckpt::*;
use cudasys::cuda::*;
use cudasys::cudart::cudaError_t;
use manager::{MEMORY_MANAGER, MemoryCheckpointMeta};

#[cfg(target_pointer_width = "64")]
pub fn cuda_malloc(dev_ptr: *mut *mut c_void, size: usize) -> cudaError_t {
    match MEMORY_MANAGER.lock().unwrap().allocate(size) {
        Ok(ptr) => {
            unsafe { *dev_ptr = ptr as *mut c_void };
            cudaError_t::cudaSuccess
        }
        Err(e) => unsafe { mem::transmute(e) },
    }
}

#[cfg(target_pointer_width = "64")]
pub fn cuda_free(dev_ptr: *mut c_void) -> cudaError_t {
    match MEMORY_MANAGER.lock().unwrap().free(dev_ptr as CUdeviceptr) {
        CUresult::CUDA_ERROR_INVALID_VALUE => cudaError_t::cudaErrorInvalidDevicePointer,
        e => unsafe { mem::transmute(e) },
    }
}

pub fn cu_mem_address_reserve(
    ptr: *mut CUdeviceptr,
    size: usize,
    alignment: usize,
    addr: CUdeviceptr,
    flags: u64,
) -> CUresult {
    assert_eq!(flags, 0);
    if addr != 0 {
        log::warn!("cuMemAddressReserve: ignoring addr hint {addr:#x}");
    }
    let dev_ptr = MEMORY_MANAGER.lock().unwrap().inner()?.reserve_address(size, alignment)?;
    unsafe { *ptr = dev_ptr };
    CUresult::CUDA_SUCCESS
}

pub fn cu_mem_address_free(_ptr: CUdeviceptr, _size: usize) -> CUresult {
    CUresult::CUDA_SUCCESS
}

pub fn cu_mem_create(
    handle: *mut CUmemGenericAllocationHandle,
    size: usize,
    prop: &CUmemAllocationProp,
    flags: u64,
) -> CUresult {
    assert_eq!(flags, 0);
    let proxy = MEMORY_MANAGER.lock().unwrap().inner()?.create_user_handle(size, prop)?;
    unsafe { *handle = proxy as _ };
    CUresult::CUDA_SUCCESS
}

pub fn cu_mem_map(
    ptr: CUdeviceptr,
    size: usize,
    offset: usize,
    handle: CUmemGenericAllocationHandle,
    flags: u64,
) -> CUresult {
    assert_eq!(offset, 0);
    assert_eq!(flags, 0);
    MEMORY_MANAGER.lock().unwrap().inner()?.map_user_handle(ptr, size, handle)
}

pub fn cu_mem_set_access(ptr: CUdeviceptr, size: usize, desc: &[CUmemAccessDesc]) -> CUresult {
    let [desc] = desc else {
        log::error!("cuMemSetAccess: only support count=1, got {}", desc.len());
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    };
    MEMORY_MANAGER.lock().unwrap().inner()?.set_access(ptr, size, desc)
}

pub fn cu_mem_unmap(ptr: CUdeviceptr, size: usize) -> CUresult {
    MEMORY_MANAGER.lock().unwrap().inner()?.unmap(ptr, size)
}

pub fn cu_mem_release(handle: CUmemGenericAllocationHandle) -> CUresult {
    MEMORY_MANAGER.lock().unwrap().inner()?.release_user_handle(handle)
}

pub fn reset() {
    MEMORY_MANAGER.lock().unwrap().reset();
}
