#![expect(non_snake_case)]

use std::cell::OnceCell;
use std::ffi::{CStr, c_char, c_int};
use std::ptr;

use cudasys::types::cudart::*;

use super::*;

#[unsafe(no_mangle)]
extern "C" fn __cudaRegisterFatBinary(fatCubin: *const FatBinaryWrapper) -> FatBinaryHandle {
    let fatCubin = unsafe { (*fatCubin).unwrap() };

    #[thread_local]
    static VALIDATE: OnceCell<bool> = OnceCell::new();

    if *VALIDATE.get_or_init(|| std::env::var_os("VALIDATE_FAT_BINARY").is_some()) {
        unsafe { (*fatCubin).validate_code() };
    }

    let mut runtime = RUNTIME_CACHE.write().unwrap();
    runtime.lazy_fatbins.push(fatCubin);
    FatBinaryHandle::from_index(runtime.lazy_fatbins.len() - 1)
}

#[unsafe(no_mangle)]
pub extern "C" fn __cudaUnregisterFatBinary(_fatCubinHandle: FatBinaryHandle) {
    // This is called when the client process exits, when the thread local storage is already dropped.
}

#[unsafe(no_mangle)]
pub extern "C" fn __cudaRegisterFatBinaryEnd(_fatCubinHandle: FatBinaryHandle) {
    // TODO: no actual impact
}

#[unsafe(no_mangle)]
pub extern "C" fn __cudaRegisterFunction(
    fatCubinHandle: FatBinaryHandle,
    hostFun: HostPtr,
    deviceFun: *mut c_char,
    deviceName: *const c_char,
    _thread_limit: c_int,
    _tid: *mut uint3,
    _bid: *mut uint3,
    _bDim: *mut dim3,
    _gDim: *mut dim3,
    _wSize: *mut c_int,
) {
    if cfg!(debug_assertions) && !ptr::eq(deviceName, deviceFun) {
        log::warn!(
            "deviceName: {:?}, deviceFun: {:?}",
            unsafe { CStr::from_ptr(deviceName) },
            unsafe { CStr::from_ptr(deviceFun) },
        );
    }

    // Some kernels are registered multiple times from different fatbins
    // e.g. "void cub::EmptyKernel<void>()"
    let mut runtime = RUNTIME_CACHE.write().unwrap();
    runtime.lazy_functions.entry(hostFun).or_insert((fatCubinHandle, deviceName));
}

#[unsafe(no_mangle)]
pub extern "C" fn __cudaRegisterVar(
    fatCubinHandle: FatBinaryHandle,
    hostVar: HostPtr,
    deviceAddress: *mut c_char,
    deviceName: *const c_char,
    _ext: c_int,
    _size: usize,
    _constant: c_int,
    _global: c_int,
) {
    if cfg!(debug_assertions) && !ptr::eq(deviceName, deviceAddress) {
        log::warn!(
            "deviceName: {:?}, deviceFun: {:?}",
            unsafe { CStr::from_ptr(deviceName) },
            unsafe { CStr::from_ptr(deviceAddress) },
        );
    }

    let mut runtime = RUNTIME_CACHE.write().unwrap();
    runtime.lazy_variables.entry(hostVar).or_insert((fatCubinHandle, deviceName));
}
