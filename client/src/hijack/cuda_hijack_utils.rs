use std::ffi::c_void;
use std::{ptr, slice};

use cudasys::cuda::CUfunction;

use crate::hijack::client::DRIVER_CACHE;

pub fn pack_kernel_args(f: CUfunction, arg_ptrs: *mut *mut c_void) -> Box<[u8]> {
    let driver = DRIVER_CACHE.read().unwrap();
    let info = driver.get_params(f);
    let Some(last) = info.last() else { return Default::default() };
    let mut result = vec![0u8; (last.offset + last.size()) as usize];
    let arg_ptrs = unsafe { slice::from_raw_parts(arg_ptrs, info.len()) };
    for (param, arg_ptr) in info.iter().zip(arg_ptrs) {
        unsafe {
            ptr::copy_nonoverlapping(
                arg_ptr.cast(),
                result.as_mut_ptr().add(param.offset as usize),
                param.size() as usize,
            );
        }
        if log::max_level() < log::Level::Trace {
            continue;
        }
        match param.size() {
            8 if arg_ptr.cast::<u64>().is_aligned() => {
                let arg = unsafe { *arg_ptr.cast::<u64>() };
                log::trace!(target: "cuLaunchKernel", "arg = {arg:#x}");
            }
            4 if arg_ptr.cast::<i32>().is_aligned() => {
                let arg = unsafe { *arg_ptr.cast::<i32>() };
                log::trace!(target: "cuLaunchKernel", "arg = {arg}");
            }
            size => log::trace!(target: "cuLaunchKernel", "arg<{size}> = {:?}", unsafe {
                slice::from_raw_parts(arg_ptr.cast::<u8>(), param.size() as usize)
            }),
        }
    }
    result.into_boxed_slice()
}
