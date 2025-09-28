use cudasys::types::cuda::*;
use std::os::raw::*;

#[unsafe(no_mangle)]
extern "C" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    super::cuda_hijack::cuModuleLoadDataInternal(module, image.cast(), false)
}
