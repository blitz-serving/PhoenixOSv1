use std::ffi::*;
use std::fs;

use cudasys::types::cuda::*;

#[unsafe(no_mangle)]
extern "C" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    super::cuda_hijack::cuModuleLoadDataInternal(module, image.cast(), false)
}

#[unsafe(no_mangle)]
extern "C" fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) -> CUresult {
    let path = unsafe { CStr::from_ptr(fname) }.to_str().unwrap();
    log::debug!(target: "cuModuleLoad", "{path}");
    let file = fs::read(path).unwrap();
    cuModuleLoadData(module, file.as_ptr().cast())
}

#[unsafe(no_mangle)]
extern "C" fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
    let (opt_shadow_desc, pctx_init) = super::ClientThread::with_borrow(|client| {
        (client.opt_shadow_desc, client.cuda_device_init || client.cuda_pctx_flags.is_some())
    });
    // HACK: CUcontext is not used in other supported APIs yet.
    // TODO: use a more robust management solution.
    if opt_shadow_desc {
        let ctx = match pctx_init {
            true => usize::MAX,
            false => 0,
        };
        unsafe { *pctx = ctx as CUcontext };
        return CUresult::CUDA_SUCCESS;
    }
    super::cuda_hijack::cuCtxGetCurrentInternal(pctx)
}
