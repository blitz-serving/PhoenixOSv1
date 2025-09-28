use crate::types::cuda::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

// TODO: depend on `dev`, device reset, PhOS restore, etc.
#[cuda_hook(proc_id = 670)]
fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> CUresult {
    'client_before_send: {
        if let (true, Some(cuda_pctx_flags)) = (client.opt_local, client.cuda_pctx_flags) {
            unsafe {
                *flags = cuda_pctx_flags;
                *active = 1;
            }
            return Default::default();
        }
    }
    'client_after_recv: {
        if *active == 1 {
            client.cuda_pctx_flags = Some(*flags);
        }
    }
}

#[cuda_hook(proc_id = 918, async_api)]
fn cuLaunchKernel(
    #[handle = "use"] f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    #[handle = "use"] hStream: CUstream,
    #[skip] kernelParams: *mut *mut c_void,
    #[value = std::ptr::null_mut()] extra: *mut *mut c_void,
) -> CUresult {
    'client_before_send: {
        let args = super::cuda_hijack_utils::pack_kernel_args(f, kernelParams);
    }
    'client_extra_send: {
        send_ctx.send_slice(&args, "args");
    }
    'server_extra_recv: {
        let args = recv_ctx.recv_slice::<u8>("args");
    }
    'server_execution: {
        let hook_result = super::cuda_exe_utils::cu_launch_kernel(
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            &args,
        );
    }
}

#[cuda_custom_hook] // calls the internal API below
fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;

#[cuda_hook(proc_id = 701, parent = cuModuleLoadData)]
fn cuModuleLoadDataInternal(
    #[handle = "create"] module: *mut CUmodule,
    #[host(len = len)] image: *const c_void,
    #[skip] is_runtime: bool,
) -> CUresult {
    'client_before_send: {
        let len = if let Some(header) = FatBinaryHeader::cast(image.cast()) {
            header.entire_len()
        } else {
            crate::elf::elf_len(image.cast())
        };
    }
    'client_extra_send: {
        let _guard = std::mem::DropGuard::new((), |_| {
            if !std::thread::panicking() {
                let module = unsafe { *module };
                DRIVER_CACHE.write().unwrap().insert_image(module, image, is_runtime);
            }
        });
    }
}

#[cuda_hook(proc_id = 705)]
fn cuModuleGetFunction(
    #[handle = "create"] hfunc: *mut CUfunction,
    #[handle = "use"] hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    'client_extra_send: {
        let _guard = std::mem::DropGuard::new((), |_| {
            if !std::thread::panicking() {
                let hfunc = unsafe { *hfunc };
                DRIVER_CACHE.write().unwrap().insert_function(hfunc, hmod, name);
            }
        });
    }
}

#[cuda_hook(proc_id = 640)]
fn cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult;

#[cuda_hook(proc_id = 630, async_api = false)]
fn cuInit(Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 684)]
fn cuCtxGetCurrent(#[handle = "create"] pctx: *mut CUcontext) -> CUresult;

#[cuda_hook(proc_id = 650)]
fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;

#[cuda_hook(proc_id = 651)]
fn cuDeviceGetAttribute(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 910)]
fn cuFuncGetAttribute(
    pi: *mut c_int,
    attrib: CUfunction_attribute,
    #[handle = "use"] hfunc: CUfunction,
) -> CUresult;

#[cuda_hook(proc_id = 844)]
fn cuPointerGetAttribute(
    #[host(output, len = attribute.data_size())] data: *mut c_void,
    attribute: CUpointer_attribute,
    // TODO: #[device(input)]
    ptr: CUdeviceptr,
) -> CUresult;

#[cuda_hook(proc_id = 1002)]
fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *mut c_int,
    #[handle = "use"] func: CUfunction,
    blockSize: c_int,
    dynamicSMemSize: usize,
    flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 912, async_api = false)]
fn cuFuncSetAttribute(
    #[handle = "modify"] hfunc: CUfunction,
    attrib: CUfunction_attribute,
    value: c_int,
) -> CUresult;
