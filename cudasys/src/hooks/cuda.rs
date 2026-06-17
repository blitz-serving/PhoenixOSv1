use std::os::raw::*;

use codegen::{cuda_custom_hook, cuda_hook};

use crate::types::cuda::*;

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

#[cuda_hook(proc_id = 999918, async_api, min_cuda_version = 12)]
fn cuLaunchKernelEx(
    #[host] config: *const CUlaunchConfig,
    #[handle = "use"] f: CUfunction,
    #[skip] kernelParams: *mut *mut c_void,
    #[value = std::ptr::null_mut()] extra: *mut *mut c_void,
) -> CUresult {
    'client_before_send: {
        let args = super::cuda_hijack_utils::pack_kernel_args(f, kernelParams);
    }
    'client_extra_send: {
        let attrs = unsafe { std::slice::from_raw_parts((*config).attrs, (*config).numAttrs as _) };
        send_ctx.send_slice(attrs, "config.attrs");
        send_ctx.send_slice(&args, "args");
    }
    'server_extra_recv: {
        let attrs = recv_ctx.recv_slice::<CUlaunchAttribute>("config.attrs");
        let args = recv_ctx.recv_slice::<u8>("args");
    }
    'server_execution: {
        unsafe { (*config__ptr).attrs = attrs.as_ptr().cast_mut() };
        let hook_result = super::cuda_exe_utils::cu_launch_kernel_ex(&config, f, &args);
    }
}

#[cuda_custom_hook] // calls cuModuleLoadData
fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) -> CUresult;

#[cuda_custom_hook] // calls the internal API below
fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;

// FIXME: modules and functions should be handled in process level
#[cuda_hook(proc_id = 701, parent = cuModuleLoadData)]
fn cuModuleLoadDataInternal(
    module: *mut CUmodule,
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
    'client_after_recv: {
        DRIVER_CACHE.write().unwrap().insert_image(*module, image, is_runtime);
    }
    'server_execution_phos: {
        let _ = image__ptr;
        let hook_result = crate::phos::kernel::cu_module_load_data(module__ptr, image);
    }
}

#[cuda_hook(proc_id = 705)]
fn cuModuleGetFunction(
    #[handle(op = "create", op_key = proc_id as u64)] hfunc: *mut CUfunction,
    hmod: CUmodule,
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
    'server_execution_phos: {
        let _ = name__ptr;
        let hook_result = crate::phos::kernel::cu_module_get_function(hfunc__ptr, hmod, name);
    }
}

#[cuda_hook(proc_id = 640)]
fn cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult;

#[cuda_hook(proc_id = 630, async_api = false)]
fn cuInit(Flags: c_uint) -> CUresult;

#[cuda_custom_hook] // calls the internal API below if not opt_shadow_desc
fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;

#[cuda_hook(proc_id = 684, parent = cuCtxGetCurrent)]
fn cuCtxGetCurrentInternal(pctx: *mut CUcontext) -> CUresult;

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

#[cuda_hook(proc_id = 912, async_api)]
fn cuFuncSetAttribute(
    #[handle(op = "modify", op_key = (proc_id as u64) << 32 | (attrib as u64))] hfunc: CUfunction,
    attrib: CUfunction_attribute,
    value: c_int,
) -> CUresult;

#[cuda_hook(proc_id = 4000, min_cuda_version = 12)]
fn cuTensorMapEncodeTiled(
    tensorMap: *mut CUtensorMap,
    tensorDataType: CUtensorMapDataType,
    tensorRank: cuuint32_t,
    #[device] globalAddress: *mut c_void,
    #[host(len = tensorRank)] globalDim: *const cuuint64_t,
    #[host(len = tensorRank)] globalStrides: *const cuuint64_t,
    #[host(len = tensorRank)] boxDim: *const cuuint32_t,
    #[host(len = tensorRank)] elementStrides: *const cuuint32_t,
    interleave: CUtensorMapInterleave,
    swizzle: CUtensorMapSwizzle,
    l2Promotion: CUtensorMapL2promotion,
    oobFill: CUtensorMapFloatOOBfill,
) -> CUresult;

#[cuda_hook(proc_id = 800, async_api)]
fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CUresult {
    'server_execution_phos: {
        let hook_result = crate::phos::memory::cu_mem_address_free(ptr, size);
    }
}

#[cuda_hook(proc_id = 801)]
fn cuMemAddressReserve(
    ptr: *mut CUdeviceptr,
    size: usize,
    alignment: usize,
    addr: CUdeviceptr,
    flags: c_ulonglong,
) -> CUresult {
    'server_execution_phos: {
        let hook_result =
            crate::phos::memory::cu_mem_address_reserve(ptr__ptr, size, alignment, addr, flags);
    }
}

#[cuda_hook(proc_id = 802)]
fn cuMemCreate(
    handle: *mut CUmemGenericAllocationHandle,
    size: usize,
    #[host] prop: *const CUmemAllocationProp,
    flags: c_ulonglong,
) -> CUresult {
    'server_execution_phos: {
        let _ = prop__ptr;
        let hook_result = crate::phos::memory::cu_mem_create(handle__ptr, size, &prop, flags);
    }
}

#[cuda_hook(proc_id = 804)]
fn cuMemGetAllocationGranularity(
    granularity: *mut usize,
    #[host] prop: *const CUmemAllocationProp,
    option: CUmemAllocationGranularity_flags,
) -> CUresult;

#[cuda_hook(proc_id = 808, async_api)]
fn cuMemMap(
    ptr: CUdeviceptr,
    size: usize,
    offset: usize,
    handle: CUmemGenericAllocationHandle,
    flags: c_ulonglong,
) -> CUresult {
    'server_execution_phos: {
        let hook_result = crate::phos::memory::cu_mem_map(ptr, size, offset, handle, flags);
    }
}

#[cuda_hook(proc_id = 810, async_api)]
fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult {
    'server_execution_phos: {
        let hook_result = crate::phos::memory::cu_mem_release(handle);
    }
}

#[cuda_hook(proc_id = 812, async_api)]
fn cuMemSetAccess(
    ptr: CUdeviceptr,
    size: usize,
    #[host(len = count)] desc: *const CUmemAccessDesc,
    count: usize,
) -> CUresult {
    'server_execution_phos: {
        let _ = (desc__ptr, count);
        let hook_result = crate::phos::memory::cu_mem_set_access(ptr, size, &desc);
    }
}

#[cuda_hook(proc_id = 813, async_api)]
fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CUresult {
    'server_execution_phos: {
        let hook_result = crate::phos::memory::cu_mem_unmap(ptr, size);
    }
}

#[cuda_hook(proc_id = 913, async_api)]
fn cuFuncSetCacheConfig(
    #[handle(op = "modify", op_key = proc_id as u64)] hfunc: CUfunction,
    config: CUfunc_cache,
) -> CUresult;
