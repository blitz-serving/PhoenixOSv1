#![expect(non_snake_case)]
use std::cell::RefCell;
use std::ffi::*;
use std::{mem, ptr};

use cudasys::types::cudart::*;

use super::*;

fn is_device_ptr(ptr: *const c_void) -> bool {
    let mut attrs = mem::MaybeUninit::uninit();
    let result = super::cudart_hijack::cudaPointerGetAttributes(attrs.as_mut_ptr(), ptr);
    assert_eq!(result, cudaError_t::cudaSuccess);
    match unsafe { attrs.assume_init() }.type_ {
        cudaMemoryType::cudaMemoryTypeUnregistered | cudaMemoryType::cudaMemoryTypeHost => false,
        cudaMemoryType::cudaMemoryTypeDevice => true,
        cudaMemoryType::cudaMemoryTypeManaged => cuda_unimplemented("cudaMemoryTypeManaged"),
    }
}

fn infer_memcpy_kind(dst: *const c_void, src: *const c_void) -> cudaMemcpyKind {
    match (is_device_ptr(dst), is_device_ptr(src)) {
        (false, false) => cudaMemcpyKind::cudaMemcpyHostToHost,
        (true, false) => cudaMemcpyKind::cudaMemcpyHostToDevice,
        (false, true) => cudaMemcpyKind::cudaMemcpyDeviceToHost,
        (true, true) => cudaMemcpyKind::cudaMemcpyDeviceToDevice,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    mut kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpy", "kind = {kind:?}");
    if kind == cudaMemcpyKind::cudaMemcpyDefault {
        kind = infer_memcpy_kind(dst, src);
        log::debug!(target: "cudaMemcpy", "inferred kind = {kind:?}");
    }
    match kind {
        cudaMemcpyKind::cudaMemcpyHostToHost => unsafe {
            ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
            cudaError_t::cudaSuccess
        },
        cudaMemcpyKind::cudaMemcpyHostToDevice => {
            super::cudart_hijack::cudaMemcpyAsyncHtod(dst, src.cast(), count, kind, ptr::null_mut())
        }
        cudaMemcpyKind::cudaMemcpyDeviceToHost => {
            super::cudart_hijack::cudaMemcpyAsyncDtoh(dst.cast(), src, count, kind, ptr::null_mut())
        }
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyDtod(dst, src, count, kind)
        }
        cudaMemcpyKind::cudaMemcpyDefault => unreachable!(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    mut kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyAsync", "kind = {kind:?}");
    if kind == cudaMemcpyKind::cudaMemcpyDefault {
        kind = infer_memcpy_kind(dst, src);
        log::debug!(target: "cudaMemcpyAsync", "inferred kind = {kind:?}");
    }
    match kind {
        cudaMemcpyKind::cudaMemcpyHostToHost => unsafe {
            super::cudart_hijack::cudaStreamSynchronize(stream);
            ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
            cudaError_t::cudaSuccess
        },
        cudaMemcpyKind::cudaMemcpyHostToDevice => {
            super::cudart_hijack::cudaMemcpyAsyncHtod(dst, src.cast(), count, kind, stream)
        }
        cudaMemcpyKind::cudaMemcpyDeviceToHost => {
            super::cudart_hijack::cudaMemcpyAsyncDtoh(dst.cast(), src, count, kind, stream)
        }
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyAsyncDtod(dst, src, count, kind, stream)
        }
        cudaMemcpyKind::cudaMemcpyDefault => unreachable!(),
    }
}

fn get_cufunction(func: HostPtr) -> cudasys::cuda::CUfunction {
    if !ClientThread::with_borrow(|client| client.cuda_device_init) {
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#initialization
        assert_eq!(super::cudart_hijack::cudaFree(ptr::null_mut()), Default::default());
        ClientThread::with_borrow_mut(|client| client.cuda_device_init = true);
    }

    if let Some(&cufunc) = RUNTIME_CACHE.read().unwrap().loaded_functions.get(&func) {
        return cufunc;
    }

    let runtime = &mut *RUNTIME_CACHE.write().unwrap();

    // TODO: In CUDA 12, use `cuLibrary{LoadData,GetKernel}` to avoid pinning device.
    if let Some(device) = runtime.cuda_device {
        assert_eq!(
            ClientThread::with_borrow(|client| client.cuda_device),
            Some(device),
            "current device (left) and registered device (right) mismatch",
        );
    } else {
        log::info!(
            "#fatbins = {}, #functions = {}",
            runtime.lazy_fatbins.len(),
            runtime.lazy_functions.len(),
        );

        let mut device = 0;
        assert_eq!(super::cudart_hijack::cudaGetDevice(&mut device), Default::default());
        runtime.cuda_device = Some(device);
    }

    let load_module = |fatCubinHandle: &FatBinaryHandle| {
        let index = fatCubinHandle.to_index();
        log::debug!("registering fatbin #{index}");
        let image = runtime.lazy_fatbins[index];
        let mut module = ptr::null_mut();
        assert_eq!(
            super::cuda_hijack::cuModuleLoadDataInternal(&raw mut module, image.cast(), true),
            Default::default(),
        );
        module
    };

    let (fatCubinHandle, deviceName) = *runtime.lazy_functions.get(&func).unwrap();
    let module = *runtime.loaded_modules.entry(fatCubinHandle).or_insert_with_key(load_module);
    log::debug!("registering function {:?}", unsafe { CStr::from_ptr(deviceName) });
    let mut cufunc = ptr::null_mut();
    assert_eq!(
        super::cuda_hijack::cuModuleGetFunction(&raw mut cufunc, module, deviceName),
        Default::default(),
    );
    runtime.loaded_functions.insert(func, cufunc);
    cufunc
}

#[unsafe(no_mangle)]
pub extern "C" fn cudaLaunchKernel(
    func: HostPtr,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaLaunchKernel", "");

    let cufunc = get_cufunction(func);

    unsafe {
        mem::transmute(super::cuda_hijack::cuLaunchKernel(
            cufunc,
            gridDim.x,
            gridDim.y,
            gridDim.z,
            blockDim.x,
            blockDim.y,
            blockDim.z,
            sharedMem.try_into().unwrap(),
            stream.cast(),
            args,
            ptr::null_mut(),
        ))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cudaHostAlloc(
    pHost: *mut *mut c_void,
    size: usize,
    flags: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaHostAlloc", "size = {size}, flags = {flags}");
    assert_eq!(flags, cudaHostAllocDefault);
    // TODO: handle pinned memory at server side in a better way
    // FIXME: some GPU kernels might write to pinned memory directly; currently CUDA will report illegal memory access
    let ptr = Box::into_raw(Box::<[u8]>::new_uninit_slice(size));
    unsafe {
        *pHost = ptr as _;
    }
    cudaError_t::cudaSuccess
}

#[unsafe(no_mangle)]
extern "C" fn cudaGetErrorName(cudaError: cudaError_t) -> *const c_char {
    log::debug!(target: "cudaGetErrorName", "{cudaError:?}");
    let result = format!("{cudaError:?}");
    let result = CString::new(result).unwrap();
    result.into_raw() // leaking the string as the program is about to fail anyway
}

#[unsafe(no_mangle)]
extern "C" fn cudaGetErrorString(cudaError: cudaError_t) -> *const c_char {
    log::debug!(target: "cudaGetErrorString", "{cudaError:?}");
    let result = format!("{cudaError:?} ({})", cudaError as u32);
    let result = CString::new(result).unwrap();
    result.into_raw() // leaking the string as the program is about to fail anyway
}

struct CallConfiguration {
    gridDim: dim3,
    blockDim: dim3,
    sharedMem: usize,
    stream: usize,
}

thread_local! {
    static CALL_CONFIGURATIONS: RefCell<Vec<CallConfiguration>> = const {
        RefCell::new(Vec::new())
    };
}

#[unsafe(no_mangle)]
pub extern "C" fn __cudaPushCallConfiguration(
    gridDim: dim3,
    blockDim: dim3,
    sharedMem: usize,
    stream: usize,
) -> cudaError_t {
    CALL_CONFIGURATIONS.with_borrow_mut(|v| {
        v.push(CallConfiguration { gridDim, blockDim, sharedMem, stream });
    });
    cudaError_t::cudaSuccess
}

#[unsafe(no_mangle)]
pub extern "C" fn __cudaPopCallConfiguration(
    gridDim: *mut dim3,
    blockDim: *mut dim3,
    sharedMem: *mut usize,
    stream: *mut usize,
) -> cudaError_t {
    if let Some(config) = CALL_CONFIGURATIONS.with_borrow_mut(Vec::pop) {
        unsafe {
            *gridDim = config.gridDim;
            *blockDim = config.blockDim;
            *sharedMem = config.sharedMem;
            *stream = config.stream;
        }
        cudaError_t::cudaSuccess
    } else {
        cudaError_t::cudaErrorMissingConfiguration
    }
}

#[unsafe(no_mangle)]
extern "C" fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: HostPtr) -> cudaError_t {
    log::debug!(target: "cudaFuncGetAttributes", "");
    let func = get_cufunction(func);
    super::cudart_hijack::cudaFuncGetAttributesInternal(attr, func)
}

#[unsafe(no_mangle)]
extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *mut c_int,
    func: HostPtr,
    blockSize: c_int,
    dynamicSMemSize: usize,
    flags: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", "");
    let result = super::cuda_hijack::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks,
        get_cufunction(func),
        blockSize,
        dynamicSMemSize,
        flags,
    );
    unsafe { mem::transmute(result) }
}

#[unsafe(no_mangle)]
extern "C" fn cudaFuncSetAttribute(
    func: HostPtr,
    attr: cudaFuncAttribute,
    value: c_int,
) -> cudaError_t {
    log::debug!(target: "cudaFuncSetAttribute", "");
    unsafe {
        mem::transmute(super::cuda_hijack::cuFuncSetAttribute(
            get_cufunction(func),
            mem::transmute(attr),
            value,
        ))
    }
}

#[unsafe(no_mangle)]
extern "C" fn __cudaGetKernel(arg1: *mut cudaKernel_t, arg2: HostPtr) -> cudaError_t {
    unsafe { *arg1 = mem::transmute(arg2) };
    cudaError_t::cudaSuccess
}
