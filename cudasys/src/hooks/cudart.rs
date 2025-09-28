use crate::types::cudart::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

// Hacked via type alias because imports are rewritten in generated code.
type CUfunction = crate::types::cuda::CUfunction;

#[cuda_hook(proc_id = 120)]
fn cudaGetDevice(device: *mut c_int) -> cudaError_t {
    'client_before_send: {
        if let (true, Some(val)) = (client.opt_local, client.cuda_device) {
            unsafe {
                *device = val;
            }
            return cudaError_t::cudaSuccess;
        }
    }
    'client_after_recv: {
        {
            client.cuda_device = Some(*device);
        }
    }
}

#[cuda_hook(proc_id = 129)]
fn cudaSetDevice(device: c_int) -> cudaError_t {
    'client_after_recv: {
        if hook_result == Default::default() {
            client.cuda_device = Some(device);
        } else {
            client.cuda_device = None;
        }
    }
}

#[cuda_hook(proc_id = 121)]
fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;

#[cuda_hook(proc_id = 152, async_api = false)]
fn cudaGetLastError() -> cudaError_t;

#[cuda_hook(proc_id = 153, async_api = false)]
fn cudaPeekAtLastError() -> cudaError_t;

#[cuda_hook(proc_id = 178, async_api = false)]
fn cudaStreamSynchronize(#[handle = "use"] stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 265)]
fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;

#[cuda_custom_hook] // calls one of the following internal APIs
fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 320, async_api, parent = cudaMemcpy)]
fn cudaMemcpyHtod(
    #[device] dst: *mut c_void,
    #[host(len = count)] src: *const c_void,
    count: usize,
    #[value = cudaMemcpyKind::cudaMemcpyHostToDevice] kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 321, parent = cudaMemcpy)]
fn cudaMemcpyDtoh(
    #[host(output, len = count)] dst: *mut c_void,
    #[device] src: *const c_void,
    count: usize,
    #[value = cudaMemcpyKind::cudaMemcpyDeviceToHost] kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 322, async_api, parent = cudaMemcpy)]
fn cudaMemcpyDtod(
    #[device] dst: *mut c_void,
    #[device] src: *const c_void,
    count: usize,
    #[value = cudaMemcpyKind::cudaMemcpyDeviceToDevice] kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // calls one of the following internal APIs
fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 323, async_api, parent = cudaMemcpyAsync)]
fn cudaMemcpyAsyncHtod(
    #[device] dst: *mut c_void,
    #[host(len = count)] src: *const c_void,
    count: usize,
    #[value = cudaMemcpyKind::cudaMemcpyHostToDevice] kind: cudaMemcpyKind,
    #[handle = "use"] stream: cudaStream_t,
) -> cudaError_t {
    'server_execution: {
        // FIXME: can't async because server deallocates memory after calling
        let hook_result = unsafe {
            assert_eq!(cudaStreamSynchronize(stream), Default::default());
            cudaMemcpy(dst, src__ptr.cast(), count, cudaMemcpyKind::cudaMemcpyHostToDevice)
        };
    }
}

#[cuda_hook(proc_id = 324, parent = cudaMemcpyAsync)]
fn cudaMemcpyAsyncDtoh(
    #[host(output, len = count)] dst: *mut c_void,
    #[device] src: *const c_void,
    count: usize,
    #[value = cudaMemcpyKind::cudaMemcpyDeviceToHost] kind: cudaMemcpyKind,
    #[handle = "use"] stream: cudaStream_t,
) -> cudaError_t {
    'server_execution: {
        // FIXME: can't async because server can't send data back async
        let hook_result = unsafe {
            assert_eq!(cudaStreamSynchronize(stream), Default::default());
            cudaMemcpy(dst__ptr.cast(), src, count, cudaMemcpyKind::cudaMemcpyDeviceToHost)
        };
    }
}

#[cuda_hook(proc_id = 325, async_api, parent = cudaMemcpyAsync)]
fn cudaMemcpyAsyncDtod(
    #[device] dst: *mut c_void,
    #[device] src: *const c_void,
    count: usize,
    #[value = cudaMemcpyKind::cudaMemcpyDeviceToDevice] kind: cudaMemcpyKind,
    #[handle = "use"] stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 253, async_api)]
fn cudaFree(#[device] devPtr: *mut c_void) -> cudaError_t;

#[cuda_hook(proc_id = 175)]
fn cudaStreamIsCapturing(
    #[handle = "use"] stream: cudaStream_t,
    pCaptureStatus: *mut cudaStreamCaptureStatus,
) -> cudaError_t;

// This function is hidden and superseded by `cudaGetDeviceProperties_v2` in CUDA 12.
// The change is that `cudaDeviceProp` grew bigger. We don't hook it in CUDA 12
// to prevent reading or writing past the end of allocated memory when sending or receiving data.
#[cuda_hook(proc_id = 123, max_cuda_version = 11)]
fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 123, min_cuda_version = 13)]
fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 400)]
fn cudaPointerGetAttributes(
    attributes: *mut cudaPointerAttributes,
    #[device] ptr: *const c_void,
) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaHostAlloc(pHost: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

#[cuda_custom_hook] // calls the internal API below
fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: *const c_void) -> cudaError_t;

#[cuda_hook(proc_id = 230, parent = cudaFuncGetAttributes)]
fn cudaFuncGetAttributesInternal(
    attr: *mut cudaFuncAttributes,
    #[handle = "use"] func: CUfunction,
) -> cudaError_t {
    'server_execution: {
        unsafe { attr__ptr.write_bytes(0u8, 1) };
        let hook_result = super::cuda_exe_utils::cu_func_get_attributes(attr__ptr, func);
    }
}

#[cuda_custom_hook] // local
fn __cudaRegisterFatBinary(fatCubin: *mut c_void) -> *mut *mut c_void;

#[cuda_custom_hook] // local
fn __cudaRegisterFatBinaryEnd(fatCubinHandle: *mut *mut c_void);

#[cuda_custom_hook] // local
fn __cudaUnregisterFatBinary(fatCubinHandle: *mut *mut c_void);

#[cuda_custom_hook] // local
fn __cudaRegisterFunction(
    fatCubinHandle: *mut *mut c_void,
    hostFun: *const c_char,
    deviceFun: *mut c_char,
    deviceName: *const c_char,
    thread_limit: c_int,
    tid: *mut uint3,
    bid: *mut uint3,
    bDim: *mut dim3,
    gDim: *mut dim3,
    wSize: *mut c_int,
);

#[cuda_custom_hook] // local
fn __cudaRegisterVar(
    fatCubinHandle: *mut *mut c_void,
    hostVar: *mut c_char,
    deviceAddress: *mut c_char,
    deviceName: *const c_char,
    ext: c_int,
    size: usize,
    constant: c_int,
    global: c_int,
);

#[cuda_custom_hook] // calls driver API
fn cudaLaunchKernel(
    func: *const c_void,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 112)]
fn cudaDeviceGetStreamPriorityRange(
    leastPriority: *mut c_int,
    greatestPriority: *mut c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 302, async_api)]
fn cudaMemsetAsync(
    #[device] devPtr: *mut c_void,
    value: c_int,
    count: usize,
    #[handle = "use"] stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaGetErrorString(error: cudaError_t) -> *const c_char;

#[cuda_hook(proc_id = 274)]
fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;

#[cuda_custom_hook] // local
fn __cudaPushCallConfiguration(
    gridDim: dim3,
    blockDim: dim3,
    sharedMem: usize,
    stream: *mut CUstream_st,
) -> c_uint;

#[cuda_custom_hook] // local
fn __cudaPopCallConfiguration(
    gridDim: *mut dim3,
    blockDim: *mut dim3,
    sharedMem: *mut usize,
    stream: *mut c_void,
) -> cudaError_t;

#[cuda_hook(proc_id = 999123, min_cuda_version = 12, max_cuda_version = 12)]
fn cudaGetDeviceProperties_v2(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 167)]
fn cudaStreamCreateWithPriority(
    #[handle = "create"] pStream: *mut cudaStream_t,
    flags: c_uint,
    priority: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 201)]
fn cudaEventCreateWithFlags(
    #[handle = "create"] event: *mut cudaEvent_t,
    flags: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 205)]
fn cudaEventRecord(
    #[handle = "modify"] event: cudaEvent_t,
    #[handle = "use"] stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 180, async_api = false)]
fn cudaStreamWaitEvent(
    #[handle = "use"] stream: cudaStream_t,
    #[handle = "use"] event: cudaEvent_t,
    flags: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 202)]
fn cudaEventDestroy(#[handle = "destroy"] event: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 102)]
fn cudaDeviceGetAttribute(value: *mut c_int, attr: cudaDeviceAttr, device: c_int) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *mut c_int,
    func: *const c_void,
    blockSize: c_int,
    dynamicSMemSize: usize,
    flags: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 126)]
fn cudaIpcGetMemHandle(
    handle: *mut cudaIpcMemHandle_t,
    #[device] devPtr: *mut c_void,
) -> cudaError_t;

#[cuda_hook(proc_id = 128)]
fn cudaIpcOpenMemHandle(
    devPtr: *mut *mut c_void,
    handle: cudaIpcMemHandle_t,
    flags: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 204, async_api = false)]
fn cudaEventQuery(#[handle = "use"] event: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 119, async_api = false)]
fn cudaDeviceSynchronize() -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaFuncSetAttribute(func: *const c_void, attr: cudaFuncAttribute, value: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 203)]
fn cudaEventElapsedTime(
    ms: *mut f32,
    #[handle = "use"] start: cudaEvent_t,
    #[handle = "use"] end: cudaEvent_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 412, async_api = false)]
fn cudaDeviceEnablePeerAccess(peerDevice: c_int, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 297, async_api)]
fn cudaMemset(#[device] devPtr: *mut c_void, value: c_int, count: usize) -> cudaError_t;

#[cuda_hook(proc_id = 114, async_api = false)]
fn cudaDeviceReset() -> cudaError_t;

#[cuda_hook(proc_id = 163, async_api = false)]
fn cudaStreamBeginCapture(
    #[handle = "use"] stream: cudaStream_t,
    mode: cudaStreamCaptureMode,
) -> cudaError_t;

#[cuda_hook(proc_id = 169)]
fn cudaStreamEndCapture(
    #[handle = "use"] stream: cudaStream_t,
    pGraph: *mut cudaGraph_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 172, max_cuda_version = 12)]
fn cudaStreamGetCaptureInfo_v2(
    #[handle = "use"] stream: cudaStream_t,
    captureStatus_out: *mut cudaStreamCaptureStatus,
    id_out: *mut c_ulonglong,
    #[value = std::ptr::null_mut()] graph_out: *mut cudaGraph_t,
    #[value = std::ptr::null_mut()] dependencies_out: *mut *const cudaGraphNode_t,
    #[value = std::ptr::null_mut()] numDependencies_out: *mut usize,
) -> cudaError_t;

// We use hooks to implement the inout parameter `mode` for now.
#[cuda_hook(proc_id = 181)]
fn cudaThreadExchangeStreamCaptureMode(mode: *mut cudaStreamCaptureMode) -> cudaError_t {
    'client_extra_send: {
        send_ctx.send(unsafe { &*mode }, "mode");
    }
    'server_extra_recv: {
        let mode_in = recv_ctx.recv("mode");
    }
    'server_before_execution: {
        mode.write(mode_in);
    }
}

#[cuda_hook(proc_id = 510)]
fn cudaDriverGetVersion(driverVersion: *mut c_int) -> cudaError_t;

#[cuda_hook(proc_id = 538, async_api = false)]
fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;

#[cuda_hook(proc_id = 545, async_api = false)]
fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t;

#[cuda_hook(proc_id = 563)]
fn cudaGraphGetNodes(
    graph: cudaGraph_t,
    #[value = std::ptr::null_mut()] nodes: *mut cudaGraphNode_t,
    numNodes: *mut usize,
) -> cudaError_t;

#[cuda_hook(proc_id = 567, min_cuda_version = 12)]
fn cudaGraphInstantiate(
    pGraphExec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    flags: c_ulonglong,
) -> cudaError_t;

#[cuda_hook(proc_id = 999567, min_cuda_version = 12)]
fn cudaGraphInstantiateWithFlags(
    pGraphExec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    flags: c_ulonglong,
) -> cudaError_t;

#[cuda_hook(proc_id = 573, async_api)]
fn cudaGraphLaunch(
    graphExec: cudaGraphExec_t,
    #[handle = "use"] stream: cudaStream_t,
) -> cudaError_t;
