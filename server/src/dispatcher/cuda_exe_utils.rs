use std::os::raw::*;
use std::ptr;

use cudasys::cuda::*;

// We take the address of len to avoid returning a dangling local pointer.
fn build_extra_array(ptr: *const u8, len: &usize) -> [*mut c_void; 5] {
    [
        1 as _, // CU_LAUNCH_PARAM_BUFFER_POINTER
        ptr as _,
        2 as _, // CU_LAUNCH_PARAM_BUFFER_SIZE
        ptr::from_ref(len) as _,
        ptr::null_mut(), // CU_LAUNCH_PARAM_END
    ]
}

#[expect(clippy::too_many_arguments)]
pub fn cu_launch_kernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    args: &[u8],
) -> CUresult {
    unsafe {
        cuLaunchKernel(
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            ptr::null_mut(),
            build_extra_array(args.as_ptr(), &args.len()).as_mut_ptr(),
        )
    }
}

#[cfg(cuda_version = "12")]
pub fn cu_launch_kernel_ex(config: &CUlaunchConfig, f: CUfunction, args: &[u8]) -> CUresult {
    unsafe {
        cuLaunchKernelEx(
            config,
            f,
            ptr::null_mut(),
            build_extra_array(args.as_ptr(), &args.len()).as_mut_ptr(),
        )
    }
}

pub fn cu_func_get_attributes(
    attr: *mut cudasys::cudart::cudaFuncAttributes,
    func: CUfunction,
) -> CUresult {
    let attr = unsafe { &mut *attr };
    // HACK: implementation with cuFuncGetAttribute depends on CUDA version
    macro_rules! get_attributes {
        ($func:ident -> $struct:ident $($field:ident: $attr:ident,)+) => {
            $(
                let mut i = 0;
                let result =
                    unsafe { cuFuncGetAttribute(&raw mut i, CUfunction_attribute::$attr, $func) };
                if result != Default::default() {
                    return result;
                }
                $struct.$field = i as _;
            )+
        }
    }
    get_attributes! { func -> attr
        sharedSizeBytes: CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        constSizeBytes: CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
        localSizeBytes: CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        maxThreadsPerBlock: CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        numRegs: CU_FUNC_ATTRIBUTE_NUM_REGS,
        ptxVersion: CU_FUNC_ATTRIBUTE_PTX_VERSION,
        binaryVersion: CU_FUNC_ATTRIBUTE_BINARY_VERSION,
        cacheModeCA: CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
        maxDynamicSharedSizeBytes: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    }
    CUresult::CUDA_SUCCESS
}
