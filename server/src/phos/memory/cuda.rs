use std::ffi::c_int;
use std::marker::PhantomData;
use std::{mem, ptr};

use cudasys::cuda::*;

pub struct PrimaryContext {
    context: CUcontext,
    device: c_int,
}

unsafe impl Send for PrimaryContext {}

impl PrimaryContext {
    pub fn retain(ordinal: c_int) -> Result<Self, CUresult> {
        let mut device = 0;
        unsafe { cuDeviceGet(&mut device, ordinal)? };
        assert_eq!(device, ordinal);
        let mut context = ptr::null_mut();
        unsafe { cuDevicePrimaryCtxRetain(&mut context, device)? };
        Ok(Self { context, device })
    }

    pub fn device(&self) -> c_int {
        self.device
    }

    pub fn push_current(&self) -> Result<ContextGuard, CUresult> {
        unsafe { cuCtxPushCurrent_v2(self.context)? };
        Ok(ContextGuard(PhantomData))
    }
}

impl Drop for PrimaryContext {
    fn drop(&mut self) {
        let result = unsafe { cuDevicePrimaryCtxRelease_v2(self.device) };
        debug_assert_eq!(result, Default::default());
    }
}

pub struct ContextGuard(PhantomData<()>);

impl Drop for ContextGuard {
    fn drop(&mut self) {
        let result = unsafe { cuCtxPopCurrent_v2(ptr::null_mut()) };
        debug_assert_eq!(result, Default::default());
    }
}

pub struct AddressRange {
    pub ptr: CUdeviceptr,
    pub end: CUdeviceptr,
}

impl AddressRange {
    pub fn reserve(addr: CUdeviceptr, size: usize, alignment: usize) -> Result<Self, CUresult> {
        let mut ptr = 0;
        unsafe { cuMemAddressReserve(&mut ptr, size, alignment, addr, 0)? };
        if ptr != addr {
            log::error!("Reserved address range: requested {addr:#x}, got {ptr:#x}");
        }
        Ok(Self { ptr, end: ptr + size as u64 })
    }
}

impl Drop for AddressRange {
    fn drop(&mut self) {
        let result = unsafe { cuMemAddressFree(self.ptr, (self.end - self.ptr) as usize) };
        debug_assert_eq!(result, Default::default());
    }
}

pub struct MemoryHandle(CUmemGenericAllocationHandle);

impl MemoryHandle {
    pub fn new(device: c_int, size: usize) -> Result<Self, CUresult> {
        Self::new_with_prop(size, &mem_prop_for(device))
    }

    pub fn new_with_prop(size: usize, prop: &CUmemAllocationProp) -> Result<Self, CUresult> {
        let mut handle = 0;
        unsafe { cuMemCreate(&mut handle, size, prop, 0)? };
        Ok(Self(handle))
    }

    pub fn get_prop(&self) -> Result<CUmemAllocationProp, CUresult> {
        let mut prop = mem::MaybeUninit::<CUmemAllocationProp>::uninit();
        unsafe { cuMemGetAllocationPropertiesFromHandle(prop.as_mut_ptr(), self.0)? };
        Ok(unsafe { prop.assume_init() })
    }
}

impl Drop for MemoryHandle {
    fn drop(&mut self) {
        let result = unsafe { cuMemRelease(self.0) };
        debug_assert_eq!(result, Default::default());
    }
}

pub struct MemoryMap {
    pub ptr: CUdeviceptr,
    pub size: usize,
}

impl MemoryMap {
    pub fn new(ptr: CUdeviceptr, size: usize, handle: &MemoryHandle) -> Result<Self, CUresult> {
        unsafe { cuMemMap(ptr, size, 0, handle.0, 0)? };
        Ok(Self { ptr, size })
    }

    pub fn set_access(&self, device: c_int) -> CUresult {
        let desc = CUmemAccessDesc {
            location: CUmemLocation {
                type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: device,
            },
            flags: CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        unsafe { cuMemSetAccess(self.ptr, self.size, &desc, 1) }
    }
}

impl Drop for MemoryMap {
    fn drop(&mut self) {
        let result = unsafe { cuMemUnmap(self.ptr, self.size) };
        debug_assert_eq!(result, Default::default());
    }
}

const fn mem_prop_for(device: c_int) -> CUmemAllocationProp {
    CUmemAllocationProp {
        type_: CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        location: CUmemLocation {
            type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: device,
        },
        ..unsafe { mem::zeroed() }
    }
}

pub fn get_alloc_granularity(device: c_int) -> Result<usize, CUresult> {
    let mut granularity = 0;
    unsafe {
        cuMemGetAllocationGranularity(
            &mut granularity,
            &mem_prop_for(device),
            CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )?
    };
    Ok(granularity)
}
