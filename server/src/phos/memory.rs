use std::collections::BTreeMap;
use std::ffi::{c_int, c_void};
use std::sync::Mutex;
use std::{mem, ptr};

use cudasys::cuda::*;
use cudasys::cudart::{cudaError_t, cudaFree};

#[cfg(target_pointer_width = "64")]
pub fn cuda_malloc(dev_ptr: *mut *mut c_void, size: usize) -> cudaError_t {
    match MEMORY_MANAGER.lock().unwrap().allocate(size) {
        Ok(ptr) => {
            unsafe { *dev_ptr = ptr as *mut c_void };
            cudaError_t::cudaSuccess
        }
        Err(e) => unsafe { mem::transmute(e) },
    }
}

#[cfg(target_pointer_width = "64")]
pub fn cuda_free(dev_ptr: *mut c_void) -> cudaError_t {
    match MEMORY_MANAGER.lock().unwrap().free(dev_ptr as CUdeviceptr) {
        CUresult::CUDA_ERROR_INVALID_VALUE => cudaError_t::cudaErrorInvalidDevicePointer,
        e => unsafe { mem::transmute(e) },
    }
}

pub static MEMORY_MANAGER: Mutex<DeviceMemoryManager> = Mutex::new(DeviceMemoryManager::new());

pub struct DeviceMemoryManager {
    inner: Option<DeviceMemoryManagerInner>,
}

impl DeviceMemoryManager {
    const fn new() -> Self {
        Self { inner: None }
    }

    /// Copied from [Option::get_or_insert_with]
    fn inner(&mut self) -> Result<&mut DeviceMemoryManagerInner, CUresult> {
        let inner = &mut self.inner;
        if let None = inner {
            *inner = Some(DeviceMemoryManagerInner::new()?);
        }
        Ok(unsafe { inner.as_mut().unwrap_unchecked() })
    }

    fn allocate(&mut self, size: usize) -> Result<CUdeviceptr, CUresult> {
        self.inner()?.allocate(size)
    }

    /// `cudaFree` should map [CUresult::CUDA_ERROR_INVALID_VALUE] to
    /// [cudaError_t::cudaErrorInvalidDevicePointer].
    fn free(&mut self, ptr: CUdeviceptr) -> CUresult {
        self.inner()?.free(ptr)
    }

    pub fn restore(&mut self, segments: &[(CUdeviceptr, usize)]) -> CUresult {
        assert!(self.inner.is_none());
        self.inner()?.restore(segments)
    }

    pub fn iter(&self) -> Option<impl Iterator<Item = (CUdeviceptr, usize)>> {
        self.inner.as_ref().map(DeviceMemoryManagerInner::iter)
    }

    pub fn reset(&mut self) {
        self.inner = None;
    }
}

struct DeviceMemoryManagerInner {
    granularity: usize,
    next: CUdeviceptr,
    allocations: BTreeMap<CUdeviceptr, cuda::MemoryMap>,
    range: cuda::AddressRange,         // drop after allocations
    guard: Option<cuda::ContextGuard>, // pop context after other API calls
    context: cuda::PrimaryContext,     // last to drop
}

// This is only used when manually resetting because static items are not dropped.
impl Drop for DeviceMemoryManagerInner {
    fn drop(&mut self) {
        self.guard = self.context.push_current().ok();
    }
}

impl DeviceMemoryManagerInner {
    const BASE: CUdeviceptr = 0x7facd0000000;

    fn new() -> Result<Self, CUresult> {
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#initialization
        unsafe { mem::transmute::<_, CUresult>(cudaFree(ptr::null_mut()))? };
        let mut device = 0;
        unsafe { cuCtxGetDevice(&mut device)? };
        let context = cuda::PrimaryContext::retain(device)?;
        let _guard = context.push_current()?;
        let granularity = cuda::get_alloc_granularity(device)?;
        assert!(Self::BASE.is_multiple_of(granularity as u64));
        let size = {
            let mut free = 0;
            unsafe { cuMemGetInfo_v2(&mut free, ptr::null_mut())? };
            let free_mb = free >> 20;
            if free_mb < 16 {
                log::error!("Not enough free memory on device {device}: {free_mb} MiB");
                return Err(CUresult::CUDA_ERROR_OUT_OF_MEMORY);
            }
            (free - free / 10).next_multiple_of(granularity) // 90%
        };
        Ok(Self {
            context,
            granularity,
            range: cuda::AddressRange::reserve(Self::BASE, size, granularity)?,
            allocations: BTreeMap::new(),
            next: Self::BASE,
            guard: None,
        })
    }

    fn device(&self) -> c_int {
        self.context.device()
    }

    fn allocate(&mut self, size: usize) -> Result<CUdeviceptr, CUresult> {
        let size = size.next_multiple_of(self.granularity);
        if self.next + size as u64 > self.range.end {
            log::error!(
                "Reserved address range exhausted: used {}/{} MiB, requesting {} MiB",
                (self.next - self.range.ptr) >> 20,
                (self.range.end - self.range.ptr) >> 20,
                size >> 20,
            );
            return Err(CUresult::CUDA_ERROR_OUT_OF_MEMORY);
        }
        let _guard = self.context.push_current()?;
        let ptr = self.next;
        self.next += size as u64;
        let map = cuda::MemoryMap::new(self.device(), ptr, size)?;
        let dupe = self.allocations.insert(ptr, map);
        assert!(dupe.is_none());
        Ok(ptr)
    }

    fn free(&mut self, ptr: CUdeviceptr) -> CUresult {
        if ptr == 0 {
            return CUresult::CUDA_SUCCESS;
        }
        let Some(allocation) = self.allocations.remove(&ptr) else {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        };
        let _guard = self.context.push_current()?;
        drop(allocation);
        CUresult::CUDA_SUCCESS
    }

    fn iter(&self) -> impl Iterator<Item = (CUdeviceptr, usize)> {
        self.allocations.values().map(|alloc| (alloc.ptr, alloc.size))
    }

    fn restore(&mut self, segments: &[(CUdeviceptr, usize)]) -> CUresult {
        if segments.is_empty() {
            return CUresult::CUDA_SUCCESS;
        }
        assert!(self.allocations.is_empty());
        let (last_ptr, last_size) = *segments.last().unwrap();
        let end = last_ptr + last_size as u64;
        assert!(
            end <= self.range.end,
            "failed to restore: reserved {} MiB but memory spans {} MiB",
            (self.range.end - self.range.ptr) >> 20,
            (end - self.range.ptr) >> 20,
        );
        self.next = end;
        let _guard = self.context.push_current()?;
        let mut allocations = Vec::with_capacity(segments.len());
        for segment in segments {
            let (ptr, size) = *segment;
            let map = cuda::MemoryMap::new(self.device(), ptr, size)?;
            allocations.push((ptr, map))
        }
        self.allocations = BTreeMap::from_iter(allocations);
        CUresult::CUDA_SUCCESS
    }
}

mod cuda {
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
            unsafe { cuDevicePrimaryCtxRelease_v2(self.device) };
        }
    }

    pub struct ContextGuard(PhantomData<()>);

    impl Drop for ContextGuard {
        fn drop(&mut self) {
            unsafe { cuCtxPopCurrent_v2(ptr::null_mut()) };
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
            assert_eq!(ptr, addr);
            Ok(Self { ptr, end: ptr + size as u64 })
        }
    }

    impl Drop for AddressRange {
        fn drop(&mut self) {
            unsafe { cuMemAddressFree(self.ptr, (self.end - self.ptr) as usize) };
        }
    }

    struct MemoryHandle(CUmemGenericAllocationHandle);

    impl MemoryHandle {
        fn new(device: c_int, size: usize) -> Result<Self, CUresult> {
            let mut handle = 0;
            unsafe { cuMemCreate(&mut handle, size, &mem_prop_for(device), 0)? };
            Ok(Self(handle))
        }
    }

    impl Drop for MemoryHandle {
        fn drop(&mut self) {
            unsafe { cuMemRelease(self.0) };
        }
    }

    pub struct MemoryMap {
        pub ptr: CUdeviceptr,
        pub size: usize,
        _handle: MemoryHandle,
    }

    impl MemoryMap {
        pub fn new(device: c_int, ptr: CUdeviceptr, size: usize) -> Result<Self, CUresult> {
            let handle = MemoryHandle::new(device, size)?;
            unsafe { cuMemMap(ptr, size, 0, handle.0, 0)? };
            let map = Self { ptr, size, _handle: handle };
            map.set_access(device)?;
            Ok(map)
        }

        fn set_access(&self, device: c_int) -> CUresult {
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
            unsafe { cuMemUnmap(self.ptr, self.size) };
        }
    }

    fn mem_prop_for(device: c_int) -> CUmemAllocationProp {
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
}
