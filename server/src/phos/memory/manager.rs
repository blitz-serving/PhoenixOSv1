use std::collections::btree_map::{BTreeMap, Entry};
use std::ffi::c_int;
use std::sync::Mutex;
use std::{mem, ptr};

use bincode::{Decode, Encode};
use cudasys::cuda::*;
use cudasys::cudart::cudaFree;

use super::super::ByteSize;
use super::cuda;

pub static MEMORY_MANAGER: Mutex<DeviceMemoryManager> = Mutex::new(DeviceMemoryManager::new());

pub struct DeviceMemoryManager {
    inner: Option<DeviceMemoryManagerInner>,
}

impl DeviceMemoryManager {
    const fn new() -> Self {
        Self { inner: None }
    }

    /// Copied from [Option::get_or_insert_with]
    pub fn inner(&mut self) -> Result<&mut DeviceMemoryManagerInner, CUresult> {
        let inner = &mut self.inner;
        if inner.is_none() {
            *inner = Some(DeviceMemoryManagerInner::new()?);
        }
        Ok(unsafe { inner.as_mut().unwrap_unchecked() })
    }

    pub fn allocate(&mut self, size: usize) -> Result<CUdeviceptr, CUresult> {
        self.inner()?.allocate(size)
    }

    /// `cudaFree` should map [CUresult::CUDA_ERROR_INVALID_VALUE] to
    /// [cudaError_t::cudaErrorInvalidDevicePointer].
    pub fn free(&mut self, ptr: CUdeviceptr) -> CUresult {
        self.inner()?.free(ptr)
    }

    pub fn checkpoint_meta(&mut self) -> Result<MemoryCheckpointMeta, CUresult> {
        self.inner()?.checkpoint_meta()
    }

    pub fn restore(&mut self, meta: &MemoryCheckpointMeta) -> CUresult {
        self.inner()?.restore(meta)
    }

    pub fn reset(&mut self) {
        self.inner = None;
    }
}

pub struct DeviceMemoryManagerInner {
    granularity: usize,
    next: CUdeviceptr,
    // Drop order: memory maps must be dropped before releasing handles and freeing address
    allocations: BTreeMap<CUdeviceptr, MemoryMapSlot>,
    user_handles: BTreeMap<u64, (cuda::MemoryHandle, usize)>,
    range: cuda::AddressRange,
    guard: Option<cuda::ContextGuard>, // pop context after other API calls
    context: cuda::PrimaryContext,     // last to drop
}

struct MemoryMapSlot {
    map: cuda::MemoryMap, // first unmap
    handle: AllocHandle,  // then release
}

enum AllocHandle {
    Owned { _handle: cuda::MemoryHandle },
    Borrowed { proxy: u64 },
}

// This is only used when manually resetting because static items are not dropped.
impl Drop for DeviceMemoryManagerInner {
    fn drop(&mut self) {
        self.guard = self.context.push_current().ok();
    }
}

impl DeviceMemoryManagerInner {
    // 0x7facd0000000: slot not large enough
    // 0x802cce000000: cuMemSetAccess errors on A800 + CUDA 13
    const BASE: CUdeviceptr = 0x10004000000;

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
            let mut total = 0;
            unsafe { cuDeviceTotalMem_v2(&mut total, device)? };
            (total * 4).next_multiple_of(granularity) // TODO: ad-hoc
        };
        let range = cuda::AddressRange::reserve(Self::BASE, size, granularity)?;
        Ok(Self {
            context,
            granularity,
            next: range.ptr,
            range,
            allocations: BTreeMap::new(),
            user_handles: BTreeMap::new(),
            guard: None,
        })
    }

    fn device(&self) -> c_int {
        self.context.device()
    }

    pub fn reserve_address(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<CUdeviceptr, CUresult> {
        assert!(alignment == 0 || self.granularity.is_multiple_of(alignment));
        self.reserve_aligned(size.next_multiple_of(self.granularity))
    }

    fn reserve_aligned(&mut self, size: usize) -> Result<CUdeviceptr, CUresult> {
        debug_assert!(size.is_multiple_of(self.granularity));
        if self.next + size as u64 > self.range.end {
            log::error!(
                "Reserved address range exhausted: used {}/{}, requesting {}",
                ByteSize(self.next - self.range.ptr),
                ByteSize(self.range.end - self.range.ptr),
                ByteSize(size as u64),
            );
            return Err(CUresult::CUDA_ERROR_OUT_OF_MEMORY);
        }
        let ptr = self.next;
        self.next += size as u64;
        Ok(ptr)
    }

    pub fn allocate(&mut self, size: usize) -> Result<CUdeviceptr, CUresult> {
        let size = size.next_multiple_of(self.granularity);
        let ptr = self.reserve_aligned(size)?;
        let _guard = self.context.push_current()?;
        let handle = cuda::MemoryHandle::new(self.device(), size)?;
        let map = cuda::MemoryMap::new(ptr, size, &handle)?;
        map.set_access(self.device())?;
        let allocation = MemoryMapSlot { map, handle: AllocHandle::Owned { _handle: handle } };
        let dupe = self.allocations.insert(ptr, allocation);
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

    pub fn create_user_handle(
        &mut self,
        size: usize,
        prop: &CUmemAllocationProp,
    ) -> Result<u64, CUresult> {
        assert_eq!(prop.location.type_, CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE);
        assert_eq!(prop.location.id, self.device());
        let proxy = match self.user_handles.last_key_value() {
            Some((k, _)) => *k + 1,
            None => 1,
        };
        let _guard = self.context.push_current()?;
        let handle = cuda::MemoryHandle::new_with_prop(size, prop)?;
        let dupe = self.user_handles.insert(proxy, (handle, size));
        assert!(dupe.is_none());
        Ok(proxy)
    }

    pub fn map_user_handle(&mut self, ptr: CUdeviceptr, size: usize, proxy: u64) -> CUresult {
        let Some((handle, _)) = self.user_handles.get(&proxy) else {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        };
        let Entry::Vacant(entry) = self.allocations.entry(ptr) else {
            return CUresult::CUDA_ERROR_ALREADY_MAPPED;
        };
        let _guard = self.context.push_current()?;
        let map = cuda::MemoryMap::new(ptr, size, handle)?;
        entry.insert(MemoryMapSlot { map, handle: AllocHandle::Borrowed { proxy } });
        CUresult::CUDA_SUCCESS
    }

    pub fn set_access(
        &mut self,
        ptr: CUdeviceptr,
        size: usize,
        desc: &CUmemAccessDesc,
    ) -> CUresult {
        if desc.location.type_ != CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE
            || desc.location.id != self.device()
            || desc.flags != CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let _guard = self.context.push_current()?;
        unsafe { cuMemSetAccess(ptr, size, desc, 1) }
    }

    pub fn unmap(&mut self, ptr: CUdeviceptr, size: usize) -> CUresult {
        let Entry::Occupied(entry) = self.allocations.entry(ptr) else {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        };
        let allocation = entry.get();
        if matches!(allocation.handle, AllocHandle::Owned { .. }) || allocation.map.size != size {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let allocation = entry.remove();
        let _guard = self.context.push_current()?;
        drop(allocation);
        CUresult::CUDA_SUCCESS
    }

    pub fn release_user_handle(&mut self, proxy: u64) -> CUresult {
        let Some((handle, _)) = self.user_handles.remove(&proxy) else {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        };
        let _guard = self.context.push_current()?;
        drop(handle);
        CUresult::CUDA_SUCCESS
    }

    fn checkpoint_meta(&mut self) -> Result<MemoryCheckpointMeta, CUresult> {
        let _guard = self.context.push_current()?;
        let segments = self
            .allocations
            .values()
            .map(|allocation| MemorySegmentMeta {
                ptr: allocation.map.ptr,
                size: allocation.map.size,
                proxy: match allocation.handle {
                    AllocHandle::Owned { .. } => 0,
                    AllocHandle::Borrowed { proxy } => proxy,
                },
            })
            .collect();

        let mut user_handles = Vec::with_capacity(self.user_handles.len());
        for (proxy, (handle, size)) in &self.user_handles {
            let prop = handle.get_prop()?;
            user_handles.push(UserHandleMeta {
                proxy: *proxy,
                size: *size,
                prop: MemAllocProp::from_raw(&prop),
            });
        }

        Ok(MemoryCheckpointMeta {
            granularity: self.granularity,
            next: self.next,
            segments,
            user_handles,
        })
    }

    fn restore(&mut self, meta: &MemoryCheckpointMeta) -> CUresult {
        if !self.allocations.is_empty() || !self.user_handles.is_empty() {
            return CUresult::CUDA_ERROR_INVALID_CONTEXT;
        }
        if meta.granularity != self.granularity {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        if meta.next > self.range.end {
            log::error!(
                "failed to restore: reserved {} but next is {}",
                ByteSize(self.range.end - self.range.ptr),
                ByteSize(meta.next - self.range.ptr),
            );
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }

        // `next` must be restored explicitly because reserved-but-unmapped VA gaps can exist.
        self.next = meta.next;

        let _guard = self.context.push_current()?;
        let mut user_handles = BTreeMap::new();
        for entry in &meta.user_handles {
            let prop = entry.prop.to_raw(self.device());
            let handle = cuda::MemoryHandle::new_with_prop(entry.size, &prop)?;
            let dupe = user_handles.insert(entry.proxy, (handle, entry.size));
            assert!(dupe.is_none());
        }

        let mut allocations = BTreeMap::new();
        for segment in &meta.segments {
            let allocation = if segment.proxy == 0 {
                let handle = cuda::MemoryHandle::new(self.device(), segment.size)?;
                let map = cuda::MemoryMap::new(segment.ptr, segment.size, &handle)?;
                MemoryMapSlot { map, handle: AllocHandle::Owned { _handle: handle } }
            } else {
                let (handle, _) =
                    user_handles.get(&segment.proxy).ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
                let map = cuda::MemoryMap::new(segment.ptr, segment.size, handle)?;
                MemoryMapSlot { map, handle: AllocHandle::Borrowed { proxy: segment.proxy } }
            };
            allocation.map.set_access(self.device())?;
            let dupe = allocations.insert(segment.ptr, allocation);
            assert!(dupe.is_none());
        }

        self.user_handles = user_handles;
        self.allocations = allocations;
        CUresult::CUDA_SUCCESS
    }
}

#[derive(Encode, Decode)]
pub struct MemoryCheckpointMeta {
    pub granularity: usize,
    pub next: CUdeviceptr,
    pub segments: Vec<MemorySegmentMeta>,
    pub user_handles: Vec<UserHandleMeta>,
}

#[derive(Encode, Decode)]
pub struct MemorySegmentMeta {
    pub ptr: CUdeviceptr,
    pub size: usize,
    pub proxy: u64,
}

#[derive(Encode, Decode)]
pub struct UserHandleMeta {
    pub proxy: u64,
    pub size: usize,
    pub prop: MemAllocProp,
}

#[derive(Encode, Decode)]
pub struct MemAllocProp {
    pub requested_handle_types: u32,
    pub compression_type: u8,
    pub gpu_direct_rdma_capable: u8,
}

impl MemAllocProp {
    fn from_raw(raw: &CUmemAllocationProp) -> Self {
        Self {
            requested_handle_types: raw.requestedHandleTypes as _,
            compression_type: raw.allocFlags.compressionType,
            gpu_direct_rdma_capable: raw.allocFlags.gpuDirectRDMACapable,
        }
    }

    fn to_raw(&self, device: c_int) -> CUmemAllocationProp {
        let mut result = CUmemAllocationProp {
            type_: CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: unsafe { mem::transmute(self.requested_handle_types) },
            location: CUmemLocation {
                type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: device,
            },
            ..unsafe { mem::zeroed() }
        };
        result.allocFlags.compressionType = self.compression_type;
        result.allocFlags.gpuDirectRDMACapable = self.gpu_direct_rdma_capable;
        result
    }
}
