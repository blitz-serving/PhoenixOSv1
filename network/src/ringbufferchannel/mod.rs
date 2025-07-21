// pub mod channel;
use crate::{
    CommChannelInner, CommChannelError, CommChannelInnerIO,
    RawMemory, RawMemoryMut
};
use std::ptr::{self, NonNull};

pub mod test;

// Only implemented in Linux now
#[cfg(target_os = "linux")]
pub mod shm;
#[cfg(target_os = "linux")]
pub use shm::SHMChannel;

#[cfg(feature = "rdma")]
pub mod rdma;
#[cfg(feature = "rdma")]
pub use rdma::RDMAChannel;

pub mod utils;

pub mod emulator;
pub use emulator::EmulatorChannel;
pub mod types;
pub use types::*;

pub const CACHE_LINE_SZ: usize = 64;

pub const HEAD_OFF: usize = 0;
pub const TAIL_OFF: usize = CACHE_LINE_SZ;
pub const FLAG_OFF: usize = CACHE_LINE_SZ * 2;
pub const META_AREA: usize = CACHE_LINE_SZ * 3;

pub const SHM_FLAG_IDLE: u64 = 0;
pub const SHM_FLAG_IN_USE: u64 = 1;
pub const SHM_FLAG_WRITE_DISABLED: u64 = 2;

/// A buffer can use arbitrary memory for its channel
///
/// It will manage the following:
/// - The buffer memory allocation and
/// - The buffer memory deallocation
pub trait BufferManager {
    fn get_ptr(&self) -> *mut u8;

    fn get_len(&self) -> usize;
}

/// A ring buffer can handle head and tail
pub trait RingBufferManager: BufferManager {
    #[inline]
    fn capacity(&self) -> usize {
        self.get_len() - META_AREA
    }

    #[inline]
    fn read_head_volatile(&self) -> usize {
        unsafe { ptr::read_volatile(self.get_ptr().add(HEAD_OFF) as *const usize) }
    }

    #[inline]
    fn write_head_volatile(&self, head: usize) {
        unsafe { ptr::write_volatile(self.get_ptr().add(HEAD_OFF) as *mut usize, head) }
    }

    #[inline]
    fn read_tail_volatile(&self) -> usize {
        unsafe { ptr::read_volatile(self.get_ptr().add(TAIL_OFF) as *const usize) }
    }

    #[inline]
    fn write_tail_volatile(&self, tail: usize) {
        unsafe { ptr::write_volatile(self.get_ptr().add(TAIL_OFF) as *mut usize, tail) }
    }

    /// See [crate::Channel::flag_ptr] for writing operations.
    #[inline]
    fn read_flag_volatile(&self) -> u64 {
        unsafe { ptr::read_volatile(self.get_ptr().add(FLAG_OFF).cast()) }
    }

    #[inline]
    fn num_bytes_stored(&self) -> usize {
        let head = self.read_head_volatile();
        let tail = self.read_tail_volatile();

        if tail >= head {
            // Tail is ahead of head
            tail - head
        } else {
            // Head is ahead of tail, buffer is wrapped
            self.capacity() - (head - tail)
        }
    }

    #[inline]
    fn num_bytes_free(&self) -> usize {
        self.capacity() - self.num_bytes_stored()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.read_head_volatile() == self.read_tail_volatile()
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.num_bytes_stored() >= self.capacity() - 1
    }

    #[inline]
    fn num_adjacent_bytes_to_read(&self, cur_head: usize) -> usize {
        let cur_tail = self.read_tail_volatile();
        if cur_tail >= cur_head {
            cur_tail - cur_head
        } else {
            self.capacity() - cur_head
        }
    }

    #[inline]
    fn num_adjacent_bytes_to_write(&self, cur_tail: usize) -> usize {
        let mut cur_head = self.read_head_volatile();
        if cur_head == 0 {
            cur_head = self.capacity();
        }

        if cur_tail >= cur_head {
            self.capacity() - cur_tail
        } else {
            cur_head - cur_tail - 1
        }
    }
}

pub trait RingBufferChannel: RingBufferManager {
    unsafe fn read_at(&self, offset: usize, dst: *mut u8, len: usize) -> usize {
        unsafe {
            std::ptr::copy_nonoverlapping(self.get_ptr().add(offset), dst, len);
        }
        len
    }

    unsafe fn write_at(&self, offset: usize, src: *const u8, len: usize) -> usize {
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.get_ptr().add(offset), len);
        }
        len
    }
}

impl<T: RingBufferChannel + CommChannelInner> CommChannelInnerIO for T {
    fn put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        let mut len = src.len;
        let mut offset = 0;

        while len > 0 {
            // current head and tail
            let read_tail = self.read_tail_volatile();
            assert!(read_tail < self.capacity(), "read_tail: {}", read_tail);

            // buf_head can be modified by the other side at any time
            // so we need to read it at the beginning and assume it is not changed
            if self.is_full() {
                self.flush_out()?;
            }

            let current = std::cmp::min(self.num_adjacent_bytes_to_write(read_tail), len);

            unsafe {
                self.write_at(META_AREA + read_tail, src.ptr.add(offset), current);
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            self.write_tail_volatile((read_tail + current) % self.capacity());
            offset += current;
            len -= current;
        }

        Ok(offset)
    }

    fn get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        let mut cur_recv = 0;
        while cur_recv != dst.len {
            let mut new_dst = dst.add_offset(cur_recv);
            let Ok(recv) = self.try_get_bytes(&mut new_dst) else { panic!() };
            if recv == 0 && self.read_flag_volatile() == SHM_FLAG_WRITE_DISABLED & !SHM_FLAG_IN_USE {
                assert_eq!(cur_recv, 0);
                return Err(CommChannelError::ShmChannelLocked);
            }
            cur_recv += recv;
        }
        Ok(cur_recv)
    }

    fn try_put_bytes(&self, _src: &RawMemory) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        let mut len = dst.len;
        let mut offset = 0;

        while len > 0 {
            if self.is_empty() {
                return Ok(offset);
            }

            let read_head = self.read_head_volatile();
            assert!(read_head < self.capacity(), "read_head: {}", read_head);
            let current = std::cmp::min(self.num_adjacent_bytes_to_read(read_head), len);

            unsafe {
                self.read_at(META_AREA + read_head, dst.ptr.add(offset), current);
            }

            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            assert!(
                read_head + current <= self.capacity(),
                "read_head: {}, current: {}, capacity: {}",
                read_head,
                current,
                self.capacity()
            );
            self.write_head_volatile((read_head + current) % self.capacity());
            offset += current;
            len -= current;
        }

        Ok(offset)
    }

    fn safe_try_get_bytes(&self, dst: &mut crate::RawMemoryMut) -> Result<usize, CommChannelError> {
        if self.num_bytes_stored() < dst.len {
            Ok(0)
        } else {
            self.try_get_bytes(dst)
        }
    }
}

/// A simple local channel buffer manager
pub struct LocalChannel {
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for LocalChannel {}
unsafe impl Sync for LocalChannel {}

impl Drop for LocalChannel {
    fn drop(&mut self) {
        let buffer: NonNull<u8> = NonNull::new(self.ptr).expect("Pointer must not be null");
        utils::deallocate(buffer, self.size, CACHE_LINE_SZ);
    }
}

impl LocalChannel {
    pub fn new(size: usize) -> LocalChannel {
        let channel = LocalChannel {
            ptr: utils::allocate_cache_line_aligned(size, CACHE_LINE_SZ).as_ptr(),
            size,
        };
        channel.write_head_volatile(0);
        channel.write_tail_volatile(0);
        channel
    }
}

impl BufferManager for LocalChannel {
    fn get_ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn get_len(&self) -> usize {
        self.size
    }
}

impl RingBufferManager for LocalChannel {}

impl RingBufferChannel for LocalChannel {}

impl CommChannelInner for LocalChannel {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        while self.is_full() {
            // Busy-waiting
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_channel_buffer_manager() {
        let size = 64;
        let manager = LocalChannel::new(size);
        let ptr = manager.get_ptr();
        let len = manager.get_len();
        assert!(utils::is_cache_line_aligned(ptr));

        assert_eq!(len, size);
    }
}
