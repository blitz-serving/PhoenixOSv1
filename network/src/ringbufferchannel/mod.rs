use std::ptr::{self, NonNull};
use std::slice;

use crate::{CommChannelError, CommChannelInner, CommChannelInnerIO, MemRead, MemWrite};

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
    #[expect(clippy::mut_from_ref)]
    fn as_buffer(&self) -> &mut [u8] {
        assert!(self.is_empty());
        let buffer = unsafe { slice::from_raw_parts_mut(self.get_ptr(), self.get_len()) };
        &mut buffer[META_AREA..]
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

    /// TODO: See [crate::Channel::flag_ptr] for writing operations.
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
    fn contiguous_read_slice_from(&self, cur_head: usize, max_len: usize) -> &[u8] {
        let cur_tail = self.read_tail_volatile();
        let contiguous_len = if cur_tail >= cur_head {
            cur_tail - cur_head
        } else {
            self.capacity() - cur_head
        };
        let len = std::cmp::min(contiguous_len, max_len);

        unsafe {
            let read_ptr = self.get_ptr().add(META_AREA + cur_head);
            slice::from_raw_parts(read_ptr, len)
        }
    }

    #[inline]
    #[expect(clippy::mut_from_ref)]
    fn contiguous_write_slice_from(&self, cur_tail: usize, max_len: usize) -> &mut [u8] {
        let mut cur_head = self.read_head_volatile();
        if cur_head == 0 {
            cur_head = self.capacity();
        }

        let contiguous_len = if cur_tail >= cur_head {
            self.capacity() - cur_tail
        } else {
            cur_head - cur_tail - 1
        };
        let len = std::cmp::min(contiguous_len, max_len);

        unsafe {
            let write_ptr = self.get_ptr().add(META_AREA + cur_tail);
            slice::from_raw_parts_mut(write_ptr, len)
        }
    }
}

impl<T: RingBufferManager + CommChannelInner> CommChannelInnerIO for T {
    fn put_bytes(&self, src: &mut impl MemRead) -> Result<(), CommChannelError> {
        let mut len = src.remaining();

        while len > 0 {
            // current head and tail
            let read_tail = self.read_tail_volatile();
            assert!(read_tail < self.capacity(), "read_tail: {}", read_tail);

            // buf_head can be modified by the other side at any time
            // so we need to read it at the beginning and assume it is not changed
            if self.is_full() {
                self.flush_out()?;
            }

            let write_slice = self.contiguous_write_slice_from(read_tail, len);
            let current = write_slice.len();
            src.read(write_slice);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            self.write_tail_volatile((read_tail + current) % self.capacity());
            len -= current;
        }

        Ok(())
    }

    fn get_bytes(&self, dst: &mut impl MemWrite) -> Result<(), CommChannelError> {
        let total = dst.remaining();
        let mut cur_recv = 0;
        while cur_recv != total {
            let Ok(recv) = self.try_get_bytes(dst) else { panic!() };
            if recv == 0 && self.read_flag_volatile() == SHM_FLAG_WRITE_DISABLED & !SHM_FLAG_IN_USE
            {
                assert_eq!(cur_recv, 0);
                return Err(CommChannelError::ShmChannelLocked);
            }
            cur_recv += recv;
        }
        Ok(())
    }

    fn try_get_bytes(&self, dst: &mut impl MemWrite) -> Result<usize, CommChannelError> {
        let mut len = dst.remaining();
        let mut recv = 0;

        while len > 0 {
            if self.is_empty() {
                return Ok(recv);
            }

            let read_head = self.read_head_volatile();
            assert!(read_head < self.capacity(), "read_head: {}", read_head);
            let read_slice = self.contiguous_read_slice_from(read_head, len);
            let current = read_slice.len();
            dst.write(read_slice);

            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            assert!(
                read_head + current <= self.capacity(),
                "read_head: {}, current: {}, capacity: {}",
                read_head,
                current,
                self.capacity()
            );
            self.write_head_volatile((read_head + current) % self.capacity());
            recv += current;
            len -= current;
        }

        Ok(recv)
    }
}

/// A simple local channel buffer manager
pub struct LocalChannel {
    ptr: *mut u8,
    size: usize,
}

impl Drop for LocalChannel {
    fn drop(&mut self) {
        let buffer: NonNull<u8> = NonNull::new(self.ptr).expect("Pointer must not be null");
        utils::deallocate(buffer, self.size, CACHE_LINE_SZ);
    }
}

impl LocalChannel {
    pub fn new(size: usize) -> LocalChannel {
        assert!(size > META_AREA);
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

impl CommChannelInner for LocalChannel {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        while self.is_full() {
            // Busy-waiting
        }
        Ok(())
    }
}

impl<C: CommChannelInner> crate::SendChannel for C {
    fn flush(&mut self) -> Result<(), CommChannelError> {
        CommChannelInner::flush_out(self)
    }

    fn put_bytes(&mut self, mut src: &[u8]) -> Result<(), CommChannelError> {
        CommChannelInnerIO::put_bytes(self, &mut src)
    }

    // TODO: optimize send()
}

impl<C: CommChannelInner> crate::RecvChannel for C {
    fn get_bytes(&mut self, mut dst: &mut [u8]) -> Result<(), CommChannelError> {
        CommChannelInnerIO::get_bytes(self, &mut dst)
    }

    // TODO: optimize recv()
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
