use std::ffi::{CString, c_int};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::{io, ptr};

use super::{BufferManager, RingBufferManager};
use crate::config::CommonConfig;
use crate::{CommChannelError, CommChannelInner};

/// A shared memory channel buffer manager
pub struct SHMChannel {
    shm: Shm,
    map: Mmap,
}

unsafe impl Send for SHMChannel {}

impl SHMChannel {
    /// Create a new shared memory channel buffer manager for the server
    /// The name server is more consistent with the remoting library
    pub fn new_server(shm_name: &str, shm_len: usize) -> io::Result<Self> {
        Self::new_inner(
            shm_name,
            shm_len,
            libc::O_CREAT | libc::O_TRUNC | libc::O_RDWR,
            libc::S_IRUSR | libc::S_IWUSR,
        )
    }

    pub fn new_server_with_id(config: &CommonConfig, id: i32) -> io::Result<(Self, Self)> {
        Ok((
            Self::new_server(&format!("{}_{}", config.ctos_channel_name, id), config.buf_size)?,
            Self::new_server(&format!("{}_{}", config.stoc_channel_name, id), config.buf_size)?,
        ))
    }

    pub fn new_client(shm_name: &str, shm_len: usize) -> io::Result<Self> {
        Self::new_inner(shm_name, shm_len, libc::O_RDWR, libc::S_IRUSR | libc::S_IWUSR)
    }

    pub fn new_client_with_id(config: &CommonConfig, id: i32) -> io::Result<(Self, Self)> {
        Ok((
            Self::new_client(&format!("{}_{}", config.ctos_channel_name, id), config.buf_size)?,
            Self::new_client(&format!("{}_{}", config.stoc_channel_name, id), config.buf_size)?,
        ))
    }

    fn new_inner(shm_name: &str, len: usize, oflag: i32, mode: libc::mode_t) -> io::Result<Self> {
        let shm_name_c_str = CString::new(shm_name).unwrap();
        let (shm, fd) = Shm::open(shm_name_c_str, len, oflag, mode)?;

        // map the shared memory to the process's address space
        let map = Mmap::new(fd.as_raw_fd(), len)?;

        Ok(Self { shm, map })
    }

    pub fn flag_ptr(&self) -> *mut u64 {
        unsafe { self.map.ptr.add(super::FLAG_OFF).cast() }
    }

    pub fn dont_unlink(&mut self) {
        self.shm.unlink_on_drop = false;
    }
}

impl BufferManager for SHMChannel {
    fn get_ptr(&self) -> *mut u8 {
        self.map.ptr
    }

    fn get_len(&self) -> usize {
        self.map.len
    }
}

impl RingBufferManager for SHMChannel {}

impl CommChannelInner for SHMChannel {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        while self.is_full() {
            // Busy-waiting
        }
        Ok(())
    }
}

struct Shm {
    name: CString,
    unlink_on_drop: bool,
}

impl Shm {
    fn open(
        name: CString,
        len: usize,
        oflag: c_int,
        mode: libc::mode_t,
    ) -> io::Result<(Self, OwnedFd)> {
        let fd = unsafe { libc::shm_open(name.as_ptr(), oflag, mode) };
        if fd == -1 {
            let e = io::Error::last_os_error();
            log::error!("Error on shm_open for new_host: {e}");
            return Err(e);
        }
        let shm = Self { name, unlink_on_drop: true };
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        if unsafe { libc::ftruncate(fd.as_raw_fd(), len as _) } == -1 {
            let e = io::Error::last_os_error();
            log::error!("Error on ftruncate: {e}");
            return Err(e);
        }
        Ok((shm, fd))
    }
}

impl Drop for Shm {
    fn drop(&mut self) {
        if !self.unlink_on_drop {
            return;
        }
        let result = unsafe { libc::shm_unlink(self.name.as_ptr()) };
        if result == -1 {
            let e = io::Error::last_os_error();
            if e.kind() != io::ErrorKind::NotFound {
                log::error!("Error on shm_unlink for {:?}: {e}", self.name);
            }
        }
    }
}

struct Mmap {
    ptr: *mut u8,
    len: usize,
}

impl Mmap {
    fn new(fd: RawFd, len: usize) -> io::Result<Self> {
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr::eq(ptr, libc::MAP_FAILED) {
            let e = io::Error::last_os_error();
            log::error!("Error on mmap the SHM pointer: {e}");
            return Err(e);
        }
        Ok(Self { ptr: ptr.cast(), len })
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        let result = unsafe { libc::munmap(self.ptr.cast(), self.len) };
        debug_assert_eq!(result, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shm_channel_buffer_manager() {
        let shm_name = "/stoc";
        let shm_len = 64;
        let manager = SHMChannel::new_server(shm_name, shm_len).unwrap();
        assert_eq!(manager.get_len(), shm_len);
    }
}
