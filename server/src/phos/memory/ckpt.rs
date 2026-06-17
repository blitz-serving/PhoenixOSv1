use std::ffi::c_void;
use std::fs::{self, File};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::{io, ptr, thread};

use bincode::{config, decode_from_slice, encode_to_vec};
use cudasys::cudart::{
    cudaLaunchHostFunc, cudaMemcpy, cudaMemcpyAsync, cudaMemcpyKind, cudaStreamSynchronize,
};

use super::super::ByteSize;
use super::{MEMORY_MANAGER, MemoryCheckpointMeta};

pub fn dump(mut path: PathBuf) {
    fs::create_dir_all(&path).unwrap();
    path.push("_");
    let mut manager = MEMORY_MANAGER.lock().unwrap();
    let meta = manager.checkpoint_meta().unwrap();
    path.set_file_name("meta.bin");
    fs::write(&path, encode_to_vec(&meta, config::standard()).unwrap()).unwrap();
    for segment in &meta.segments {
        path.set_file_name(format!("{:x}.bin", segment.ptr));
        let mut buf = Box::new_uninit_slice(segment.size);
        let result = unsafe {
            cudaMemcpy(
                buf.as_mut_ptr().cast(),
                segment.ptr as _,
                segment.size,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
        };
        assert_eq!(result, Default::default());
        fs::write(&path, unsafe { buf.assume_init() }).unwrap();
    }
}

pub fn dump_to_slice(raw: &mut [u8]) -> (Vec<u8>, usize) {
    let mut manager = MEMORY_MANAGER.lock().unwrap();
    let meta = manager.checkpoint_meta().unwrap();
    let total: usize = meta.segments.iter().map(|segment| segment.size).sum();
    assert!(raw.len() >= total, "sender shm too small: need {total}, got {}", raw.len());
    let meta_bytes = encode_to_vec(&meta, config::standard()).unwrap();
    let mut offset = 0;
    for segment in &meta.segments {
        let result = unsafe {
            cudaMemcpyAsync(
                raw[offset..].as_mut_ptr().cast(),
                segment.ptr as _,
                segment.size,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                ptr::null_mut(),
            )
        };
        assert_eq!(result, Default::default());
        offset += segment.size;
    }
    let result = unsafe { cudaStreamSynchronize(ptr::null_mut()) };
    assert_eq!(result, Default::default());
    (meta_bytes, total)
}

pub fn restore(mut path: PathBuf) {
    path.push("_");
    let mut manager = MEMORY_MANAGER.lock().unwrap();
    let meta: MemoryCheckpointMeta = {
        path.set_file_name("meta.bin");
        decode_from_slice(&fs::read(&path).unwrap(), config::standard()).unwrap().0
    };
    assert_eq!(manager.restore(&meta), Default::default());
    let mmaps: Vec<_> = meta
        .segments
        .into_iter()
        .map(|segment| {
            path.set_file_name(format!("{:x}.bin", segment.ptr));
            let mmap = Mmap::new(&path, segment.size).unwrap();
            let result = unsafe {
                cudaMemcpyAsync(
                    segment.ptr as _,
                    mmap.ptr as _,
                    segment.size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    ptr::null_mut(),
                )
            };
            assert_eq!(result, Default::default());
            mmap
        })
        .collect();
    extern "C" fn callback(ptr: *mut c_void) {
        log::info!("memory restored");
        let mmaps: Box<Vec<Mmap>> = unsafe { Box::from_raw(ptr.cast()) };
        thread::spawn(|| drop(mmaps));
    }
    let result = unsafe {
        cudaLaunchHostFunc(ptr::null_mut(), Some(callback), Box::into_raw(Box::new(mmaps)).cast())
    };
    assert_eq!(result, Default::default());
}

pub fn restore_from_slice(meta_bytes: &[u8], raw: &[u8]) {
    let mut manager = MEMORY_MANAGER.lock().unwrap();
    let meta: MemoryCheckpointMeta = decode_from_slice(meta_bytes, config::standard()).unwrap().0;
    let total: usize = meta.segments.iter().map(|segment| segment.size).sum();
    assert!(raw.len() >= total, "sender shm snapshot truncated: need {total}, got {}", raw.len());
    log::info!("restoring memory size: {}", ByteSize(total as u64));
    assert_eq!(manager.restore(&meta), Default::default());
    let mut offset = 0;
    for segment in meta.segments {
        let result = unsafe {
            cudaMemcpyAsync(
                segment.ptr as _,
                raw[offset..].as_ptr().cast(),
                segment.size,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                ptr::null_mut(),
            )
        };
        assert_eq!(result, Default::default());
        offset += segment.size;
    }
}

pub fn finish_restore_from_slice() {
    let result = unsafe { cudaStreamSynchronize(ptr::null_mut()) };
    assert_eq!(result, Default::default());
}

struct Mmap {
    ptr: *mut c_void,
    len: usize,
}

unsafe impl Send for Mmap {}

impl Mmap {
    fn new(path: &Path, len: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        assert_eq!(file.metadata()?.len(), len as _);
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr::eq(ptr, libc::MAP_FAILED) {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { ptr, len })
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        let result = unsafe { libc::munmap(self.ptr, self.len) };
        debug_assert_eq!(result, 0);
    }
}
