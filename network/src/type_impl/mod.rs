use std::mem::MaybeUninit;
use std::{ptr, slice};

use crate::CommChannelError;

pub trait SendChannel {
    fn flush(&mut self) -> Result<(), CommChannelError>;

    fn put_bytes(&mut self, src: &[u8]) -> Result<(), CommChannelError>;

    fn send<T: Copy>(&mut self, src: &T) -> Result<(), CommChannelError> {
        self.send_unaligned(src as *const T)
    }

    fn send_unaligned<T: Copy>(&mut self, src: *const T) -> Result<(), CommChannelError> {
        self.put_bytes(unsafe { slice::from_raw_parts(src.cast(), size_of::<T>()) })
    }

    fn send_slice<T: Copy>(&mut self, src: &[T]) -> Result<(), CommChannelError> {
        self.send(&src.len())?;
        self.put_bytes(unsafe { slice::from_raw_parts(src.as_ptr().cast(), size_of_val(src)) })
    }
}

pub trait RecvChannel {
    fn get_bytes(&mut self, dst: &mut [u8]) -> Result<(), CommChannelError>;

    fn recv<T: Copy>(&mut self) -> Result<T, CommChannelError> {
        let mut data = MaybeUninit::uninit();
        self.get_bytes(unsafe {
            slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, size_of::<T>())
        })?;
        Ok(unsafe { data.assume_init() })
    }

    fn recv_to<T: Copy>(&mut self, dst: &mut T) -> Result<(), CommChannelError> {
        self.get_bytes(unsafe {
            slice::from_raw_parts_mut(dst as *mut T as *mut u8, size_of::<T>())
        })
    }

    fn recv_slice<T: Copy>(&mut self) -> Result<Box<[T]>, CommChannelError> {
        let len: usize = self.recv()?;
        let mut data = Box::new_uninit_slice(len);
        self.get_bytes(unsafe {
            slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, len * size_of::<T>())
        })?;
        Ok(unsafe { data.assume_init() })
    }

    fn recv_slice_to<T: Copy>(&mut self, dst: &mut [T]) -> Result<(), CommChannelError> {
        let len: usize = self.recv()?;
        assert_eq!(len, dst.len()); // TODO: relax to <=?
        self.get_bytes(unsafe {
            slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, len * size_of::<T>())
        })
    }
}

pub fn save<T: Copy>(data: &T, output: &mut Vec<u8>) {
    output.extend_from_slice(unsafe {
        slice::from_raw_parts(ptr::from_ref(data).cast(), size_of::<T>())
    });
}

pub fn save_slice<T: Copy>(data: &[T], output: &mut Vec<u8>) {
    let len = data.len();
    save(&len, output);
    output.extend_from_slice(unsafe {
        slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ringbufferchannel::{LocalChannel, META_AREA};

    /// Test bool Transportable impl
    #[test]
    fn test_bool_io() {
        let mut channel = LocalChannel::new(10 + META_AREA);
        let a = true;
        let mut b = false;
        channel.send(&a).unwrap();
        channel.recv_to(&mut b).unwrap();
        assert_eq!(a, b);
    }

    /// Test i32 Transportable impl
    #[test]
    fn test_i32_io() {
        let mut channel = LocalChannel::new(10 + META_AREA);
        let a = 123;
        let mut b = 0;
        channel.send(&a).unwrap();
        channel.recv_to(&mut b).unwrap();
        assert_eq!(a, b);
    }

    /// Test [u8] Transportable impl
    #[test]
    fn test_u8_array_io() {
        let mut channel = LocalChannel::new(10 + META_AREA);
        let a = [1u8, 2, 3, 4, 5];
        let mut b = [0u8; 5];
        channel.send_slice(&a).unwrap();
        channel.recv_slice_to(&mut b).unwrap();
        assert_eq!(a, b);
    }

    /// Test [i32] Transportable impl
    #[test]
    fn test_i32_array_io() {
        let mut channel = LocalChannel::new(50 + META_AREA);
        let a = [1i32, 2, 3, 4, 5];
        let mut b = [0i32; 5];
        channel.send_slice(&a).unwrap();
        channel.recv_slice_to(&mut b).unwrap();
        assert_eq!(a, b);
    }
}
