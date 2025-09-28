use std::{ptr, slice};

use crate::{CommChannel, CommChannelError, RawMemory, RawMemoryMut, Transportable};

impl<T: Copy> Transportable for T {
    fn send<C: CommChannel>(&self, channel: &C) -> Result<(), CommChannelError> {
        if size_of::<Self>() == 0 {
            return Ok(());
        }
        let memory = RawMemory::new(self, size_of::<Self>());
        match channel.put_bytes(&memory)? == size_of::<Self>() {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }

    fn recv<C: CommChannel>(&mut self, channel: &C) -> Result<(), CommChannelError> {
        if size_of::<Self>() == 0 {
            return Ok(());
        }
        let mut memory = RawMemoryMut::new(self, size_of::<Self>());
        match channel.get_bytes(&mut memory)? == size_of::<Self>() {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }
}

pub fn save<T: Copy>(data: &T, output: &mut Vec<u8>) {
    output.extend_from_slice(unsafe {
        slice::from_raw_parts(ptr::from_ref(data).cast(), size_of::<T>())
    });
}

pub fn send_slice<T: Copy, C: CommChannel>(
    data: &[T],
    channel: &C,
) -> Result<(), CommChannelError> {
    let len = data.len();
    len.send(channel)?;
    let bytes = size_of_val(data);
    let memory = RawMemory::from_ptr(data.as_ptr() as *const u8, bytes);
    match channel.put_bytes(&memory)? == bytes {
        true => Ok(()),
        false => Err(CommChannelError::IoError),
    }
}

pub fn recv_slice<T: Copy, C: CommChannel>(channel: &C) -> Result<Box<[T]>, CommChannelError> {
    let mut len = 0;
    len.recv(channel)?;
    let mut data = Box::<[T]>::new_uninit_slice(len);
    let bytes = len * size_of::<T>();
    let mut memory = RawMemoryMut::from_ptr(data.as_mut_ptr() as *mut u8, bytes);
    match channel.get_bytes(&mut memory)? == bytes {
        true => Ok(unsafe { data.assume_init() }),
        false => Err(CommChannelError::IoError),
    }
}

pub fn recv_slice_to<T: Copy, C: CommChannel>(
    data: &mut [T],
    channel: &C,
) -> Result<(), CommChannelError> {
    let mut len = 0;
    len.recv(channel)?;
    assert_eq!(len, data.len()); // TODO: relax to <=?
    let bytes = len * size_of::<T>();
    let mut memory = RawMemoryMut::from_ptr(data.as_mut_ptr() as *mut u8, bytes);
    match channel.get_bytes(&mut memory)? == bytes {
        true => Ok(()),
        false => Err(CommChannelError::IoError),
    }
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
    use crate::Channel;
    use crate::ringbufferchannel::{LocalChannel, META_AREA};

    /// Test bool Transportable impl
    #[test]
    fn test_bool_io() {
        let mut channel = Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = true;
        let mut b = false;
        a.send(&mut channel).unwrap();
        b.recv(&mut channel).unwrap();
        assert_eq!(a, b);
    }

    /// Test i32 Transportable impl
    #[test]
    fn test_i32_io() {
        let mut channel = Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = 123;
        let mut b = 0;
        a.send(&mut channel).unwrap();
        b.recv(&mut channel).unwrap();
        assert_eq!(a, b);
    }

    /// Test [u8] Transportable impl
    #[test]
    fn test_u8_array_io() {
        let mut channel = Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = [1u8, 2, 3, 4, 5];
        let mut b = [0u8; 5];
        send_slice(&a, &mut channel).unwrap();
        recv_slice_to(&mut b, &mut channel).unwrap();
        assert_eq!(a, b);
    }

    /// Test [i32] Transportable impl
    #[test]
    fn test_i32_array_io() {
        let mut channel = Channel::new(Box::new(LocalChannel::new(50 + META_AREA)));
        let a = [1i32, 2, 3, 4, 5];
        let mut b = [0i32; 5];
        send_slice(&a, &mut channel).unwrap();
        recv_slice_to(&mut b, &mut channel).unwrap();
        assert_eq!(a, b);
    }
}
