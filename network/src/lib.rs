use std::error::Error;
use std::fmt;

pub mod channel;
pub mod config;
pub mod oob;
pub mod restore;
pub mod ringbufferchannel;
pub mod session;
pub mod tcp;
mod type_impl;

pub use ringbufferchannel::types::NsTimestamp;
pub use type_impl::{RecvChannel, SendChannel};

pub trait MemRead {
    fn read(&mut self, dst: &mut [u8]);
    fn remaining(&self) -> usize;
}

impl MemRead for &[u8] {
    // See `impl Read for &[u8]`.
    #[inline]
    fn read(&mut self, dst: &mut [u8]) {
        let amount = dst.len();
        debug_assert!(amount <= self.len());
        dst.copy_from_slice(&self[..amount]);
        *self = &self[amount..];
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }
}

pub trait MemWrite {
    fn write(&mut self, src: &[u8]);
    fn remaining(&self) -> usize;
}

impl MemWrite for &mut [u8] {
    // See `impl Write for &mut [u8]`.
    #[inline]
    fn write(&mut self, src: &[u8]) {
        let amount = src.len();
        debug_assert!(amount <= self.len());
        let (dst, rest) = std::mem::take(self).split_at_mut(amount);
        dst.copy_from_slice(src);
        *self = rest;
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }
}

#[derive(Debug)]
pub enum CommChannelError {
    // Define error types, for example:
    InvalidOperation,
    IoError,
    Timeout,
    NoLeftSpace,
    BlockOperation,
    // Add other relevant errors
    ShmChannelLocked,
    RestoreEof,
}

impl fmt::Display for CommChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: refine
        write!(f, "Channel Error: {:?}", self)
    }
}

impl Error for CommChannelError {}

/// communication interface
pub(crate) trait CommChannelInner: CommChannelInnerIO {
    fn flush_out(&self) -> Result<(), CommChannelError>;
}

pub(crate) trait CommChannelInnerIO {
    fn put_bytes(&self, src: &mut impl MemRead) -> Result<(), CommChannelError>;

    fn get_bytes(&self, dst: &mut impl MemWrite) -> Result<(), CommChannelError>;

    fn try_get_bytes(&self, dst: &mut impl MemWrite) -> Result<usize, CommChannelError>;
}
