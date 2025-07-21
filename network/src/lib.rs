use serde::Deserialize;
use std::any::Any;
use std::error::Error;
use std::boxed::Box;
use std::fmt;

pub mod ringbufferchannel;
pub mod tcp;
pub mod type_impl;

pub use ringbufferchannel::types::NsTimestamp;

#[derive(Default, Deserialize)]
pub struct NetworkConfig {
    pub comm_type: String,
    pub receiver_socket: String,
    pub device_name: String,
    pub device_port: u8,
    pub daemon_socket: String,
    pub stoc_channel_name: String,
    pub ctos_channel_name: String,
    pub buf_size: usize,
    pub rtt: f64,
    pub bandwidth: f64,
    pub emulator: bool,
    pub opt_async_api: bool,
    pub opt_shadow_desc: bool,
    pub opt_local: bool,
}


impl NetworkConfig {
    pub fn read_from_file() -> Self {
        // Use environment variable to set config file's path.
        let path = match std::env::var("NETWORK_CONFIG") {
            Ok(val) => val,
            Err(_) => concat!(env!("CARGO_MANIFEST_DIR"), "/../config.toml").to_owned(),
        };
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path, e));
        toml::from_str(&content).expect("Failed to parse config.toml")
    }
}

/// A raw memory struct
/// used to wrap raw memory pointer and length,
/// for brevity of `CommChannel` interface
pub struct RawMemory {
    pub ptr: *const u8,
    pub len: usize,
}

impl RawMemory {
    pub fn new<T>(var: &T, len: usize) -> Self {
        RawMemory {
            ptr: var as *const T as *const u8,
            len,
        }
    }

    pub fn from_ptr(ptr: *const u8, len: usize) -> Self {
        RawMemory { ptr, len }
    }

    pub fn add_offset(&self, offset: usize) -> Self {
        RawMemory {
            ptr: unsafe { self.ptr.add(offset) },
            len: self.len - offset,
        }
    }
}

/// A *mutable* raw memory struct
/// used to wrap raw memory pointer and length,
/// for brevity of `CommChannel` interface
pub struct RawMemoryMut {
    pub ptr: *mut u8,
    pub len: usize,
}

impl RawMemoryMut {
    pub fn new<T>(var: &mut T, len: usize) -> Self {
        RawMemoryMut {
            ptr: var as *mut T as *mut u8,
            len,
        }
    }

    pub fn from_ptr(ptr: *mut u8, len: usize) -> Self {
        RawMemoryMut { ptr, len }
    }

    pub fn add_offset(&self, offset: usize) -> Self {
        RawMemoryMut {
            ptr: unsafe { self.ptr.add(offset) },
            len: self.len - offset,
        }
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
}

impl fmt::Display for CommChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: refine
        write!(f, "Channel Error: {:?}", self)
    }
}

impl Error for CommChannelError {}

pub use CommChannelInner as CommChannel;

impl CommChannelInnerIO for Channel {
    /// Write bytes to the channel
    /// It may flush if the channel has no left space
    fn put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        self.get_inner().put_bytes(src)
    }

    /// Non-block version
    /// Immediately return if error such as no left space
    fn try_put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        self.get_inner().try_put_bytes(src)
    }

    /// Read bytes from the channel
    /// Wait if dont receive enough bytes
    fn get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.get_inner().get_bytes(dst)
    }

    /// Non-block version
    /// Immediately return after receive however long bytes (maybe =0 or <len)
    fn try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.get_inner().try_get_bytes(dst)
    }

    /// Non-block versoin
    /// Return immediately if there's not enough bytes in channel
    fn safe_try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.get_inner().safe_try_get_bytes(dst)
    }
}

impl CommChannelInner for Channel {
    /// Flush the all the buffered results to the channel
    fn flush_out(&self) -> Result<(), CommChannelError> {
        self.get_inner().flush_out()
    }

    /// Only for emulator channel
    /// Will recv timestamp and busy waiting until it's time to receive
    fn recv_ts(&self) -> Result<(), CommChannelError> {
        self.get_inner().recv_ts()
    }
}

///
/// A communication channel allows TBD
pub struct Channel {
    inner: Box<dyn CommChannelInner>,
}

impl Channel {
    pub fn new(inner: Box<dyn CommChannelInner>) -> Self {
        Self {
            inner,
        }
    }

    fn get_inner(&self) -> &Box<dyn CommChannelInner> {
        &self.inner
    }

    pub fn flag_ptr(&self) -> Option<*mut u64> {
        let inner: &dyn Any = self.inner.as_ref();
        inner.downcast_ref::<crate::ringbufferchannel::SHMChannel>().map(|shm| {
            let ptr = crate::ringbufferchannel::BufferManager::get_ptr(shm);
            ptr.wrapping_add(crate::ringbufferchannel::FLAG_OFF).cast()
        })
    }

    pub fn flag(&self) -> Option<&std::sync::atomic::AtomicU64> {
        self.flag_ptr().map(|ptr| {
            unsafe { std::sync::atomic::AtomicU64::from_ptr(ptr) }
        })
    }
}

/// communication interface
pub trait CommChannelInner: CommChannelInnerIO + Send + Any {
    fn flush_out(&self) -> Result<(), CommChannelError>;

    fn recv_ts(&self) -> Result<(), CommChannelError> {
        Ok(())
    }
}

pub trait CommChannelInnerIO {
    fn put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError>;

    fn try_put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError>;

    fn get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError>;

    fn try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError>;

    fn safe_try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError>;
}

///
/// The type itself use `CommChannel` to implicitly implement (de-)serialization logic.
///
/// Every type wanted to be transfered should implement this trait.
pub trait Transportable {
    fn send<C: CommChannel>(&self, channel: &C) -> Result<(), CommChannelError>;

    fn recv<C: CommChannel>(&mut self, channel: &C) -> Result<(), CommChannelError>;
}

pub trait TransportableMarker: Copy {}
