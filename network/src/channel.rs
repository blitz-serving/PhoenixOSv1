use std::net::TcpListener;

use crate::config::{CommType, CommonConfig, NetworkConfig};
#[cfg(feature = "rdma")]
use crate::ringbufferchannel::RDMAChannel;
use crate::ringbufferchannel::{EmulatorChannel, SHMChannel};
use crate::tcp::{self, TcpReceiver, TcpSender};
use crate::{
    CommChannelError, CommChannelInnerIO, MemRead, MemWrite, RecvChannel, SendChannel, restore,
};

type Result<T> = std::result::Result<T, CommChannelError>;

pub fn client(config: &CommonConfig, network: &NetworkConfig, id: i32) -> (Sender, Receiver) {
    match config.checked_comm_type() {
        CommType::Shm { emulator } => {
            let (sender, receiver) = SHMChannel::new_client_with_id(config, id).unwrap();
            if emulator {
                (
                    Sender::Emulator(EmulatorChannel::new(sender, config)),
                    Receiver::Emulator(EmulatorChannel::new(receiver, config)),
                )
            } else {
                (Sender::Shm(sender), Receiver::Shm(receiver))
            }
        }
        CommType::Tcp => {
            let (sender, receiver) = tcp::new_client(network, id).unwrap();
            (Sender::Tcp(sender), Receiver::Tcp(receiver))
        }
        #[cfg(feature = "rdma")]
        CommType::Rdma => {
            let (sender, receiver) = RDMAChannel::new_client(config, network, id);
            (Sender::Rdma(sender), Receiver::Rdma(receiver))
        }
    }
}

pub fn attach_server(config: &CommonConfig, id: i32) -> (Receiver, Sender) {
    match config.checked_comm_type() {
        CommType::Shm { emulator } => {
            let (receiver, sender) = SHMChannel::new_client_with_id(config, id).unwrap();
            if emulator {
                (
                    Receiver::Emulator(EmulatorChannel::new(receiver, config)),
                    Sender::Emulator(EmulatorChannel::new(sender, config)),
                )
            } else {
                (Receiver::Shm(receiver), Sender::Shm(sender))
            }
        }
        CommType::Tcp => panic!("attach_server only supports shm"),
        #[cfg(feature = "rdma")]
        CommType::Rdma => panic!("attach_server only supports shm"),
    }
}

pub enum Listener {
    Shm(SHMChannel, SHMChannel),
    Tcp(TcpListener),
    #[cfg(feature = "rdma")]
    Rdma,
}

impl Listener {
    pub fn new(config: &CommonConfig, network: &NetworkConfig, id: i32) -> Self {
        match config.checked_comm_type() {
            CommType::Shm { .. } => {
                let (receiver, sender) = SHMChannel::new_server_with_id(config, id).unwrap();
                Self::Shm(receiver, sender)
            }
            CommType::Tcp => Self::Tcp(tcp::new_listener(network, id).unwrap()),
            #[cfg(feature = "rdma")]
            CommType::Rdma => Self::Rdma,
        }
    }

    #[cfg_attr(not(feature = "rdma"), expect(unused_variables))]
    pub fn accept(
        self,
        config: &CommonConfig,
        network: &NetworkConfig,
        id: i32,
    ) -> (Receiver, Sender) {
        match self {
            Self::Shm(receiver, sender) => {
                if config.emulator {
                    (
                        Receiver::Emulator(EmulatorChannel::new(receiver, config)),
                        Sender::Emulator(EmulatorChannel::new(sender, config)),
                    )
                } else {
                    (Receiver::Shm(receiver), Sender::Shm(sender))
                }
            }
            Self::Tcp(listener) => {
                let (receiver, sender) = tcp::new_server(listener).unwrap();
                (Receiver::Tcp(receiver), Sender::Tcp(sender))
            }
            #[cfg(feature = "rdma")]
            Self::Rdma => {
                let (receiver, sender) = RDMAChannel::new_server(config, network, id);
                (Receiver::Rdma(receiver), Sender::Rdma(sender))
            }
        }
    }
}

macro_rules! ring_buffer_methods {
    () => {
        pub fn register(&self, f: fn(*const u8, usize)) {
            fn inner(
                channel: &impl crate::ringbufferchannel::BufferManager,
                f: fn(*const u8, usize),
            ) {
                f(channel.get_ptr(), channel.get_len())
            }
            match self {
                Self::Shm(channel) => inner(channel, f),
                #[cfg(feature = "rdma")]
                Self::Rdma(channel) => inner(channel, f),
                Self::Emulator(channel) => inner(channel.inner(), f),
                _ => return,
            }
        }

        pub fn as_buffer(&self) -> Option<&mut [u8]> {
            let inner = crate::ringbufferchannel::RingBufferManager::as_buffer;
            match self {
                Self::Shm(channel) => Some(inner(channel)),
                #[cfg(feature = "rdma")]
                Self::Rdma(channel) => Some(inner(channel)),
                Self::Emulator(channel) => Some(inner(channel.inner())),
                _ => None,
            }
        }
    };
}

pub enum Sender {
    Shm(SHMChannel),
    Tcp(TcpSender),
    #[cfg(feature = "rdma")]
    Rdma(RDMAChannel),
    Emulator(EmulatorChannel),
    BlackHole(restore::BlackHole),
}

impl Sender {
    ring_buffer_methods!();

    pub fn put_bytes(&mut self, src: &mut impl MemRead) -> Result<()> {
        match self {
            Self::Shm(channel) => CommChannelInnerIO::put_bytes(channel, src),
            #[cfg(feature = "rdma")]
            Self::Rdma(channel) => CommChannelInnerIO::put_bytes(channel, src),
            Self::Emulator(channel) => channel.put_bytes(src),
            Self::Tcp(_) | Self::BlackHole(_) => Err(CommChannelError::InvalidOperation),
        }
    }

    delegate::delegate! {
        to match self {
            Self::Shm(channel) => channel,
            Self::Tcp(sender) => sender,
            #[cfg(feature = "rdma")]
            Self::Rdma(channel) => channel,
            Self::Emulator(channel) => channel,
            Self::BlackHole(hole) => hole,
        } {
            #[through(SendChannel)]
            pub fn flush(&mut self) -> Result<()>;
            #[through(SendChannel)]
            pub fn send<T: Copy>(&mut self, src: &T) -> Result<()>;
            #[through(SendChannel)]
            pub fn send_unaligned<T: Copy>(&mut self, src: *const T) -> Result<()>;
            #[through(SendChannel)]
            pub fn send_slice<T: Copy>(&mut self, src: &[T]) -> Result<()>;
        }
    }
}

pub enum Receiver {
    Shm(SHMChannel),
    Tcp(TcpReceiver),
    #[cfg(feature = "rdma")]
    Rdma(RDMAChannel),
    Emulator(EmulatorChannel),
    RestoreVec(restore::RestoreVec),
}

impl Receiver {
    ring_buffer_methods!();

    pub fn get_bytes(&mut self, dst: &mut impl MemWrite) -> Result<()> {
        match self {
            Self::Shm(channel) => CommChannelInnerIO::get_bytes(channel, dst),
            #[cfg(feature = "rdma")]
            Self::Rdma(channel) => CommChannelInnerIO::get_bytes(channel, dst),
            Self::Emulator(channel) => channel.get_bytes(dst),
            Self::Tcp(_) | Self::RestoreVec(_) => Err(CommChannelError::InvalidOperation),
        }
    }

    delegate::delegate! {
        to match self {
            Self::Shm(channel) => channel,
            Self::Tcp(receiver) => receiver,
            Self::Emulator(channel) => channel,
            #[cfg(feature = "rdma")]
            Self::Rdma(channel) => channel,
            Self::RestoreVec(vec) => vec,
        } {
            #[through(RecvChannel)]
            pub fn recv<T: Copy>(&mut self) -> Result<T>;
            #[through(RecvChannel)]
            pub fn recv_to<T: Copy>(&mut self, dst: &mut T) -> Result<()>;
            #[through(RecvChannel)]
            pub fn recv_slice<T: Copy>(&mut self) -> Result<Box<[T]>>;
            #[through(RecvChannel)]
            pub fn recv_slice_to<T: Copy>(&mut self, dst: &mut [T]) -> Result<()>;
        }
    }
}
