use std::cell::Cell;
use std::net::SocketAddr;
use std::sync::Arc;

use KRdmaKit::ControlpathError::CreationError;
use KRdmaKit::context::Context;
use KRdmaKit::services_user::{
    ConnectionManagerServer, DefaultConnectionManagerHandler, MRInfo, MRWrapper,
};
use KRdmaKit::{MemoryRegion, QueuePair, QueuePairBuilder, QueuePairStatus, UDriver};

use super::{BufferManager, HEAD_OFF, META_AREA, RingBufferManager, TAIL_OFF};
use crate::config::{CommonConfig, NetworkConfig};
use crate::{CommChannelError, CommChannelInner};

/// Controls how often we send a signaled request.
/// https://github.com/jcxue/RDMA-Tutorial/wiki#selective-signaling
/// https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/#Gotchas_and_Pitfalls
const SIGNAL_INTERVAL: u64 = 16;

/// The default sq size is 128, see [`QueuePairBuilder::set_max_send_wr()`].
/// We can't change it due to hardcoded logic in [`DefaultConnectionManagerHandler`].
const MAX_SEND_WR: u64 = 128;

pub struct RDMAChannel {
    mr: MemoryRegion,
    qp: Arc<QueuePair>,
    rinfo: MRInfo,
    // FIXME: this should be tied to the qp, but currently the receiver does not use it
    next_req_id: Cell<u64>,
    last_poll: Cell<u64>,
    last_tail: Cell<usize>,
}

pub struct RdmaConfig<'a> {
    pub handshake_socket: &'a str,
    pub device_name: &'a str,
    pub device_port: u8,
    pub stoc_channel_name: &'a str,
    pub ctos_channel_name: &'a str,
    pub buf_size: usize,
}

impl<'a> RdmaConfig<'a> {
    pub fn from(common: &'a CommonConfig, network: &'a NetworkConfig) -> Self {
        Self {
            handshake_socket: network.handshake_socket.as_str(),
            device_name: network.device_name.as_str(),
            device_port: network.device_port,
            stoc_channel_name: common.stoc_channel_name.as_str(),
            ctos_channel_name: common.ctos_channel_name.as_str(),
            buf_size: common.buf_size,
        }
    }
}

impl RDMAChannel {
    pub fn new_server(common: &CommonConfig, network: &NetworkConfig, id: i32) -> (Self, Self) {
        Self::new_server_inner(&RdmaConfig::from(common, network), id)
    }

    pub fn new_server_inner(config: &RdmaConfig<'_>, id: i32) -> (Self, Self) {
        let mut addr: SocketAddr = config.handshake_socket.parse().unwrap();
        addr.set_port(addr.port() + id as u16);

        let (ctx, mr, mr2) = Self::allocate_mr(&config.device_name, config.buf_size);
        let mut handler = DefaultConnectionManagerHandler::new(&ctx, config.device_port);
        handler.register_mr(vec![
            (config.ctos_channel_name.to_owned(), mr),
            (config.stoc_channel_name.to_owned(), mr2),
        ]);
        let cm = ConnectionManagerServer::new(handler);
        let listener = cm.spawn_listener(addr);

        // Wait client side connection, then get qp.
        while cm.handler().registered_rc.lock().unwrap().is_empty() {
            std::hint::spin_loop();
        }
        // Wait to get client side mr info.
        while cm.handler().remote_mr.lock().unwrap().inner().is_empty() {
            std::hint::spin_loop();
        }

        cm.stop_listening();
        let _ = listener.join();

        let mut handler = Arc::into_inner(cm).unwrap().into_handler();
        let registered_mr = &mut handler.registered_mr.inner;
        let remote_mr = Arc::into_inner(handler.remote_mr).unwrap().into_inner().unwrap();
        let qp = {
            let registered_rc =
                Arc::into_inner(handler.registered_rc).unwrap().into_inner().unwrap();
            assert_eq!(registered_rc.len(), 1);
            registered_rc.into_values().next().unwrap()
        };

        (
            Self::new(
                registered_mr.remove(config.ctos_channel_name).unwrap(),
                Arc::clone(&qp),
                *remote_mr.inner().get(config.ctos_channel_name).unwrap(),
            ),
            Self::new(
                registered_mr.remove(config.stoc_channel_name).unwrap(),
                qp,
                *remote_mr.inner().get(config.stoc_channel_name).unwrap(),
            ),
        )
    }

    pub fn new_client(common: &CommonConfig, network: &NetworkConfig, id: i32) -> (Self, Self) {
        Self::new_client_inner(&RdmaConfig::from(common, network), id)
    }

    pub fn new_client_inner(config: &RdmaConfig<'_>, id: i32) -> (Self, Self) {
        let mut addr: SocketAddr = config.handshake_socket.parse().unwrap();
        addr.set_port(addr.port() + id as u16);

        let (ctx, mr, mr2) = Self::allocate_mr(&config.device_name, config.buf_size);
        let mut builder = QueuePairBuilder::new(&ctx);
        builder.allow_remote_rw().allow_remote_atomic().set_port_num(config.device_port);
        let qp = loop {
            let qp = builder.clone().build_rc().expect("failed to create the client QP");
            match qp.handshake(addr) {
                Ok(res) => {
                    break res;
                }
                Err(e) => {
                    if let CreationError(msg, _) = &e {
                        if *msg == "Failed to connect server" {
                            std::thread::sleep(std::time::Duration::from_millis(10));
                            continue;
                        }
                    }
                    panic!("Handshake failed!");
                }
            }
        };
        match qp.status().expect("Query status failed!") {
            QueuePairStatus::ReadyToSend => log::info!("[#{id}] QP bring up succeeded"),
            _ => eprintln!("Error : Bring up failed"),
        }

        let mr_infos = qp.query_mr_info().expect("Failed to query MR info");

        // Send client side mr info to server.
        let mrs = vec![
            (config.ctos_channel_name.to_owned(), mr),
            (config.stoc_channel_name.to_owned(), mr2),
        ];
        let mut mr_wrapper: MRWrapper = Default::default();
        mr_wrapper.insert(mrs);
        let mr_info = mr_wrapper.to_mrinfos();
        let _ = qp.send_mr_info(mr_info).unwrap();

        (
            Self::new(
                mr_wrapper.inner.remove(config.ctos_channel_name).unwrap(),
                Arc::clone(&qp),
                *mr_infos.inner().get(config.ctos_channel_name).unwrap(),
            ),
            Self::new(
                mr_wrapper.inner.remove(config.stoc_channel_name).unwrap(),
                qp,
                *mr_infos.inner().get(config.stoc_channel_name).unwrap(),
            ),
        )
    }

    /// A simple loop queue pair poll to poll completion queue synchronously.
    fn poll_batch(&self, wr_id: u64) {
        let mut completions = [Default::default(); (MAX_SEND_WR / SIGNAL_INTERVAL) as usize];
        loop {
            match self.qp.poll_send_cq(&mut completions).unwrap() {
                [] => std::hint::spin_loop(),
                [.., last] => {
                    let poll = last.wr_id;
                    self.last_poll.set(poll);
                    if poll >= wr_id {
                        return;
                    }
                }
            }
        }
    }

    #[inline]
    pub fn get_last_tail(&self) -> usize {
        self.last_tail.get()
    }

    #[inline]
    pub fn set_last_tail(&self, last_tail: usize) {
        self.last_tail.set(last_tail);
    }

    fn allocate_mr(device: &str, buf_len: usize) -> (Arc<Context>, MemoryRegion, MemoryRegion) {
        let ctx = UDriver::create()
            .expect("failed to query device")
            .devices()
            .iter()
            .find(|dev| dev.name() == device)
            .expect("no rdma device available")
            .open_context()
            .expect("failed to create RDMA context");
        let mr = MemoryRegion::new(ctx.clone(), buf_len).expect("Failed to allocate MR");
        let mr2 = MemoryRegion::new(ctx.clone(), buf_len).expect("Failed to allocate MR");
        (ctx, mr, mr2)
    }

    fn new(mr: MemoryRegion, qp: Arc<QueuePair>, rinfo: MRInfo) -> Self {
        Self {
            mr,
            qp,
            rinfo,
            next_req_id: Cell::new(0),
            last_poll: Cell::new(0),
            last_tail: Cell::new(0),
        }
    }
}

impl RDMAChannel {
    fn get_req_id(&self) -> u64 {
        let req_id = self.next_req_id.get();
        self.next_req_id.set(req_id + 1);
        req_id
    }

    fn read_remote(&self, offset: usize, len: usize) -> usize {
        let l: u64 = offset as u64;
        let r: u64 = l + len as u64;
        let wr_id = self.get_req_id();
        Result::unwrap(self.qp.post_send_read(
            &self.mr,
            l..r,
            true,
            self.rinfo.addr + l,
            self.rinfo.rkey,
            wr_id,
        ));
        self.poll_batch(wr_id);
        len
    }

    fn write_remote(&self, offset: usize, len: usize) -> usize {
        let l: u64 = offset as u64;
        let r: u64 = l + len as u64;
        let wr_id = self.get_req_id();
        Result::unwrap(self.qp.post_send_write(
            &self.mr,
            l..r,
            wr_id % SIGNAL_INTERVAL == 0,
            self.rinfo.addr + l,
            self.rinfo.rkey,
            wr_id,
        ));

        if self.last_poll.get() + (MAX_SEND_WR - SIGNAL_INTERVAL) < wr_id {
            self.poll_batch(self.last_poll.get());
        }
        len
    }

    fn write_tail_remote(&self, tail: usize) {
        let len = std::mem::size_of::<usize>();
        let t: u64 = TAIL_OFF as u64;
        let wr_id = self.get_req_id();

        Result::unwrap(self.qp.post_send_cas(
            &self.mr,
            t + len as u64, // dump useless value next to tail
            wr_id % SIGNAL_INTERVAL == 0,
            self.rinfo.addr + t,
            self.rinfo.rkey,
            wr_id,
            self.get_last_tail() as u64,
            tail as u64,
        ));
        if self.last_poll.get() + (MAX_SEND_WR - SIGNAL_INTERVAL) < wr_id {
            self.poll_batch(self.last_poll.get());
        }
    }
}

impl BufferManager for RDMAChannel {
    fn get_ptr(&self) -> *mut u8 {
        self.mr.get_virt_addr() as _
    }

    fn get_len(&self) -> usize {
        self.mr.capacity()
    }
}

impl RingBufferManager for RDMAChannel {}

impl CommChannelInner for RDMAChannel {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        let cur_tail = self.read_tail_volatile();
        let last_tail = self.get_last_tail();
        if last_tail < cur_tail {
            self.write_remote(META_AREA + last_tail, cur_tail - last_tail);
        }
        if cur_tail < last_tail {
            self.write_remote(META_AREA + last_tail, self.capacity() - last_tail);
            self.write_remote(META_AREA, cur_tail);
        }

        self.write_tail_volatile(cur_tail); // TODO: ???
        self.write_tail_remote(cur_tail);
        self.set_last_tail(cur_tail);

        // FIXME: this waits until all previous writes are completed
        while self.is_full() {
            self.read_remote(HEAD_OFF, std::mem::size_of::<usize>());
        }
        Ok(())
    }
}
