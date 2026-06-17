use std::{mem, ptr, slice};

use network::channel::{Receiver, Sender};
use network::restore::{BlackHole, RestoreVec};

use super::{kernel, memory};
use crate::ServerThread;

const SNAPSHOT_MAGIC: u64 = 0x5048_4f53_4445_5441;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SnapshotHeader {
    magic: u64,
    client_pid: u32,
    id: i32,
    is_pinned_memory: bool,
    kernel_len: u64,
    handles_len: u64,
    memory_meta_len: u64,
    memory_raw_len: u64,
}

pub struct Snapshot<'a> {
    pub client_pid: u32,
    pub id: i32,
    pub is_pinned_memory: bool,
    pub kernels: &'a [u8],
    pub handles: &'a [u8],
    pub memory_meta: &'a [u8],
    pub memory_raw: &'a [u8],
}

impl SnapshotHeader {
    fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(ptr::from_ref(self).cast(), size_of::<Self>()) }
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= size_of::<Self>());
        unsafe { bytes.as_ptr().cast::<Self>().read_unaligned() }
    }
}

fn dump_handles_to_vec(server: &ServerThread) -> Vec<u8> {
    let mut bytes = Vec::new();
    server.resources.serialize(&mut bytes).unwrap();
    bytes
}

pub fn dont_unlink(sender: &mut Sender, receiver: &mut Receiver) {
    let Sender::Shm(channel) = sender else { panic!("SHM channel expected") };
    channel.dont_unlink();
    let Receiver::Shm(channel) = receiver else { panic!("SHM channel expected") };
    channel.dont_unlink();
}

pub fn dump_state(server: &mut ServerThread) {
    dont_unlink(&mut server.channel_sender, &mut server.channel_receiver);

    let kernels = kernel::dump();
    let handles = dump_handles_to_vec(server);
    let receiver = server.channel_receiver.as_buffer().unwrap();
    let sender = server.channel_sender.as_buffer().unwrap();
    let (memory_meta, memory_raw_len) = memory::dump_to_slice(sender);
    let header = SnapshotHeader {
        magic: SNAPSHOT_MAGIC,
        client_pid: server.client_pid,
        id: server.id,
        is_pinned_memory: server.is_pinned_memory,
        kernel_len: kernels.len().try_into().unwrap(),
        handles_len: handles.len().try_into().unwrap(),
        memory_meta_len: memory_meta.len().try_into().unwrap(),
        memory_raw_len: memory_raw_len.try_into().unwrap(),
    };
    let total = size_of::<SnapshotHeader>() + kernels.len() + handles.len() + memory_meta.len();
    assert!(
        receiver.len() >= total,
        "receiver shm too small: need {total}, got {}",
        receiver.len()
    );
    let (header_dst, rest) = receiver.split_at_mut(size_of::<SnapshotHeader>());
    header_dst.copy_from_slice(header.as_bytes());
    let (kernel_dst, rest) = rest.split_at_mut(kernels.len());
    kernel_dst.copy_from_slice(&kernels);
    let (handles_dst, rest) = rest.split_at_mut(handles.len());
    handles_dst.copy_from_slice(&handles);
    let (memory_meta_dst, _) = rest.split_at_mut(memory_meta.len());
    memory_meta_dst.copy_from_slice(&memory_meta);
}

const SYNC_RESTORE_MEM: bool = true;

pub fn begin_attach(server: &mut ServerThread, has_standby: bool) -> (Sender, Receiver) {
    let header = read_snapshot_header(&server.channel_receiver);
    assert_eq!(header.client_pid, server.client_pid, "detach snapshot pid mismatch");
    assert_eq!(header.id, server.id, "detach snapshot id mismatch");
    if header.is_pinned_memory {
        super::initialize_context();
        server.pin_memory(); // if has_standby, this will be a no-op
    }
    let snapshot = read_snapshot(&server.channel_receiver, &server.channel_sender);
    if has_standby {
        kernel::attach_warm(snapshot.kernels);
    } else {
        kernel::restore(snapshot.kernels);
    }
    memory::restore_from_slice(snapshot.memory_meta, snapshot.memory_raw);
    if SYNC_RESTORE_MEM {
        memory::finish_restore_from_slice();
        super::clear_flag(&server.channel_receiver);
        super::nvml::log_memory("memory restored");
    }
    let handles = if server.resources.len() == 0 {
        snapshot.handles.to_vec()
    } else {
        Vec::new()
    };
    let handles = Receiver::RestoreVec(RestoreVec::new(handles));
    let receiver = mem::replace(&mut server.channel_receiver, handles);
    let sender = mem::replace(&mut server.channel_sender, Sender::BlackHole(BlackHole));
    (sender, receiver)
}

pub fn on_restore_finished(receiver: &Receiver) {
    if super::is_locked(receiver) {
        memory::finish_restore_from_slice();
        super::clear_flag(receiver);
    }
    super::nvml::log_memory("attach finished");
}

fn read_snapshot_header(receiver: &Receiver) -> SnapshotHeader {
    let receiver = receiver.as_buffer().unwrap();
    let header = SnapshotHeader::from_bytes(receiver);
    assert_eq!(header.magic, SNAPSHOT_MAGIC, "invalid detach snapshot magic");
    header
}

pub fn read_snapshot<'a>(receiver: &'a Receiver, sender: &'a Sender) -> Snapshot<'a> {
    let header = read_snapshot_header(receiver);
    let receiver = receiver.as_buffer().unwrap();
    let sender = sender.as_buffer().unwrap();

    let kernel_len = usize::try_from(header.kernel_len).unwrap();
    let handles_len = usize::try_from(header.handles_len).unwrap();
    let memory_meta_len = usize::try_from(header.memory_meta_len).unwrap();
    let memory_raw_len = usize::try_from(header.memory_raw_len).unwrap();
    let total = size_of::<SnapshotHeader>() + kernel_len + handles_len + memory_meta_len;
    assert!(
        receiver.len() >= total,
        "receiver shm snapshot truncated: need {total}, got {}",
        receiver.len()
    );
    assert!(
        sender.len() >= memory_raw_len,
        "sender shm snapshot truncated: need {memory_raw_len}, got {}",
        sender.len()
    );

    let mut offset = size_of::<SnapshotHeader>();
    let kernels = &receiver[offset..offset + kernel_len];
    offset += kernel_len;
    let handles = &receiver[offset..offset + handles_len];
    offset += handles_len;
    let memory_meta = &receiver[offset..offset + memory_meta_len];
    let memory_raw = &sender[..memory_raw_len];

    Snapshot {
        client_pid: header.client_pid,
        id: header.id,
        is_pinned_memory: header.is_pinned_memory,
        kernels,
        handles,
        memory_meta,
        memory_raw,
    }
}
