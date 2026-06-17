use std::collections::BTreeMap;
use std::mem;
use std::sync::Mutex;

use network::channel::{Receiver, Sender, attach_server};
use network::config::CommonConfig;
use network::restore::{BlackHole, RestoreVec};

use super::{ByteSize, detach, kernel, nvml};
use crate::ServerThread;
use crate::control::StandbyClient;
use crate::worker::{Connection, pin_channel_buffers};

static STANDBY_CACHE: Mutex<Option<StandbyCache>> = Mutex::new(None);

struct StandbyCache {
    connections: BTreeMap<(u32, i32), Connection>,
}

pub fn standby(config: &CommonConfig, clients: &[StandbyClient]) {
    let mut cache = STANDBY_CACHE.lock().unwrap();
    assert!(cache.is_none(), "standby cache already exists");
    let mut connections = BTreeMap::new();
    let mut total_image_bytes = 0_u64;
    for client in clients {
        let (mut receiver, mut sender) = attach_server(config, client.id);
        detach::dont_unlink(&mut sender, &mut receiver);
        let snapshot = detach::read_snapshot(&receiver, &sender);
        assert_eq!(snapshot.client_pid, client.client_pid, "standby snapshot pid mismatch");
        assert_eq!(snapshot.id, client.id, "standby snapshot id mismatch");
        let is_pinned_memory = snapshot.is_pinned_memory;
        total_image_bytes += snapshot.kernels.len() as u64;
        kernel::standby(snapshot.kernels);
        nvml::log_memory(&format!(
            "after standby kernels loaded for client_pid {}",
            client.client_pid
        ));
        if is_pinned_memory {
            pin_channel_buffers(&sender, &receiver);
        }
        let dupe = connections.insert(
            (client.client_pid, client.id),
            Connection { receiver, sender, is_pinned_memory, handles: None },
        );
        assert!(dupe.is_none());
    }
    log::info!("total kernel images size: {}", ByteSize(total_image_bytes));
    log::info!("deduped images size: {}", ByteSize(kernel::get_all_images_size() as u64));
    nvml::log_memory("after all standby kernels loaded");
    *cache = Some(StandbyCache { connections });
}

pub fn has_standby(client_pid: u32, id: i32) -> bool {
    STANDBY_CACHE
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|cache| cache.connections.contains_key(&(client_pid, id)))
}

pub fn take_standby(client_pid: u32, id: i32) -> Option<Connection> {
    STANDBY_CACHE.lock().unwrap().as_mut()?.connections.remove(&(client_pid, id))
}

pub fn try_put_standby(server: &mut ServerThread) {
    let mut cache = STANDBY_CACHE.lock().unwrap();
    let Some(cache) = cache.as_mut() else { return };
    let old = cache.connections.insert(
        (server.client_pid, server.id),
        Connection {
            receiver: mem::replace(
                &mut server.channel_receiver,
                Receiver::RestoreVec(RestoreVec::new(Vec::new())),
            ),
            sender: mem::replace(&mut server.channel_sender, Sender::BlackHole(BlackHole)),
            is_pinned_memory: server.is_pinned_memory,
            handles: Some(mem::take(&mut server.resources)),
        },
    );
    assert!(old.is_none(), "standby already exists for ({}, {})", server.client_pid, server.id);
}

pub fn get_client_pids() -> Option<Vec<u32>> {
    let cache = STANDBY_CACHE.lock().unwrap();
    Some(cache.as_ref()?.connections.keys().map(|(client_pid, _)| *client_pid).collect())
}
