use std::collections::btree_map::{BTreeMap, Entry};
use std::io::{self, Read as _, Write as _};
use std::net::TcpListener;
use std::sync::{Arc, Barrier};
use std::time::Duration;
use std::{ptr, thread};

use network::NetworkConfig;
use server::*;

fn main() {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    // core_affinity::set_for_current(0);
    let config = NetworkConfig::read_from_file();
    #[cfg(feature = "phos")]
    assert_eq!(config.comm_type, "shm", "PhOS only supports SHM communication");
    match config.comm_type.as_str() {
        "shm" => {
            log::info!("Using shared memory channel")
        }
        "tcp" => log::info!("Using TCP channel"),
        #[cfg(feature = "rdma")]
        "rdma" => log::info!("Using RDMA channel"),
        _ => panic!("Unsupported communication type in config"),
    }

    let (rx, tx) = daemon(&config); // fork return
    server_process(config, rx, tx);
}

fn daemon(config: &NetworkConfig) -> (io::PipeReader, io::PipeWriter) {
    let mut children = BTreeMap::new();

    let listener = TcpListener::bind(&config.daemon_socket).unwrap();
    let listener = socket2::Socket::from(listener);
    listener.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
    let mut id: i32 = 0;
    loop {
        let finished = reap_children();
        if !finished.is_empty() {
            children.retain(|_, (server_pid, _, _)| !finished.contains(server_pid));
        }
        drop(finished);

        let mut stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
            Err(e) => panic!("accept failed: {:?}", e),
        };
        stream.set_read_timeout(None).unwrap();
        let mut buf = [0u8; 4];
        stream.read_exact(&mut buf).unwrap();
        let client_pid = u32::from_be_bytes(buf);
        log::info!("[#{id}] Client PID = {client_pid}");

        let (_, daemon_rx, daemon_tx) = match children.entry(client_pid) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let (child_rx, daemon_tx) = io::pipe().unwrap();
                let (daemon_rx, child_tx) = io::pipe().unwrap();
                let server_pid = unsafe { libc::fork() };
                if server_pid == 0 {
                    return (child_rx, child_tx);
                }
                entry.insert((server_pid, daemon_rx, daemon_tx))
            }
        };
        daemon_tx.write_all(&id.to_ne_bytes()).unwrap();
        daemon_tx.write_all(&client_pid.to_ne_bytes()).unwrap();
        daemon_rx.read_exact(&mut buf).unwrap();
        assert_eq!(id.to_ne_bytes(), buf);
        stream.write_all(&id.to_be_bytes()).unwrap();
        id += 1;
    }
}

fn server_process(config: NetworkConfig, mut rx: io::PipeReader, mut tx: io::PipeWriter) {
    let config = Arc::new(config);
    let mut is_main_thread = true;
    loop {
        let mut buf = [0u8; 4];
        rx.read_exact(&mut buf).unwrap();
        let id = i32::from_ne_bytes(buf);
        rx.read_exact(&mut buf).unwrap();
        let client_pid = u32::from_ne_bytes(buf);
        let child_config = Arc::clone(&config);
        let (barrier, child_barrier) = match config.comm_type.as_str() {
            "shm" | "tcp" => {
                let barrier = Arc::new(Barrier::new(2));
                let child_barrier = Arc::clone(&barrier);
                (Some(barrier), Some(child_barrier))
            }
            #[cfg(feature = "rdma")]
            "rdma" => (None, None),
            _ => panic!("Unsupported communication type in config"),
        };
        thread::spawn(move || {
            launch_server(&child_config, id, client_pid, child_barrier, is_main_thread);
        });
        barrier.map(|barrier| barrier.wait());
        tx.write_all(&id.to_ne_bytes()).unwrap();
        is_main_thread = false;
    }
}

fn reap_children() -> Vec<libc::pid_t> {
    let mut result = Vec::new();
    loop {
        let pid = unsafe { libc::waitpid(-1, ptr::null_mut(), libc::WNOHANG) };
        if pid > 0 {
            result.push(pid);
        } else {
            return result;
        }
    }
}
