use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::future::ready;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs, io, process, ptr};

use futures::StreamExt as _;
use network::config::Config;
use network::oob::{CheckpointRequest, NewClientThreadRequest, Oob};
use tarpc::context::Context;
use tarpc::serde_transport::{tcp, unix};
use tarpc::server::{BaseChannel, Channel as _};
use tarpc::tokio_serde::formats::Bincode;
use tokio::select;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::Mutex;

use crate::control::{ControlClient, StandbyClient, StandbyRequest};
use crate::worker::SERVER_WORKER_SOCKET_ENV;

pub async fn daemon(config: Config) {
    let addr = &config.server.network.daemon_socket;
    let mut listener = tcp::listen(&addr, Bincode::default).await.unwrap();
    listener.config_mut().max_frame_length(usize::MAX);
    log::info!("daemon listening on {addr}");

    let daemon = Daemon { config, inner: Default::default() };
    // Moved out of select! to make code formattable
    let mut sessions = listener
        .filter_map(|transport| {
            ready(match transport {
                Ok(t) => Some(t),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => None,
                Err(e) => panic!("daemon accept failed: {e}"),
            })
        })
        .map(|transport| {
            #[cfg(feature = "phos")]
            let daemon = crate::phos::PhosDaemon(&daemon);
            BaseChannel::with_defaults(transport).execute(daemon.serve()).for_each(|f| f)
        });

    let mut sigchld = signal(SignalKind::child()).unwrap();

    loop {
        select! {
            signal = sigchld.recv() => {
                signal.unwrap();
                daemon.reap_children().await;
            }
            session = sessions.next() => {
                session.unwrap().await;
            }
        }
    }
}

pub struct Daemon {
    pub config: Config,
    pub inner: Mutex<DaemonInner>,
}

impl Daemon {
    async fn reap_children(&self) {
        self.inner.lock().await.reap_children();
    }
}

fn make_socket_path() -> String {
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    format!("/tmp/phos-worker-{}-{ts}.sock", process::id())
}

impl Oob for &Daemon {
    async fn new_client_thread(self, context: Context, request: NewClientThreadRequest) -> i32 {
        let NewClientThreadRequest { pid, job_name, cuda_visible_devices, config, is_phos } =
            request;
        assert_eq!(config, self.config.common);
        assert_eq!(is_phos, cfg!(feature = "phos"));
        let mut inner = self.inner.lock().await;
        let id = inner.next_id();
        log::info!(
            "[#{id}] client PID: {pid}, job name: {job_name}, CUDA_VISIBLE_DEVICES: {cuda_visible_devices}"
        );
        let child = inner
            .get_or_spawn_child(pid, job_name, cuda_visible_devices, &self.config)
            .await;
        child
            .control
            .as_ref()
            .map(|(_, control)| control)
            .unwrap_or_else(|| panic!("worker for pid {pid} is detached"))
            .create_thread(context, id, pid)
            .await
            .unwrap();
        child.ids.push(id);
        id
    }

    async fn checkpoint_job(self, _: Context, request: CheckpointRequest) {
        unimplemented!("checkpoint_job({request:?})")
    }

    async fn restore_job(self, _: Context, ckpt_dir: String, cuda_visible_devices: String) {
        unimplemented!(
            "restore_job(ckpt_dir={ckpt_dir}, cuda_visible_devices={cuda_visible_devices})"
        )
    }

    async fn attach(self, _: Context, client_pid: u32) {
        unimplemented!("attach(client_pid={client_pid})")
    }

    async fn detach(self, _: Context, client_pid: u32) {
        unimplemented!("detach(client_pid={client_pid})")
    }

    async fn standby(self, _: Context, client_pids: Vec<u32>) {
        unimplemented!("standby(client_pids={client_pids:?})")
    }
}

#[derive(Default)]
pub struct DaemonInner {
    next_id: i32,
    children: BTreeMap<u32, Child>,
    pub standby_worker: Option<StandbyWorker>,
}

pub struct Child {
    pub job_name: String,
    pub cuda_visible_devices: String,
    /// None if client detached.
    pub control: Option<(libc::pid_t, ControlClient)>,
    pub ids: Vec<i32>,
}

pub struct StandbyWorker {
    pub worker_pid: libc::pid_t,
    pub control: ControlClient,
    pub client_pids: Vec<u32>,
}

impl DaemonInner {
    fn next_id(&mut self) -> i32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn reserve_ids(&mut self, min_id: i32, max_id: i32) {
        assert!(self.next_id <= min_id && min_id <= max_id);
        self.next_id = max_id + 1;
    }

    pub fn controls_for_job(&self, job_name: &str) -> Vec<ControlClient> {
        self.children
            .values()
            .filter(|child| child.job_name == job_name)
            .filter_map(|child| child.control.as_ref().map(|(_, control)| control.clone()))
            .collect()
    }

    pub async fn get_or_spawn_child(
        &mut self,
        client_pid: u32,
        job_name: String,
        cuda_visible_devices: String,
        config: &Config,
    ) -> &mut Child {
        self.reap_children();
        match self.children.entry(client_pid) {
            Entry::Occupied(entry) => {
                let child = entry.into_mut();
                assert_eq!(child.job_name, job_name);
                assert_eq!(child.cuda_visible_devices, cuda_visible_devices);
                child
            }
            Entry::Vacant(entry) => {
                let (pid, control) = Self::spawn_worker(&cuda_visible_devices, config).await;
                let child = Child {
                    job_name,
                    cuda_visible_devices,
                    control: Some((pid, control)),
                    ids: Vec::new(),
                };
                entry.insert(child)
            }
        }
    }

    pub fn get_child_mut(&mut self, client_pid: u32) -> &mut Child {
        self.children
            .get_mut(&client_pid)
            .unwrap_or_else(|| panic!("unknown client_pid {client_pid}"))
    }

    pub fn detach(&mut self, client_pid: u32) -> ControlClient {
        self.reap_children();
        let child = self.get_child_mut(client_pid);
        if child.ids.len() != 1 {
            panic!(
                "client_pid {client_pid} must have exactly one client id, found {:?}",
                child.ids
            );
        }
        let (_, control) = child
            .control
            .take()
            .unwrap_or_else(|| panic!("client_pid {client_pid} is already detached"));
        control
    }

    pub fn get_child_for_attach(&mut self, client_pid: u32) -> &mut Child {
        self.reap_children();
        let child = self.get_child_mut(client_pid);
        assert_eq!(child.ids.len(), 1, "client_pid {client_pid} must have exactly one client id");
        assert!(child.control.is_none(), "client_pid {client_pid} is not detached");
        child
    }

    pub fn standby_request(&self, client_pids: &[u32]) -> StandbyRequest {
        assert!(!client_pids.is_empty(), "standby target pids cannot be empty");
        assert!(self.standby_worker.is_none(), "standby worker already exists");

        let mut dedup = BTreeSet::new();
        let mut clients = Vec::with_capacity(client_pids.len());
        let mut cuda_visible_devices = None;
        for &client_pid in client_pids {
            assert!(dedup.insert(client_pid), "duplicate client_pid {client_pid} in standby list");
            let child = self
                .children
                .get(&client_pid)
                .unwrap_or_else(|| panic!("unknown client_pid {client_pid}"));
            assert!(child.control.is_none(), "client_pid {client_pid} is not detached");
            match &cuda_visible_devices {
                None => {
                    cuda_visible_devices = Some(child.cuda_visible_devices.clone());
                }
                Some(expected) => {
                    assert_eq!(
                        child.cuda_visible_devices, *expected,
                        "standby target must share CUDA_VISIBLE_DEVICES"
                    );
                }
            }
            clients.push(StandbyClient { client_pid, id: child.ids[0] });
        }
        StandbyRequest { clients, cuda_visible_devices: cuda_visible_devices.unwrap() }
    }

    pub fn set_standby_worker(&mut self, standby_worker: StandbyWorker) {
        assert!(self.standby_worker.is_none(), "standby worker already exists");
        self.standby_worker = Some(standby_worker);
    }

    pub fn take_standby_worker(&mut self, client_pid: u32) -> Option<StandbyWorker> {
        self.standby_worker.take_if(|standby| standby.client_pids.contains(&client_pid))
    }

    pub async fn spawn_worker(
        cuda_visible_devices: &str,
        config: &Config,
    ) -> (libc::pid_t, ControlClient) {
        let socket_path = make_socket_path();
        if Path::new(&socket_path).exists() {
            fs::remove_file(&socket_path).unwrap();
        }

        let mut listener = unix::listen(&socket_path, Bincode::default).await.unwrap();
        listener.config_mut().max_frame_length(usize::MAX);

        #[expect(clippy::zombie_processes)] // see reap_children()
        let child = Command::new(env::current_exe().unwrap())
            .env(SERVER_WORKER_SOCKET_ENV, &socket_path)
            .env("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
            .spawn()
            .unwrap();

        let transport = listener
            .next()
            .await
            .expect("worker unix listener closed")
            .unwrap_or_else(|e| panic!("worker unix accept failed: {e}"));
        let _ = fs::remove_file(&socket_path);
        let control = ControlClient::new(Default::default(), transport).spawn();
        control.assert_same_config(Context::current(), config.clone()).await.unwrap();
        let pid = i32::try_from(child.id()).unwrap();
        log::info!("[worker:{pid}] spawned, CUDA_VISIBLE_DEVICES: {cuda_visible_devices}");
        (pid, control)
    }

    fn reap_children(&mut self) {
        let mut finished = Vec::new();
        while let pid @ 1.. = unsafe { libc::waitpid(-1, ptr::null_mut(), libc::WNOHANG) } {
            finished.push(pid);
        }
        if finished.is_empty() {
            return;
        }
        if let Some(standby) = &self.standby_worker
            && finished.contains(&standby.worker_pid)
        {
            self.standby_worker = None;
        }
        self.children.retain(|_, child| {
            let Some((pid, _)) = &child.control else {
                return true; // retain detached children
            };
            !finished.contains(pid)
        });
    }
}
