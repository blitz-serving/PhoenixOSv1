use std::ffi::OsString;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::{process, thread};

use cudasys::cudart::{cudaError_t, cudaGetDeviceCount};
use futures::StreamExt as _;
use log::{error, info};
use network::CommChannelError;
use network::channel::{Listener, Receiver, Sender, attach_server};
use network::config::Config;
use network::oob::CheckpointRequest;
use tarpc::context::Context;
use tarpc::serde_transport::unix;
use tarpc::server::{BaseChannel, Channel as _};
use tarpc::tokio_serde::formats::Bincode;

use crate::control::{Control, RestoreProcessRequest, StandbyRequest};
use crate::dispatcher::dispatch;
#[cfg(not(feature = "phos"))]
use crate::handle::HandleManager;
#[cfg(feature = "phos")]
use crate::phos::{self, handle::HandleManager};

pub const SERVER_WORKER_SOCKET_ENV: &str = "PHOS_SERVER_WORKER_SOCKET";

pub async fn server_process(config: Config, socket_path: OsString) {
    let transport = unix::connect(socket_path, Bincode::default).await.unwrap();
    let service = WorkerProcess::new(config);
    #[cfg(feature = "phos")]
    let service = phos::PhosWorkerProcess(&service);

    BaseChannel::with_defaults(transport)
        .execute(service.serve())
        .for_each(|f| f)
        .await;

    log::info!("server process terminated because control channel is closed");
}

pub struct WorkerProcess {
    pub config: Arc<Config>,
    is_main_thread: AtomicBool,
}

#[cfg_attr(not(feature = "phos"), expect(dead_code))]
enum ThreadExitReason {
    Normal,
    CheckpointStop,
    Detached,
}

#[derive(Clone)]
#[cfg_attr(not(feature = "phos"), expect(dead_code))]
pub enum ThreadLaunchMode {
    Normal,
    RestoreDisk(String),
    AttachCold,
    AttachStandby,
}

impl WorkerProcess {
    fn new(config: Config) -> Self {
        Self { config: Arc::new(config), is_main_thread: AtomicBool::new(true) }
    }

    pub fn spawn_thread(
        &self,
        id: i32,
        client_pid: u32,
        mode: ThreadLaunchMode,
    ) -> thread::JoinHandle<()> {
        let config = Arc::clone(&self.config);
        // Listener must be started before responding to the client
        let listener = match mode {
            ThreadLaunchMode::Normal | ThreadLaunchMode::RestoreDisk(_) => {
                Some(Listener::new(&self.config.common, &self.config.server.network, id))
            }
            ThreadLaunchMode::AttachCold | ThreadLaunchMode::AttachStandby => None,
        };
        let is_main_thread = self.is_main_thread.swap(false, Ordering::AcqRel);
        thread::spawn(move || {
            let exit_reason =
                launch_server(&config, id, client_pid, is_main_thread, listener, mode);
            if !is_main_thread {
                return;
            }
            match exit_reason {
                ThreadExitReason::Normal => {
                    let delay = Duration::from_secs(3);
                    log::info!("Terminating process in {delay:?}...");
                    thread::sleep(delay);
                    process::exit(0);
                }
                ThreadExitReason::CheckpointStop => {
                    // TODO(phos): join other worker threads before terminating the process.
                }
                ThreadExitReason::Detached => {
                    // The main worker thread will exit after the control channel is closed.
                }
            }
        })
    }
}

impl Control for &WorkerProcess {
    async fn assert_same_config(self, _: Context, config: Config) {
        self.config.assert_same_server(&config);
    }

    async fn create_thread(self, _: Context, id: i32, client_pid: u32) {
        drop(self.spawn_thread(id, client_pid, ThreadLaunchMode::Normal));
    }

    async fn checkpoint_process(self, _: Context, request: CheckpointRequest) {
        unimplemented!("checkpoint_process({request:?})")
    }

    async fn restore_process(self, _: Context, request: RestoreProcessRequest) {
        unimplemented!("restore_process({request:?})")
    }

    async fn attach(self, _: Context, id: i32, client_pid: u32) {
        unimplemented!("attach(id={id}, client_pid={client_pid})")
    }

    async fn standby(self, _: Context, request: StandbyRequest) {
        unimplemented!("standby({request:?})")
    }

    async fn detach(self, _: Context) -> Option<Vec<u32>> {
        unimplemented!("detach()")
    }
}

pub struct ServerThread {
    pub id: i32,
    pub client_pid: u32,
    pub is_main_thread: bool,
    pub channel_sender: Sender,
    pub channel_receiver: Receiver,
    pub is_pinned_memory: bool,
    // TODO: handles should be process-wide; currently mitigated by using one channel at client side
    pub resources: HandleManager,
    pub opt_async_api: bool,
    pub opt_shadow_desc: bool,
}

impl ServerThread {
    pub fn before_call(&mut self, func: &'static str) {
        log::debug!(target: func, "[#{}]", self.id);
        if matches!(func, "cudaMalloc" | "cudaFree" | "cuMemCreate") {
            self.pin_memory();
        }
    }

    fn receive_request(&mut self) -> Result<i32, CommChannelError> {
        self.channel_receiver.recv()
    }

    pub fn pin_memory(&mut self) {
        if self.is_pinned_memory {
            return;
        }
        log::info!("registering channel buffers as pinned memory");
        pin_channel_buffers(&self.channel_sender, &self.channel_receiver);
        log::info!("channel buffers registered");
        self.is_pinned_memory = true;
    }
}

pub fn pin_channel_buffers(sender: &Sender, receiver: &Receiver) {
    let f = |ptr, len| {
        let result = unsafe { cudasys::cudart::cudaHostRegister(ptr as _, len, 0) };
        assert_eq!(result, cudaError_t::cudaSuccess);
    };
    sender.register(f);
    receiver.register(f);
}

pub struct Connection {
    pub receiver: Receiver,
    pub sender: Sender,
    pub is_pinned_memory: bool,
    pub handles: Option<HandleManager>,
}

fn launch_server(
    config: &Config,
    id: i32,
    client_pid: u32,
    is_main_thread: bool,
    listener: Option<Listener>,
    mode: ThreadLaunchMode,
) -> ThreadExitReason {
    let connection = match &mode {
        ThreadLaunchMode::Normal | ThreadLaunchMode::RestoreDisk(_) => {
            let (receiver, sender) =
                listener.unwrap().accept(&config.common, &config.server.network, id);
            Connection { receiver, sender, is_pinned_memory: false, handles: None }
        }
        ThreadLaunchMode::AttachCold => {
            assert!(listener.is_none());
            let (receiver, sender) = attach_server(&config.common, id);
            Connection { receiver, sender, is_pinned_memory: false, handles: None }
        }
        #[cfg(feature = "phos")]
        ThreadLaunchMode::AttachStandby => {
            assert!(listener.is_none());
            phos::take_standby(client_pid, id).unwrap()
        }
        #[cfg(not(feature = "phos"))]
        ThreadLaunchMode::AttachStandby => unreachable!(),
    };
    info!("[{}:{}] {} channel created", std::file!(), std::line!(), config.common.comm_type);

    let mut max_devices = 0;
    match unsafe { cudaGetDeviceCount(&mut max_devices) } {
        cudaError_t::cudaSuccess => info!("found {max_devices} cuda devices"),
        e => panic!("failed to find cuda devices: {e:?}"),
    }

    let mut server = ServerThread {
        id,
        client_pid,
        is_main_thread,
        channel_sender: connection.sender,
        channel_receiver: connection.receiver,
        is_pinned_memory: connection.is_pinned_memory,
        resources: connection.handles.unwrap_or_default(),
        opt_async_api: config.common.opt_async_api,
        opt_shadow_desc: config.common.opt_shadow_desc,
    };

    #[cfg(feature = "phos")]
    let mut channels = {
        phos::register_flag(&server.channel_receiver);
        match &mode {
            ThreadLaunchMode::Normal => None,
            ThreadLaunchMode::RestoreDisk(ckpt_dir) => {
                Some(phos::begin_handle_restore(&mut server, ckpt_dir))
            }
            ThreadLaunchMode::AttachCold => Some(phos::detach::begin_attach(&mut server, false)),
            ThreadLaunchMode::AttachStandby => Some(phos::detach::begin_attach(&mut server, true)),
        }
    };

    let exit_reason = loop {
        match server.receive_request() {
            Ok(-1) => break ThreadExitReason::Normal,
            Ok(proc_id) => dispatch(proc_id, &mut server),
            #[cfg(feature = "phos")]
            Err(CommChannelError::ShmChannelLocked) => {
                std::assert_matches!(
                    server.receive_request(),
                    Err(CommChannelError::ShmChannelLocked),
                    "remaining async requests",
                );
                let ckpt_mode = phos::checkpoint_thread(&mut server);
                log::info!("PhOS checkpoint done");
                use network::oob::CheckpointMode::*;
                match ckpt_mode {
                    Kill => break ThreadExitReason::CheckpointStop,
                    LeaveRunning => {}
                    Detach => break ThreadExitReason::Detached,
                }
            }
            #[cfg(feature = "phos")]
            Err(CommChannelError::RestoreEof) => {
                let (sender, receiver) = channels.take().unwrap();
                server.channel_sender = sender;
                server.channel_receiver = receiver;
                if matches!(mode, ThreadLaunchMode::AttachCold | ThreadLaunchMode::AttachStandby) {
                    phos::detach::on_restore_finished(&server.channel_receiver);
                }
                log::info!("handles restored");
            }
            Err(e) => {
                error!("failed to receive request: {e:?}");
                break ThreadExitReason::Normal;
            }
        }
    };

    info!("server #{id} (client PID: {client_pid}) terminated");
    exit_reason
}
