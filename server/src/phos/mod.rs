pub mod cli;
mod daemon;
pub mod detach;
pub mod handle;
pub mod kernel;
pub mod memory;
mod nvml;
mod standby;

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Barrier, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use std::{fmt, mem, ptr};

pub use daemon::PhosDaemon;
use network::channel::{Receiver, Sender};
use network::config::{CommonConfig, Config};
use network::oob::{CheckpointMode, CheckpointRequest};
use network::restore::{BlackHole, RestoreVec};
use network::ringbufferchannel::SHM_FLAG_WRITE_DISABLED;
pub use standby::take_standby;
use tarpc::context::Context;

use crate::ServerThread;
use crate::control::{Control, RestoreProcessRequest, StandbyRequest};
use crate::worker::{ThreadLaunchMode, WorkerProcess};

static CLIENT_PID: AtomicU32 = AtomicU32::new(0);

pub fn checkpoint_thread(server: &mut ServerThread) -> CheckpointMode {
    CLIENT_PID.store(server.client_pid, Ordering::Relaxed);
    let request = REQUEST.read().unwrap();
    let request = request.as_ref().unwrap();
    let barrier = BARRIER.read().unwrap();
    let barrier = barrier.as_ref().unwrap();
    barrier.wait();
    match request.mode {
        CheckpointMode::Kill | CheckpointMode::LeaveRunning => {
            if server.is_main_thread {
                let dir = Path::new(&request.ckpt_dir).join("criu");
                fs::create_dir_all(&dir).unwrap();
                Command::new("criu")
                    .arg("dump")
                    .arg("--images-dir")
                    .arg(&dir)
                    .arg("--tree")
                    .arg(server.client_pid.to_string())
                    .arg("--shell-job")
                    .arg("--display-stats")
                    .arg("--link-remap")
                    .arg("--tcp-established")
                    .spawn()
                    .unwrap()
                    .wait()
                    .unwrap();
            }
            dump_handles(server, &request.ckpt_dir).unwrap();
            barrier.wait();
            clear_flag(&server.channel_receiver);
        }
        CheckpointMode::Detach => {
            detach::dump_state(server);
            standby::try_put_standby(server);
            barrier.wait();
        }
    }
    nvml::log_memory("after checkpoint");
    request.mode
}

fn call_criu_restore(ckpt_dir: &str) {
    let dir = Path::new(ckpt_dir).join("criu");
    Command::new("criu")
        .arg("restore")
        .arg("--images-dir")
        .arg(&dir)
        .arg("--shell-job")
        .arg("--display-stats")
        .arg("--restore-detached")
        .arg("--tcp-close")
        .spawn()
        .unwrap()
        .wait()
        .unwrap();
}

fn dump_handles(server: &mut ServerThread, ckpt_dir: &str) -> Result<(), io::Error> {
    let mut path = get_subdir(server, ckpt_dir);
    fs::create_dir_all(&path)?;
    path.push("handles.bin");
    server.resources.serialize(&mut BufWriter::new(File::create(&path)?))?;
    path.set_file_name("pin_memory.bin");
    fs::write(&path, [u8::from(server.is_pinned_memory)])?;
    Ok(())
}

pub fn begin_handle_restore(server: &mut ServerThread, ckpt_dir: &str) -> (Sender, Receiver) {
    let mut path = get_subdir(server, ckpt_dir);
    path.push("pin_memory.bin");
    match fs::read(&path).unwrap().as_slice() {
        [0] => {}
        [1] => {
            initialize_context();
            server.pin_memory();
        }
        content => panic!("invalid pin_memory.bin content: {content:#x?}"),
    }
    path.set_file_name("handles.bin");
    let receiver = mem::replace(
        &mut server.channel_receiver,
        Receiver::RestoreVec(RestoreVec::new(fs::read(&path).unwrap())),
    );
    let sender = mem::replace(&mut server.channel_sender, Sender::BlackHole(BlackHole));
    (sender, receiver)
}

// FIXME: should include setting device, etc.
pub fn initialize_context() {
    let start = Instant::now();
    let result = unsafe { cudasys::cudart::cudaFree(ptr::null_mut()) };
    assert_eq!(result, Default::default());
    let elapsed_ms = start.elapsed().as_millis();
    if elapsed_ms >= 700 {
        panic!("CUDA context initialization took too long: {elapsed_ms} ms.");
    }
}

fn get_subdir(server: &ServerThread, ckpt_dir: &str) -> PathBuf {
    format!("{ckpt_dir}/{}_{}", server.client_pid, server.id).into()
}

fn is_locked(channel: &Receiver) -> bool {
    let Receiver::Shm(channel) = channel else { panic!("SHM channel expected") };
    let flag = unsafe { AtomicU64::from_ptr(channel.flag_ptr()) };
    flag.load(Ordering::Relaxed) & SHM_FLAG_WRITE_DISABLED != 0
}

fn clear_flag(channel: &Receiver) {
    let Receiver::Shm(channel) = channel else { panic!("SHM channel expected") };
    let flag = unsafe { AtomicU64::from_ptr(channel.flag_ptr()) };
    flag.fetch_and(!SHM_FLAG_WRITE_DISABLED, Ordering::Release);
}

pub fn check_config(config: &CommonConfig) {
    assert_eq!(config.comm_type, "shm", "PhOS only supports SHM communication");
    assert!(!config.emulator, "PhOS does not support emulator");
    assert!(config.opt_shadow_desc, "PhOS requires shadow (proxied) handles")
}

struct FlagManager {
    threads: Vec<JoinHandle<()>>,
    flags: BTreeMap<NonZeroU64, *mut u64>,
}

unsafe impl Send for FlagManager {}

static MANAGER: Mutex<FlagManager> =
    Mutex::new(FlagManager { threads: Vec::new(), flags: BTreeMap::new() });

static REQUEST: RwLock<Option<CheckpointRequest>> = RwLock::new(None);
static BARRIER: RwLock<Option<Barrier>> = RwLock::new(None);

fn register_thread(handle: JoinHandle<()>) {
    let mut manager = MANAGER.lock().unwrap();
    manager.threads.push(handle);
}

pub fn register_flag(channel: &Receiver) {
    let Receiver::Shm(channel) = channel else { panic!("SHM channel expected") };
    let ptr = channel.flag_ptr();
    let mut manager = MANAGER.lock().unwrap();
    let dupe = manager.flags.insert(thread::current_id().as_u64(), ptr);
    assert!(dupe.is_none());
}

fn request_checkpoint(request: CheckpointRequest) {
    *REQUEST.write().unwrap() = Some(request);
    let mut manager = MANAGER.lock().unwrap();
    let mut ids = Vec::with_capacity(manager.threads.len());
    manager.threads.retain(|handle| {
        if handle.is_finished() {
            return false;
        }
        ids.push(handle.thread().id().as_u64());
        true
    });
    *BARRIER.write().unwrap() = Some(Barrier::new(ids.len() + 1));
    let mut count = 0;
    manager.flags.retain(|id, ptr| {
        if !ids.contains(id) {
            return false;
        }
        let flag = unsafe { AtomicU64::from_ptr(*ptr) };
        let old = flag.fetch_or(SHM_FLAG_WRITE_DISABLED, Ordering::Release);
        log::debug!("setting flag 0x2, old value: {old:#x}");
        count += 1;
        true
    });
    assert_eq!(count, ids.len());
}

fn block_and_dump_process_state() {
    let guard = BARRIER.read().unwrap();
    let barrier = guard.as_ref().unwrap();
    barrier.wait();
    let request_guard = REQUEST.read().unwrap();
    let request = request_guard.as_ref().unwrap();
    match request.mode {
        CheckpointMode::Kill | CheckpointMode::LeaveRunning => {
            let ckpt_dir = request.ckpt_dir.as_str();
            let client_pid = CLIENT_PID.load(Ordering::Relaxed);
            log::info!("dumping kernel modules...");
            let kernel_path = format!("{ckpt_dir}/{client_pid}_kernel.bin");
            fs::write(&kernel_path, kernel::dump()).unwrap();
            log::info!("dumped kernel modules");
            log::info!("dumping memory...");
            let memory_path = format!("{ckpt_dir}/{client_pid}_mem");
            memory::dump(memory_path.into());
            log::info!("dumped memory");
        }
        CheckpointMode::Detach => {}
    }
    drop(request_guard);
    barrier.wait();
    drop(guard);
    *BARRIER.write().unwrap() = None;
    *REQUEST.write().unwrap() = None;
}

fn restore_process_state(ckpt_dir: &str, client_pid: u32) {
    let kernel_path = format!("{ckpt_dir}/{client_pid}_kernel.bin");
    kernel::restore(&fs::read(&kernel_path).unwrap());
    let memory_path = format!("{ckpt_dir}/{client_pid}_mem");
    memory::restore(memory_path.into());
}

#[derive(Clone, Copy)]
pub struct PhosWorkerProcess<'p>(pub &'p WorkerProcess);

impl<'p> Control for PhosWorkerProcess<'p> {
    async fn assert_same_config(self, context: Context, config: Config) {
        self.0.assert_same_config(context, config).await
    }

    async fn create_thread(self, _: Context, id: i32, client_pid: u32) {
        register_thread(self.0.spawn_thread(id, client_pid, ThreadLaunchMode::Normal));
    }

    async fn checkpoint_process(self, _: Context, request: CheckpointRequest) {
        request_checkpoint(request);
        block_and_dump_process_state();
    }

    async fn restore_process(self, _: Context, request: RestoreProcessRequest) {
        let RestoreProcessRequest { job_name: _, ckpt_dir, client_pid, mut ids } = request;
        restore_process_state(&ckpt_dir, client_pid);
        ids.sort_unstable();

        for id in ids {
            register_thread(self.0.spawn_thread(
                id,
                client_pid,
                ThreadLaunchMode::RestoreDisk(ckpt_dir.clone()),
            ));
        }
    }

    async fn attach(self, _: Context, id: i32, client_pid: u32) {
        let mode = if standby::has_standby(client_pid, id) {
            ThreadLaunchMode::AttachStandby
        } else {
            ThreadLaunchMode::AttachCold
        };
        register_thread(self.0.spawn_thread(id, client_pid, mode));
    }

    async fn standby(self, _: Context, request: StandbyRequest) {
        standby::standby(&self.0.config.common, &request.clients);
    }

    async fn detach(self, _: Context) -> Option<Vec<u32>> {
        let result = standby::get_client_pids();
        if result.is_some() {
            memory::reset();
        }
        result
    }
}

struct ByteSize(pub u64);

impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 1 << 20 {
            write!(f, "{:.2} MiB", self.0 as f64 / (1 << 20) as f64)
        } else if self.0 >= 1 << 10 {
            write!(f, "{:.2} KiB", self.0 as f64 / (1 << 10) as f64)
        } else {
            write!(f, "{} B", self.0)
        }
    }
}
