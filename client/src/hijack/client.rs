use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::io::{Read as _, Write as _};
use std::net::TcpStream;
use std::sync::RwLock;
use std::{process, thread};

use cudasys::types::cuda::{CUfunction, CUmodule};
#[cfg(feature = "rdma")]
use network::ringbufferchannel::RDMAChannel;
use network::ringbufferchannel::{EmulatorChannel, SHMChannel};
use network::session::SendSession;
use network::{Channel, tcp};

use crate::elf::{FatBinaryHeader, KernelParamInfo};

pub struct ClientThread {
    pub id: i32,
    pub channel_sender: Channel,
    pub channel_receiver: Channel,
    pub cuda_device: Option<c_int>,
    pub cuda_device_init: bool,
    pub cuda_pctx_flags: Option<c_uint>,
    pub opt_async_api: bool,
    pub opt_shadow_desc: bool,
    pub opt_local: bool,
}

impl ClientThread {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the client side, the sender's name is ctos_channel_name,
    // receiver's name is stoc_channel_name.
    fn new() -> Self {
        log::info!("[{}:{}] client init", std::file!(), std::line!());
        for (i, arg) in std::env::args().enumerate() {
            log::info!("arg[{i}]: {arg}");
        }
        for (key, value) in std::env::vars() {
            if key.starts_with("LD_") || key.starts_with("RUST_") {
                log::info!("{key}: {value}");
            }
        }
        let config = network::NetworkConfig::read_from_file();
        let id = {
            let mut stream = TcpStream::connect(&config.daemon_socket).unwrap();
            stream.write_all(&process::id().to_be_bytes()).unwrap();
            let mut buf = [0u8; 4];
            stream.read_exact(&mut buf).unwrap();
            i32::from_be_bytes(buf)
        };
        log::info!("[#{id}] PID = {}, {:?}", process::id(), thread::current().id());
        let (channel_sender, channel_receiver) = match config.comm_type.as_str() {
            "shm" => {
                let (sender, receiver) = SHMChannel::new_client_with_id(&config, id).unwrap();
                if config.emulator {
                    (
                        Channel::new(Box::new(EmulatorChannel::new(sender, &config))),
                        Channel::new(Box::new(EmulatorChannel::new(receiver, &config))),
                    )
                } else {
                    (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
                }
            }
            "tcp" => {
                let (sender, receiver) = tcp::new_client(&config, id).unwrap();
                (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
            }
            #[cfg(feature = "rdma")]
            "rdma" => {
                let (sender, receiver) = RDMAChannel::new_client(&config, id);
                (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
            }
            &_ => panic!("Unsupported communication type in config"),
        };

        CLIENT_THREAD_INIT.set(true);
        unsafe {
            unsafe extern "C" fn atfork() {
                assert!(!CLIENT_THREAD_INIT.get());
            }
            assert_eq!(0, libc::pthread_atfork(Some(atfork), None, None));

            // HACK: should just send something to the daemon socket
            fn atsignal(_info: &libc::siginfo_t) {
                process::exit(0);
            }
            signal_hook_registry::register_sigaction(libc::SIGQUIT, atsignal).unwrap();
            signal_hook_registry::register_sigaction(libc::SIGTERM, atsignal).unwrap();
        }

        Self {
            id,
            channel_sender,
            channel_receiver,
            cuda_device: None,
            cuda_device_init: false,
            cuda_pctx_flags: None,
            opt_async_api: config.opt_async_api,
            opt_shadow_desc: config.opt_shadow_desc,
            opt_local: config.opt_local,
        }
    }

    pub fn with_borrow<F: FnOnce(&Self) -> R, R>(f: F) -> R {
        CLIENT_THREAD.with_borrow(f)
    }

    pub fn with_borrow_mut<F: FnOnce(&mut Self) -> R, R>(f: F) -> R {
        CLIENT_THREAD.with_borrow_mut(f)
    }

    pub fn before_call(&mut self, name: &'static str) {
        log::debug!(target: name, "[#{}]", self.id);
        #[cfg(feature = "phos")]
        crate::phos::set_client_flag_blocking(&self.channel_sender);
    }

    pub fn after_call(&mut self) {
        #[cfg(feature = "phos")]
        crate::phos::clear_client_flag(&self.channel_sender);
    }
}

impl Drop for ClientThread {
    fn drop(&mut self) {
        let proc_id: i32 = -1;
        let send_ctx = SendSession::begin(self.id, &self.channel_sender, "drop");
        send_ctx.send(&proc_id, "proc_id");
        send_ctx.finish();
    }
}

thread_local! {
    static CLIENT_THREAD: RefCell<ClientThread> = RefCell::new(ClientThread::new());
    static CLIENT_THREAD_INIT: Cell<bool> = const { Cell::new(false) };
}

pub static DRIVER_CACHE: RwLock<DriverCache> = RwLock::new(DriverCache::new());
pub static RUNTIME_CACHE: RwLock<RuntimeCache> = RwLock::new(RuntimeCache::new());

pub struct DriverCache {
    /// Used in `cuModuleGetFunction`, populated by `cuModuleLoadData`.
    images: BTreeMap<CUmodule, Cow<'static, [u8]>>,
    /// Used in `cuLaunchKernel`, populated by `cuModuleGetFunction`.
    function_params: BTreeMap<CUfunction, Box<[KernelParamInfo]>>,
}

// The pointers are server-side.
unsafe impl Send for DriverCache {}
unsafe impl Sync for DriverCache {}

impl DriverCache {
    const fn new() -> Self {
        Self { images: BTreeMap::new(), function_params: BTreeMap::new() }
    }

    pub fn insert_image(&mut self, module: CUmodule, image: &'static [u8], is_runtime: bool) {
        let image = if is_runtime {
            Cow::Borrowed(image)
        } else {
            Cow::Owned(image.to_vec())
        };
        assert!(self.images.insert(module, image).is_none());
    }

    pub fn insert_function(&mut self, hfunc: CUfunction, hmod: CUmodule, name: &CStr) {
        let image = self.images.get(&hmod).unwrap();
        let name = name.to_str().unwrap();
        let params = if let Some(fatbin) = FatBinaryHeader::cast(image.as_ptr()) {
            fatbin.find_kernel_params(name)
        } else {
            crate::elf::find_kernel_params(image, name)
        };
        assert!(self.function_params.insert(hfunc, params).is_none());
    }

    pub fn get_params(&self, f: CUfunction) -> &[KernelParamInfo] {
        self.function_params.get(&f).unwrap()
    }
}

pub struct RuntimeCache {
    pub cuda_device: Option<c_int>,
    /// Populated by `__cudaRegisterFatBinary`.
    pub lazy_fatbins: Vec<*const FatBinaryHeader>,
    /// Populated by `__cudaRegisterFunction`.
    pub lazy_functions: BTreeMap<HostPtr, (FatBinaryHandle, *const c_char)>,
    /// Populated by `__cudaRegisterVar`.
    pub lazy_variables: BTreeMap<HostPtr, (FatBinaryHandle, *const c_char)>,
    /// Result of `cuModuleLoadData` calls.
    pub loaded_modules: BTreeMap<FatBinaryHandle, CUmodule>,
    /// Used in `cudaLaunchKernel`. Cache of `cuModuleGetFunction` calls.
    pub loaded_functions: BTreeMap<HostPtr, CUfunction>,
}

// The pointers are either static or server-side.
unsafe impl Send for RuntimeCache {}
unsafe impl Sync for RuntimeCache {}

impl RuntimeCache {
    const fn new() -> Self {
        Self {
            cuda_device: None,
            lazy_fatbins: Vec::new(),
            lazy_functions: BTreeMap::new(),
            lazy_variables: BTreeMap::new(),
            loaded_modules: BTreeMap::new(),
            loaded_functions: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct FatBinaryHandle(usize);

impl FatBinaryHandle {
    pub fn from_index(index: usize) -> Self {
        Self((index + 1) << 4)
    }

    pub fn to_index(self) -> usize {
        (self.0 >> 4) - 1
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct HostPtr(*const c_void);
