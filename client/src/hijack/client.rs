use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, RwLock};
use std::{env, process, thread};

use cudasys::types::cuda::{CUfunction, CUmodule};
use network::channel::{Receiver, Sender};
use network::config::Config;
use network::{channel, oob};

use crate::elf::{FatBinaryHeader, KernelParamInfo};

pub struct ClientThread {
    pub id: i32,
    pub channel_sender: Sender,
    pub channel_receiver: Receiver,
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
        for (i, arg) in env::args().enumerate() {
            log::info!("arg[{i}]: {arg}");
        }
        for (key, value) in env::vars() {
            if key.starts_with("LD_") || key.starts_with("RUST_") {
                log::info!("{key}: {value}");
            }
        }
        let Config { common: config, client: client_config, .. } = Config::read_env();
        log::info!("Using {:?}", config.checked_comm_type());
        let id = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let oob = oob::connect(&client_config.network.daemon_socket).await;
                oob.new_client_thread(
                    oob::current_context(),
                    oob::NewClientThreadRequest {
                        pid: process::id(),
                        job_name: client_config.job_name.clone(),
                        cuda_visible_devices: env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(),
                        config: config.clone(),
                        is_phos: cfg!(feature = "phos"),
                    },
                )
                .await
            })
            .unwrap();
        log::info!("[#{id}] PID = {}, {:?}", process::id(), thread::current_id());
        let (channel_sender, channel_receiver) =
            channel::client(&config, &client_config.network, id);

        CLIENT_THREAD_INIT.store(true, Ordering::Relaxed);
        unsafe {
            extern "C" fn atexit() {
                match CLIENT_THREAD.lock() {
                    Ok(mut client) => *client = None,
                    Err(e) => *e.into_inner() = None,
                }
            }
            assert_eq!(0, libc::atexit(atexit));

            unsafe extern "C" fn atfork() {
                assert!(!CLIENT_THREAD_INIT.load(Ordering::Relaxed));
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
            opt_local: client_config.opt_local,
        }
    }

    #[inline]
    pub fn with_borrow<F: FnOnce(&Self) -> R, R>(f: F) -> R {
        Self::with_borrow_mut(|client| f(client))
    }

    pub fn with_borrow_mut<F: FnOnce(&mut Self) -> R, R>(f: F) -> R {
        let mut client = CLIENT_THREAD.lock().unwrap();
        let client = client.get_or_insert_with(ClientThread::new);
        f(client)
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
        self.channel_sender.send(&proc_id).unwrap();
        self.channel_sender.flush().unwrap();
    }
}

static CLIENT_THREAD: Mutex<Option<ClientThread>> = Mutex::new(None);
static CLIENT_THREAD_INIT: AtomicBool = AtomicBool::new(false);

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
        if self.images.insert(module, image).is_some() {
            log::warn!("Module {module:p} already exists in driver cache, overwritten");
        }
    }

    pub fn insert_function(&mut self, hfunc: CUfunction, hmod: CUmodule, name: &CStr) {
        let image = self.images.get(&hmod).unwrap();
        let name = name.to_str().unwrap();
        let params = match FatBinaryHeader::cast(image.as_ptr()) {
            Some(fatbin) => fatbin.find_kernel_params(name),
            None => crate::elf::find_kernel_params(image, name),
        };
        let Some(params) = params else {
            panic!("kernel not found: {name}");
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
