use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::sync::Mutex;
use std::{mem, ptr};

use bincode::{Decode, Encode, config, decode_from_slice, encode_to_vec};
use cudasys::cuda::*;

use super::nvml;

pub fn cu_module_load_data(module: *mut CUmodule, image: Box<[u8]>) -> CUresult {
    let mut manager = KERNEL_MANAGER.lock().unwrap();
    let hash = manager.load_module(image)?;
    unsafe { *module = mem::transmute(hash) };
    CUresult::CUDA_SUCCESS
}

pub fn cu_module_get_function(hfunc: *mut CUfunction, module: CUmodule, name: CString) -> CUresult {
    let mut manager = KERNEL_MANAGER.lock().unwrap();
    let hash = unsafe { mem::transmute(module) };
    let hmod = manager.all_modules.get(&hash).unwrap().hmod;
    let result = unsafe { cuModuleGetFunction(hfunc, hmod, name.as_ptr()) };
    if result == CUresult::CUDA_SUCCESS {
        manager.functions.insert((hash, name));
    }
    result
}

pub fn dump() -> Vec<u8> {
    let manager = KERNEL_MANAGER.lock().unwrap();
    let checkpoint = manager.checkpoint();
    encode_to_vec(&checkpoint, config::standard()).unwrap()
}

pub fn restore(bytes: &[u8]) {
    let checkpoint: KernelCheckpoint = decode_checkpoint(bytes);
    let mut manager = KERNEL_MANAGER.lock().unwrap();
    manager.restore(checkpoint).unwrap();
}

pub fn standby(bytes: &[u8]) {
    let checkpoint: KernelCheckpoint = decode_checkpoint(bytes);
    let mut manager = KERNEL_MANAGER.lock().unwrap();
    manager.register_modules(checkpoint.modules).unwrap();
    manager.trigger_module_loads(&checkpoint.functions);
}

pub fn attach_warm(bytes: &[u8]) {
    let checkpoint: KernelCheckpointHeader = decode_checkpoint(bytes);
    let mut manager = KERNEL_MANAGER.lock().unwrap();
    manager.active_modules = checkpoint.active_modules;
    manager.functions.clear();
}

type Function = (ModuleHash, CString);

#[derive(Decode)]
struct KernelCheckpoint {
    active_modules: Vec<ModuleHash>,
    functions: Vec<Function>,
    modules: Vec<(ModuleHash, Box<[u8]>)>,
}

#[derive(Decode)]
struct KernelCheckpointHeader {
    active_modules: Vec<ModuleHash>,
}

#[derive(Encode)]
struct BorrowedKernelCheckpoint<'a> {
    active_modules: &'a [ModuleHash],
    functions: &'a BTreeSet<Function>,
    modules: BTreeMap<ModuleHash, &'a [u8]>,
}

fn decode_checkpoint<T: Decode<()>>(bytes: &[u8]) -> T {
    decode_from_slice(bytes, config::standard()).unwrap().0
}

pub fn get_all_images_size() -> usize {
    KERNEL_MANAGER
        .lock()
        .unwrap()
        .all_modules
        .values()
        .map(|module| module.image.len())
        .sum()
}

static KERNEL_MANAGER: Mutex<KernelManager> = Mutex::new(KernelManager::new());

/// Similar to `DriverCache` and `RuntimeCache` in client crate.
struct KernelManager {
    all_modules: BTreeMap<ModuleHash, Module>,
    active_modules: Vec<ModuleHash>,
    functions: BTreeSet<Function>,
}

// for CUmodule and CUfunction
unsafe impl Send for KernelManager {}

struct Module {
    image: Box<[u8]>,
    hmod: CUmodule,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Encode, Decode)]
#[repr(transparent)]
struct ModuleHash(usize);

impl ModuleHash {
    #[cfg(target_pointer_width = "64")]
    fn compute(image: &[u8]) -> Self {
        let hash = xxh3::hash64_with_seed(image, 0);
        Self(hash as usize)
    }
}

impl KernelManager {
    const fn new() -> Self {
        KernelManager {
            all_modules: BTreeMap::new(),
            active_modules: Vec::new(),
            functions: BTreeSet::new(),
        }
    }

    fn load_module(&mut self, image: Box<[u8]>) -> Result<ModuleHash, CUresult> {
        let hash = ModuleHash::compute(&image);
        self.register_module(hash, image)?;
        self.active_modules.push(hash);
        Ok(hash)
    }

    fn checkpoint(&self) -> BorrowedKernelCheckpoint<'_> {
        let modules = self
            .active_modules
            .iter()
            .map(|hash| (*hash, self.all_modules.get(hash).unwrap().image.as_ref()))
            .collect();
        BorrowedKernelCheckpoint {
            active_modules: self.active_modules.as_slice(),
            functions: &self.functions,
            modules,
        }
    }

    fn restore(&mut self, checkpoint: KernelCheckpoint) -> Result<(), CUresult> {
        assert!(
            self.all_modules.is_empty(),
            "kernel modules must be restored into an empty manager"
        );
        self.active_modules.clear();
        self.functions.clear();

        self.register_modules(checkpoint.modules)?;
        self.active_modules = checkpoint.active_modules;
        Ok(())
    }

    fn trigger_module_loads(&self, functions: &[Function]) {
        for (module, name) in functions {
            let Some(entry) = self.all_modules.get(module) else { continue };
            let mut hfunc = ptr::null_mut();
            let _ = unsafe { cuModuleGetFunction(&mut hfunc, entry.hmod, name.as_ptr()) };
        }
    }

    fn register_modules(&mut self, modules: Vec<(ModuleHash, Box<[u8]>)>) -> Result<(), CUresult> {
        if modules.is_empty() {
            return Ok(());
        }
        if self.all_modules.is_empty() {
            nvml::log_memory("before cudaFree(0)");
            super::initialize_context();
            nvml::log_memory("after cudaFree(0)");
        }
        for (hash, image) in modules {
            log::debug!(target: "kernel_hash", "{:016x} {}", hash.0, image.len());
            self.register_module(hash, image)?;
        }
        Ok(())
    }

    fn register_module(&mut self, hash: ModuleHash, image: Box<[u8]>) -> Result<(), CUresult> {
        if let Entry::Vacant(entry) = self.all_modules.entry(hash) {
            let mut hmod = ptr::null_mut();
            unsafe { cuModuleLoadData(&mut hmod, image.as_ptr().cast())? };
            entry.insert(Module { image, hmod });
        }
        Ok(())
    }
}
