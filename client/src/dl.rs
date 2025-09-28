use std::ffi::*;
use std::sync::{LazyLock, OnceLock};
use std::{env, mem};

// original dlsym
unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

static DLOPEN_ORIG: LazyLock<extern "C" fn(*const c_char, c_int) -> *mut c_void> =
    LazyLock::new(|| unsafe { mem::transmute(dlsym_next(c"dlopen")) });

pub fn dlsym_next(symbol: &CStr) -> *mut c_void {
    const RTLD_NEXT: *mut c_void = usize::MAX as _;
    let result = unsafe { dlsym(RTLD_NEXT, symbol.as_ptr()) };
    if result.is_null() {
        panic!("failed to find symbol {symbol:?}");
    }
    result
}

#[unsafe(no_mangle)]
extern "C" fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void {
    // use the original dlopen to load the library
    if filename.is_null() {
        return DLOPEN_ORIG(filename, flags);
    }
    let name = unsafe { CStr::from_ptr(filename) }.to_str().unwrap();
    if name.contains("cpython") {
        return DLOPEN_ORIG(filename, flags);
    }
    log::debug!(target: "dlopen", "{name} (flags: {flags:#x})");
    if name.contains("libcuda") || name.contains("libnvrtc.so") || name.contains("libnvidia-ml") || name.contains("libnccl.so") {
        if cfg!(feature = "passthrough") {
            assert!(!DLOPEN_ORIG(filename, 0x101).is_null());
        }
        // if the library is libcuda, libnvrtc or libnvidia-ml, return a handle to the client
        log::debug!(target: "dlopen", "replacing dlopen call to {} library with a handle to the client", name);
        static SELF_PATH: OnceLock<CString> = OnceLock::new();
        let self_path = SELF_PATH.get_or_init(|| {
            let mut result = env::var("LD_PRELOAD").unwrap().into_bytes();
            if result.last() == Some(&b':') {
                result.pop();
            }
            assert!(!result.contains(&b':'));
            CString::new(result).unwrap()
        });
        let self_handle = DLOPEN_ORIG(self_path.as_ptr(), flags);
        if self_handle.is_null() {
            panic!("Failed to load the client handle");
        }
        return self_handle;
    }
    DLOPEN_ORIG(filename, flags)
}
