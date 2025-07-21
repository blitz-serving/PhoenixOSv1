use std::{env, fs, path::PathBuf};

fn main() {
    create_cuda_symlinks();

    #[cfg(not(feature = "passthrough"))]
    hookgen::generate_impls(
        "../cudasys/src/hooks/{}.rs",
        "../cudasys/src/bindings/funcs",
        "./src/hijack",
        "_hijack",
        Some("_unimplement"),
        (cudasys::cuda::CUDA_VERSION / 1000) as u8,
    );

    #[cfg(feature = "passthrough")]
    hookgen::generate_passthrough("../cudasys/src/bindings/funcs", "./src/passthrough", |sig| {
        let name = sig.ident.to_string();
        if name.starts_with("cuGetProcAddress") {
            return None;
        }
        let c_name = std::ffi::CString::new(name.into_bytes()).unwrap();
        let name = c_name.to_str().unwrap();
        let inputs: Vec<_> = sig
            .inputs
            .iter()
            .map(|arg| match arg {
                syn::FnArg::Typed(arg) => arg,
                _ => panic!("unexpected argument type"),
            })
            .collect();
        let params = inputs.iter().map(|arg| arg.ty.as_ref());
        let output = &sig.output;
        let args = inputs.iter().map(|arg| arg.pat.as_ref());
        Some(syn::parse_quote!({
            #[thread_local]
            static __F: std::cell::OnceCell<extern "C" fn(#(#params),*) #output> =
                std::cell::OnceCell::new();
            let __f = __F.get_or_init(|| unsafe {
                std::mem::transmute(crate::dl::dlsym_next(#c_name))
            });
            super::begin(#name);
            let __result = __f(#(#args),*);
            super::end(#name);
            __result
        }))
    });

    println!("cargo:rerun-if-changed=build.rs");
}

/// Recursive calls in `dlopen(libtorch_global_deps.so)` escaped our hook in `dl.rs`.
/// This prevents them from loading local CUDA libs and helps diagnose missing symbols.
fn create_cuda_symlinks() {
    let mut symlink_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    while symlink_dir.file_name().unwrap().to_str().unwrap() != "build" {
        assert!(symlink_dir.pop());
    }
    assert!(symlink_dir.pop());
    symlink_dir.push("cuda-symlinks");
    let alt_dir = symlink_dir.with_file_name("cuda-symlinks-passthrough");
    if cfg!(feature = "passthrough") {
        match (symlink_dir.exists(), alt_dir.exists()) {
            (true, true) => fs::remove_dir_all(&symlink_dir).unwrap(),
            (true, false) => fs::rename(&symlink_dir, &alt_dir).unwrap(),
            _ => {}
        };
        return;
    }
    match (symlink_dir.exists(), alt_dir.exists()) {
        (false, true) => fs::rename(&alt_dir, &symlink_dir).unwrap(),
        (false, false) => fs::create_dir(&symlink_dir).unwrap(),
        _ => {}
    }
    symlink_dir.push("x");
    for lib in [
        "libcuda.so.1",
        "libcudart.so.11.0",
        "libcudart.so.12",
        "libnvidia-ml.so.1",
        "libcudnn.so.8",
        "libcudnn.so.9",
        "libcublas.so.11",
        "libcublas.so.12",
        "libcublasLt.so.11",
        "libcublasLt.so.12",
        "libnvrtc.so.11.2",
        "libnvrtc.so.11.3",
    ] {
        symlink_dir.set_file_name(lib);
        let _ = std::os::unix::fs::symlink("../libclient.so", &symlink_dir);
    }
}
