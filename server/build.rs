fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    hookgen::generate_impls(
        "../cudasys/src/hooks/{}.rs",
        "../cudasys/src/bindings/funcs",
        "./src/dispatcher",
        "_exe",
        None,
        (cudasys::cuda::CUDA_VERSION / 1000) as u8,
    );
}
