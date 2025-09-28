# CUDASYS

This is Rust bindings for the CUDA toolkit APIs. The bindings are generated using [bindgen](https://github.com/rust-lang/rust-bindgen).

## Build Script

The code is in `build.rs`. Logic:

- Use `bindgen` to generate Rust bindings for the *raw* CUDA toolkit APIs.
- Split the bindings into separate parts (`types` and `funcs`) for client/server separating usage, `types` part is in `src/types.rs`, `funcs` part is in `src/lib.rs`, and raw bindings code are left in `src/bindings`.

## Usage

- For client part, only need types, so it `use cudasys::types::cuda::*`, etc.
- For server part, need both types and funcs, so it `use cudasys::cuda::*`, etc.
