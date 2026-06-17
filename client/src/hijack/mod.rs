#![expect(non_snake_case)]

mod client;
mod handle;

mod cuda_hijack;
mod cuda_hijack_custom;
mod cuda_hijack_utils;
mod cuda_unimplement;
mod cudart_hijack;
mod cudart_hijack_custom;
mod cudart_register;
mod cudart_unimplement;
mod nvml_hijack;
mod nvml_unimplement;
mod cudnn_hijack;
mod cudnn_hijack_custom;
mod cudnn_unimplement;
mod cublas_hijack;
mod cublas_unimplement;
mod cublasLt_hijack;
mod cublasLt_unimplement;
mod nvrtc_hijack;
mod nvrtc_unimplement;
mod nccl_hijack;
mod nccl_unimplement;

use codegen::cuda_hook_hijack;

use crate::elf::{FatBinaryHeader, FatBinaryWrapper};
use client::{ClientThread, FatBinaryHandle, HostPtr, DRIVER_CACHE, RUNTIME_CACHE};
use handle::next_handle;

fn cuda_unimplemented(name: &'static str) -> ! {
    const PRINT_MAPS: bool = false;
    if PRINT_MAPS {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        eprintln!("/proc/self/maps:\n{maps}");
    }
    #[cfg(feature = "python-stack")]
    pyo3::Python::try_attach(|py| {
        use pyo3::prelude::*;
        let traceback = PyModule::import(py, "traceback").unwrap();
        traceback.call_method0("print_stack").unwrap();
    });
    #[cfg(not(feature = "python-stack"))]
    log::warn!("enable `python-stack` feature to print Python stack traces");
    unimplemented!("{name}")
}
