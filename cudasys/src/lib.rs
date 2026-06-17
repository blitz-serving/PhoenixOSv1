#![expect(non_snake_case)]
#![feature(try_trait_v2)]
#![feature(try_trait_v2_residual)]

// Type definitions extracted from the bindings.
pub mod types;

mod hooks;

/// Functions requiring at least CUDA 12 can be marked with
/// `#[cfg(cuda_version = "12")]`.
/// 
/// Functions working *only* on CUDA 12 can be marked with
/// `#[cfg(all(cuda_version = "12", not(cuda_version = "13")))]`.
pub fn emit_cuda_version_cfg() {
    println!("cargo::rustc-check-cfg=cfg(cuda_version, values(\"11\", \"12\", \"13\"))");

    if cuda::CUDA_VERSION >= 11000 {
        println!("cargo::rustc-cfg=cuda_version=\"11\"");
    }
    if cuda::CUDA_VERSION >= 12000 {
        println!("cargo::rustc-cfg=cuda_version=\"12\"");
    }
    if cuda::CUDA_VERSION >= 13000 {
        println!("cargo::rustc-cfg=cuda_version=\"13\"");
    }
}

pub mod cuda {
    pub use crate::types::cuda::*;
    include!("bindings/funcs/cuda.rs");
}

pub mod cudart {
    pub use crate::types::cudart::*;
    include!("bindings/funcs/cudart.rs");
}

pub mod nvml {
    pub use crate::types::nvml::*;
    include!("bindings/funcs/nvml.rs");
}

pub mod cudnn {
    pub use crate::types::cudnn::*;
    include!("bindings/funcs/cudnn.rs");
}

pub mod cublas {
    pub use crate::types::cublas::*;
    include!("bindings/funcs/cublas.rs");
}

pub mod cublasLt {
    pub use crate::types::cublasLt::*;
    include!("bindings/funcs/cublasLt.rs");
}

pub mod nvrtc {
    pub use crate::types::nvrtc::*;
    include!("bindings/funcs/nvrtc.rs");
}

pub mod nccl {
    pub use crate::types::nccl::*;
    include!("bindings/funcs/nccl.rs");
}

#[cfg(test)]
mod tests {
    use super::*;

    // This should work without GPU
    #[test]
    fn get_version() {
        let mut version: i32 = 0;
        let result = unsafe { cudart::cudaDriverGetVersion(&mut version as *mut i32) };
        if result != cudart::cudaError::cudaSuccess {
            panic!("Cannot get driver version: ERROR={:?}", result);
        }
        println!("Version = {}", version);
    }
}
