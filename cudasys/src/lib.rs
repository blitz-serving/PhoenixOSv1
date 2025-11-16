#![expect(non_snake_case)]
#![feature(try_trait_v2)]

// Type definitions extracted from the bindings.
pub mod types;

mod hooks;

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
