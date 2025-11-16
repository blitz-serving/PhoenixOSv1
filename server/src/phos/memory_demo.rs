use std::time::{Duration, Instant};

use cudasys::cudart::{cudaDeviceSynchronize, cudaMemcpy, cudaMemcpyKind};

pub enum MemoryDemo {
    Disabled,
    Waiting { until: Instant },
}

impl MemoryDemo {
    pub fn start() -> Self {
        let until = Instant::now() + Duration::from_secs(5);
        Self::Waiting { until }
    }

    pub fn reset_and_restore(&mut self) {
        match self {
            Self::Disabled => return,
            Self::Waiting { until } => {
                if Instant::now() < *until {
                    return;
                }
            }
        }
        log::info!("Starting memory demo");
        let manager = &mut *super::memory::MEMORY_MANAGER.lock().unwrap();
        let mut result;
        result = unsafe { cudaDeviceSynchronize() };
        assert_eq!(result, Default::default());
        let mut total = 0;
        let bufs: Vec<_> = manager
            .iter()
            .unwrap()
            .map(|(ptr, size)| {
                total += size;
                let mut buf = Box::<[u8]>::new_uninit_slice(size);
                result = unsafe {
                    cudaMemcpy(
                        buf.as_mut_ptr().cast(),
                        ptr as _,
                        size,
                        cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    )
                };
                assert_eq!(result, Default::default());
                (ptr, unsafe { buf.assume_init() })
            })
            .collect();
        manager.reset();
        log::info!("Saved {} segments, total size: {} MiB", bufs.len(), total >> 20);
        let segments: Vec<_> = bufs.iter().map(|(ptr, buf)| (*ptr, buf.len())).collect();
        manager.restore(&segments);
        for (ptr, buf) in bufs {
            result = unsafe {
                cudaMemcpy(
                    ptr as _,
                    buf.as_ptr().cast(),
                    buf.len(),
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            };
            assert_eq!(result, Default::default());
        }
        log::info!("Restored {} segments", segments.len());
        *self = Self::Disabled;
    }
}
