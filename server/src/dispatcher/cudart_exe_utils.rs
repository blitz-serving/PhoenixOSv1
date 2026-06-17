use cudasys::cudart::*;
use network::{MemRead, MemWrite};

pub struct Dtoh {
    pub src: *const u8,
    pub len: usize,
    pub stream: cudaStream_t,
}

impl MemRead for Dtoh {
    fn read(&mut self, buf: &mut [u8]) {
        let amount = buf.len();
        debug_assert!(amount <= self.len);

        log::trace!("Dtoh read: {amount} bytes of {} remaining", self.len);

        let status = if self.stream.is_null() {
            unsafe {
                cudaMemcpy(
                    buf.as_mut_ptr().cast(),
                    self.src.cast(),
                    amount,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            }
        } else {
            let status = unsafe {
                cudaMemcpyAsync(
                    buf.as_mut_ptr().cast(),
                    self.src.cast(),
                    amount,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    self.stream,
                )
            };
            if status == cudaError_t::cudaSuccess {
                unsafe { cudaStreamSynchronize(self.stream) }
            } else {
                status
            }
        };
        assert_eq!(status, cudaError_t::cudaSuccess);

        self.src = unsafe { self.src.add(amount) };
        self.len -= amount;
    }

    fn remaining(&self) -> usize {
        self.len
    }
}

pub struct Htod {
    pub dst: *mut u8,
    pub len: usize,
    pub stream: cudaStream_t,
}

impl MemWrite for Htod {
    fn write(&mut self, buf: &[u8]) {
        let amount = buf.len();
        debug_assert!(amount <= self.len);

        log::trace!("Htod write: {amount} bytes of {} remaining", self.len);

        let status = if self.stream.is_null() {
            unsafe {
                cudaMemcpy(
                    self.dst.cast(),
                    buf.as_ptr().cast(),
                    amount,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            }
        } else {
            let status = unsafe {
                cudaMemcpyAsync(
                    self.dst.cast(),
                    buf.as_ptr().cast(),
                    amount,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    self.stream,
                )
            };
            if status == cudaError_t::cudaSuccess {
                unsafe { cudaStreamSynchronize(self.stream) }
            } else {
                status
            }
        };
        assert_eq!(status, cudaError_t::cudaSuccess);

        self.dst = unsafe { self.dst.add(amount) };
        self.len -= amount;
    }

    fn remaining(&self) -> usize {
        self.len
    }
}
