use crate::{CommChannelError, RecvChannel, SendChannel};

pub struct RestoreVec {
    vec: Vec<u8>,
    pos: usize,
}

impl RestoreVec {
    pub const fn new(vec: Vec<u8>) -> Self {
        Self { vec, pos: 0 }
    }
}

impl RecvChannel for RestoreVec {
    fn get_bytes(&mut self, dst: &mut [u8]) -> Result<(), CommChannelError> {
        let end = self.pos + dst.len();
        if end > self.vec.len() {
            Err(CommChannelError::RestoreEof)
        } else {
            dst.copy_from_slice(&self.vec[self.pos..end]);
            self.pos = end;
            Ok(())
        }
    }

    fn recv<T: Copy>(&mut self) -> Result<T, CommChannelError> {
        let end = self.pos + size_of::<T>();
        if end > self.vec.len() {
            Err(CommChannelError::RestoreEof)
        } else {
            let result = unsafe { self.vec.as_ptr().add(self.pos).cast::<T>().read_unaligned() };
            self.pos = end;
            Ok(result)
        }
    }
}

pub struct BlackHole;

impl SendChannel for BlackHole {
    fn flush(&mut self) -> Result<(), CommChannelError> {
        Ok(())
    }

    fn put_bytes(&mut self, _src: &[u8]) -> Result<(), CommChannelError> {
        Ok(())
    }

    fn send<T: Copy>(&mut self, _src: &T) -> Result<(), CommChannelError> {
        Ok(())
    }

    fn send_unaligned<T: Copy>(&mut self, _src: *const T) -> Result<(), CommChannelError> {
        Ok(())
    }

    fn send_slice<T: Copy>(&mut self, _src: &[T]) -> Result<(), CommChannelError> {
        Ok(())
    }
}
