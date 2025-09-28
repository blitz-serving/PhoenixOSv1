use std::cell::Cell;
use std::fs::File;
use std::io::{ErrorKind, Read as _};
use std::slice;

use crate::{CommChannel, CommChannelError, CommChannelInnerIO, RawMemory, RawMemoryMut};

pub struct RestoreFile(pub File);

impl CommChannel for RestoreFile {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        unimplemented!()
    }
}

impl CommChannelInnerIO for RestoreFile {
    fn put_bytes(&self, _src: &RawMemory) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn try_put_bytes(&self, _src: &RawMemory) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        let buf = unsafe { slice::from_raw_parts_mut(dst.ptr, dst.len) };
        match (&self.0).read_exact(buf) {
            Ok(()) => Ok(dst.len),
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => Err(CommChannelError::RestoreEof),
            Err(e) => {
                log::error!("read failed: {e}");
                Err(CommChannelError::IoError)
            }
        }
    }

    fn try_get_bytes(&self, _dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn safe_try_get_bytes(&self, _dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        unimplemented!()
    }
}

pub struct RestoreVec {
    vec: Vec<u8>,
    pos: Cell<usize>,
}

impl RestoreVec {
    pub const fn new(vec: Vec<u8>) -> Self {
        Self { vec, pos: Cell::new(0) }
    }
}

impl CommChannel for RestoreVec {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        unimplemented!()
    }
}

impl CommChannelInnerIO for RestoreVec {
    fn put_bytes(&self, _src: &RawMemory) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn try_put_bytes(&self, _src: &RawMemory) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        let pos = self.pos.get();
        let end = pos + dst.len;
        if end > self.vec.len() {
            Err(CommChannelError::RestoreEof)
        } else {
            let buf = unsafe { slice::from_raw_parts_mut(dst.ptr, dst.len) };
            buf.copy_from_slice(&self.vec[pos..end]);
            self.pos.set(end);
            Ok(dst.len)
        }
    }

    fn try_get_bytes(&self, _dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn safe_try_get_bytes(&self, _dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        unimplemented!()
    }
}

pub struct BlackHole;

impl CommChannel for BlackHole {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        Ok(())
    }
}

impl CommChannelInnerIO for BlackHole {
    fn put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        Ok(src.len)
    }

    fn try_put_bytes(&self, _src: &RawMemory) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn get_bytes(&self, _dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn try_get_bytes(&self, _dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn safe_try_get_bytes(&self, _dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        unimplemented!()
    }
}
