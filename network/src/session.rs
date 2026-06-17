use std::ffi::{CStr, CString};
use std::fmt::Debug;
use std::slice;

use log::trace;

use crate::channel::{Receiver, Sender};
use crate::type_impl::{save, save_slice};
use crate::{CommChannelError, MemRead, MemWrite};

// TODO: stats, emulator

pub struct SendSession<'ch> {
    id: i32,
    channel: &'ch mut Sender,
    func: &'static str,
}

macro_rules! bail_send {
    ($name:ident) => {
        |e| panic!("failed to send {}: {}", $name, e)
    };
}

impl<'ch> SendSession<'ch> {
    pub fn begin(id: i32, channel: &'ch mut Sender, func: &'static str) -> Self {
        Self { id, channel, func }
    }

    pub fn check_not_null<T>(&self, ptr: *const T, name: &'static str) {
        if ptr.is_null() {
            panic!("[#{} {}] {name} is null", self.id, self.func);
        }
    }

    pub fn send<T>(&mut self, data: &T, name: &'static str)
    where
        T: Copy + Debug,
    {
        trace!(target: self.func, "[#{}] (send) {name} = {data:?}", self.id);
        self.channel.send(data).unwrap_or_else(bail_send!(name))
    }

    pub unsafe fn send_unaligned<T>(&mut self, data: *const T, name: &'static str)
    where
        T: Copy + Debug,
    {
        self.check_not_null(data, name);
        trace!(
            target: self.func,
            "[#{}] (send) {name} = {:?}",
            self.id,
            unsafe { data.read_unaligned() },
        );
        self.channel.send_unaligned(data).unwrap_or_else(bail_send!(name))
    }

    pub unsafe fn slice_from<'a, T>(
        &self,
        data: *const T,
        len: usize,
        name: &'static str,
    ) -> &'a [T] {
        self.check_not_null(data, name);
        unsafe { slice::from_raw_parts(data, len) }
    }

    pub fn send_slice<T>(&mut self, data: &[T], name: &'static str)
    where
        T: Copy,
    {
        trace!(target: self.func, "[#{}] (send) {name} = {data:p}[{}]", self.id, data.len());
        self.channel.send_slice(data).unwrap_or_else(bail_send!(name))
    }

    pub fn send_cstr(&mut self, data: &CStr, name: &'static str) {
        trace!(target: self.func, "[#{}] (send) {name} = {data:?}", self.id);
        self.channel
            .send_slice(data.to_bytes_with_nul())
            .unwrap_or_else(bail_send!(name))
    }

    pub fn put_bytes(&mut self, mut src: impl MemRead) -> Result<(), CommChannelError> {
        self.channel.put_bytes(&mut src)
    }

    #[inline]
    pub fn send_handle<T>(
        &mut self,
        output: *mut *mut T,
        name: &'static str,
        next_proxy: fn() -> usize,
    ) {
        self.send_handle_inner(output.cast(), name, next_proxy());
    }

    fn send_handle_inner(&mut self, output: *mut usize, name: &'static str, proxy: usize) {
        trace!(target: self.func, "[#{}] (send) {name} = {proxy:#x}", self.id);
        self.channel.send(&proxy).unwrap_or_else(bail_send!(name));
        unsafe { *output = proxy };
    }

    pub fn finish(self) {
        if let Sender::Emulator(channel) = self.channel {
            let name = "timestamp";
            channel.send_ts().unwrap_or_else(bail_send!(name))
        }
        match self.channel.flush() {
            Ok(()) => {}
            Err(e) => panic!("failed to flush_out: {}", e),
        }
    }
}

pub struct RecvSession<'ch> {
    pub id: i32,
    pub channel: &'ch mut Receiver,
    pub func: &'static str,
    pub save: Option<Vec<u8>>,
}

macro_rules! bail_recv {
    ($name:ident) => {
        |e| panic!("failed to receive {}: {}", $name, e)
    };
}

impl<'ch> RecvSession<'ch> {
    pub fn begin(id: i32, channel: &'ch mut Receiver, func: &'static str) -> Self {
        Self { id, channel, func, save: None }
    }

    pub fn begin_server(
        id: i32,
        channel: &'ch mut Receiver,
        func: &'static str,
        save: bool,
        proc_id: i32,
    ) -> Self {
        let save = if save {
            Some(proc_id.to_ne_bytes().to_vec())
        } else {
            None
        };
        Self { id, channel, func, save }
    }

    fn check_not_null<T>(&self, ptr: *const T, name: &'static str) {
        if ptr.is_null() {
            panic!("[#{} {}] {name} is null", self.id, self.func);
        }
    }

    pub unsafe fn mut_from<'a, T>(&self, ptr: *mut T, name: &'static str) -> &'a mut T {
        self.check_not_null(ptr, name);
        unsafe { &mut *ptr }
    }

    pub fn recv_mut<T>(&mut self, data: &mut T, name: &'static str)
    where
        T: Copy + Debug,
    {
        self.channel.recv_to(data).unwrap_or_else(bail_recv!(name));
        trace!(target: self.func, "[#{}] (recv) {name} = {data:?}", self.id);
        assert!(self.save.is_none());
    }

    pub unsafe fn mut_slice_from<'a, T>(
        &self,
        data: *mut T,
        len: usize,
        name: &'static str,
    ) -> &'a mut [T] {
        self.check_not_null(data, name);
        unsafe { slice::from_raw_parts_mut(data, len) }
    }

    pub fn recv_mut_slice<T>(&mut self, data: &mut [T], name: &'static str)
    where
        T: Copy,
    {
        self.channel.recv_slice_to(data).unwrap_or_else(bail_recv!(name));
        trace!(target: self.func, "[#{}] (recv) {name} = {data:p}[{}]", self.id, data.len());
        assert!(self.save.is_none());
    }

    pub fn recv<T>(&mut self, name: &'static str) -> T
    where
        T: Copy + Debug,
    {
        // TODO: extract from stream directly
        let data = self.channel.recv().unwrap_or_else(bail_recv!(name));
        trace!(target: self.func, "[#{}] (recv) {name} = {data:?}", self.id);
        if let Some(output) = &mut self.save {
            save(&data, output);
        }
        data
    }

    pub fn recv_slice<T>(&mut self, name: &'static str) -> Box<[T]>
    where
        T: Copy,
    {
        let data = self.channel.recv_slice().unwrap_or_else(bail_recv!(name));
        trace!(target: self.func, "[#{}] (recv) {name} = {data:p}[{}]", self.id, data.len());
        if let Some(output) = &mut self.save {
            save_slice(&data, output);
        }
        data
    }

    pub fn recv_cstr(&mut self, name: &'static str) -> CString {
        let bytes = self.channel.recv_slice().unwrap_or_else(bail_recv!(name));
        let result = CString::from_vec_with_nul(bytes.into_vec()).unwrap();
        trace!(target: self.func, "[#{}] (recv) {name} = {result:?}", self.id);
        if let Some(output) = &mut self.save {
            save_slice(result.as_bytes_with_nul(), output);
        }
        result
    }

    pub fn get_bytes(&mut self, mut dst: impl MemWrite) -> Result<(), CommChannelError> {
        assert!(self.save.is_none());
        self.channel.get_bytes(&mut dst)
    }

    pub fn finish(self) -> Option<Vec<u8>> {
        if let Receiver::Emulator(channel) = self.channel {
            let name = "timestamp";
            channel.recv_ts().unwrap_or_else(bail_recv!(name))
        }
        self.save
    }
}
