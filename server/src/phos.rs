use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::Ordering;

use libc::pid_t;
use network::ringbufferchannel::SHM_FLAG_WRITE_DISABLED;
use network::type_impl::recv_slice;
use network::Channel;

#[link(name = "pos")]
extern "C" {
    fn pos_create_workspace_cuda() -> *mut c_void;
    fn pos_process(
        pos_cuda_ws: *mut c_void,
        api_id: u64,
        uuid: u64,
        param_desps: *mut u64,
        param_num: i32,
    ) -> i32;
    fn pos_destory_workspace_cuda(pos_cuda_ws: *mut c_void) -> i32;
    fn pos_remoting_stop_query(pos_cuda_ws: *mut c_void, uuid: u64) -> i32;
    fn pos_remoting_stop_confirm(pos_cuda_ws: *mut c_void, uuid: u64) -> i32;
    fn pos_remoting_create_client(
        pos_cuda_ws: *mut c_void,
        job_name: *const c_char,
        pid: pid_t,
    ) -> u64;
    fn pos_remoting_set_flag_ptr(pos_cuda_ws: *mut c_void, uuid: u64, flag: *mut u64) -> i32;
}

#[expect(non_camel_case_types)]
pub struct POSWorkspace_CUDA(*mut c_void);

impl POSWorkspace_CUDA {
    pub fn new() -> Self {
        log::info!("Starting PhOS server ...");
        let pos_cuda_ws = unsafe { pos_create_workspace_cuda() };
        assert!(!pos_cuda_ws.is_null());
        log::info!("PhOS daemon is running. You can run a program like \"env $phos python3 train.py \" now");
        Self(pos_cuda_ws)
    }

    pub fn create_client(&self, job_name: &CStr, pid: pid_t) -> u64 {
        let uuid = unsafe { pos_remoting_create_client(self.0, job_name.as_ptr(), pid) };
        assert_ne!(u64::MAX, uuid);
        uuid
    }

    pub fn set_flag_ptr(&self, uuid: u64, flag: *mut u64) {
        assert_eq!(0, unsafe { pos_remoting_set_flag_ptr(self.0, uuid, flag) })
    }

    #[cfg(target_pointer_width = "64")]
    pub fn pos_process(&self, api_id: i32, uuid: u64, param_desps: &[usize]) -> i32 {
        unsafe {
            pos_process(
                self.0,
                api_id as u64,
                uuid,
                param_desps.as_ptr() as *mut u64,
                (param_desps.len() / 2) as i32,
            )
        }
    }

    pub fn stop_and_block(&self, uuid: u64) {
        log::info!("Confirm checkpoint for PhOS client {uuid}");
        assert_eq!(0, unsafe { pos_remoting_stop_confirm(self.0, uuid) });
        loop {
            match unsafe { pos_remoting_stop_query(self.0, uuid) } {
                0 => return,
                1 => panic!("just confirmed"),
                2 => std::hint::spin_loop(),
                _ => panic!("unknown value"),
            }
        }
    }
}

impl Drop for POSWorkspace_CUDA {
    fn drop(&mut self) {
        unsafe { pos_destory_workspace_cuda(self.0) };
    }
}

pub fn recv_job_name(channel_receiver: &Channel) -> CString {
    let bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
    let result = CString::new(bytes).unwrap();
    log::info!("job_name: {result:?}");
    result
}

pub fn clear_flag(channel_receiver: &Channel) {
    let flag = channel_receiver.flag().unwrap();
    flag.fetch_and(!SHM_FLAG_WRITE_DISABLED, Ordering::SeqCst);
}
