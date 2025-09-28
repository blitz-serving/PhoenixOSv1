fn main() {
    #[cfg(target_os = "linux")]
    {
        use network::{ringbufferchannel::SHMChannel, Channel, type_impl::recv_slice_to};
        use std::boxed::Box;

        let shm_name = "/stoc";
        let shm_len = 1024;
        let mut channel = Channel::new(Box::new(SHMChannel::new_server(shm_name, shm_len).unwrap()));

        loop {
            let mut dst = [0u8; 5];
            let res = recv_slice_to(&mut dst, &mut channel);
            match res {
                Ok(()) => {
                    println!("Received {:?}", dst);
                    break;
                }
                Err(e) => {
                    println!("Error {}", e);
                    assert!(false);
                }
            }
        }
    }

    println!("SHM server only works on linux");
}
