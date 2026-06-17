fn main() {
    #[cfg(target_os = "linux")]
    {
        use network::RecvChannel as _;
        use network::ringbufferchannel::SHMChannel;

        let shm_name = "/stoc";
        let shm_len = 1024;
        let mut channel = SHMChannel::new_server(shm_name, shm_len).unwrap();

        loop {
            let mut dst = [0u8; 5];
            let res = channel.recv_slice_to(&mut dst);
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
