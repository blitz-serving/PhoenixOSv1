fn main() {
    #[cfg(target_os = "linux")]
    {
        use network::{ringbufferchannel::SHMChannel, Channel, type_impl::send_slice};
        use std::boxed::Box;

        let shm_name = "/stoc";
        let shm_len = 1024;
        let mut channel = Channel::new(Box::new(SHMChannel::new_client(shm_name, shm_len).unwrap()));
        let buf = [1, 2, 3, 4, 5];
        send_slice(&buf, &mut channel).unwrap();

        println!("send done");
    }

    println!("SHM client only works on linux");
}
