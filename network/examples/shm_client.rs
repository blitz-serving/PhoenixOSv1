fn main() {
    #[cfg(target_os = "linux")]
    {
        use network::SendChannel;
        use network::ringbufferchannel::SHMChannel;

        let shm_name = "/stoc";
        let shm_len = 1024;
        let mut channel = SHMChannel::new_client(shm_name, shm_len).unwrap();
        let buf = [1, 2, 3, 4, 5];
        channel.send_slice(&buf).unwrap();

        println!("send done");
    }

    println!("SHM client only works on linux");
}
