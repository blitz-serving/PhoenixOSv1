#[cfg(test)]
mod tests {
    use crate::ringbufferchannel::{LocalChannel, META_AREA};
    use crate::{RecvChannel, SendChannel};

    #[test]
    fn basic_send_receive() {
        let mut channel = LocalChannel::new(10 + META_AREA);
        let data_to_send: [u8; 5] = [1, 2, 3, 4, 5];
        let mut receive_buffer = [0u8; 5];

        channel.put_bytes(&data_to_send).unwrap();

        channel.get_bytes(&mut receive_buffer).unwrap();

        assert_eq!(receive_buffer, data_to_send);
    }

    #[test]
    fn partial_receive() {
        let mut channel = LocalChannel::new(10 + META_AREA);
        let data_to_send: [u8; 5] = [1, 2, 3, 4, 5];
        let mut receive_buffer = [0u8; 3];

        channel.put_bytes(&data_to_send).unwrap();

        channel.get_bytes(&mut receive_buffer).unwrap();

        assert_eq!(receive_buffer, [1, 2, 3]);
    }

    #[test]
    fn wrap_around() {
        let mut channel = LocalChannel::new(10 + META_AREA);

        let first_send: [u8; 3] = [1, 2, 3];
        let second_send: [u8; 3] = [4, 5, 6];
        let mut first_recv = [0u8; 3];
        let mut second_recv = [0u8; 3];

        channel.put_bytes(&first_send).unwrap();

        channel.get_bytes(&mut first_recv).unwrap(); // Create a gap for wrap-around

        channel.put_bytes(&second_send).unwrap();

        channel.get_bytes(&mut second_recv).unwrap();

        assert_eq!(first_recv, [1, 2, 3]);
        assert_eq!(second_recv, [4, 5, 6]);
    }

    // TBD
}
