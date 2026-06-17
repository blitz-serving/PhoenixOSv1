use network::ringbufferchannel::rdma::{RDMAChannel, RdmaConfig};
use network::{RecvChannel, SendChannel};

const BUF_SIZE: usize = 1024 + network::ringbufferchannel::META_AREA;
const PORT: u8 = 1;

#[test]
fn rdma_channel_buffer_manager() {
    let config = RdmaConfig {
        handshake_socket: "127.0.0.1:8001",
        device_name: "mlx5_1",
        device_port: PORT,
        stoc_channel_name: "/stoc",
        ctos_channel_name: "/ctos",
        buf_size: BUF_SIZE,
    };

    // First, new a RDMA server to listen at a socket address (s_sender_addr).
    // Then new a client with server's socket address to handshake with it.
    // The client side will use the server name (s_sender_name) to get its
    // remote info like raddr and rkey.

    let (mut recver, mut sender) = std::thread::scope(|scope| {
        let s_handler = scope.spawn(|| match RDMAChannel::new_server_inner(&config, 0) {
            (server, _) => {
                println!("Server created successfully");
                server
            }
        });

        let c_handler = scope.spawn(|| match RDMAChannel::new_client_inner(&config, 0) {
            (client, _) => {
                println!("Client created successfully");
                client
            }
        });

        (s_handler.join().unwrap(), c_handler.join().unwrap())
    });

    const SZ: usize = 256;
    let data = [48_u8; SZ];
    sender.put_bytes(&data).unwrap();

    let data2 = [97_u8; SZ];
    sender.put_bytes(&data2).unwrap();

    sender.flush().unwrap();

    let mut buffer = [0u8; 2 * SZ];
    let size = buffer.len();
    match recver.get_bytes(&mut buffer) {
        Ok(()) => {
            for i in 0..SZ {
                assert_eq!(buffer[i], 48);
            }
            for i in SZ..size {
                assert_eq!(buffer[i], 97);
            }
        }
        Err(_) => todo!(),
    }
}
