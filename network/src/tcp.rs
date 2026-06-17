use std::io::{self, BufReader, BufWriter, Read as _, Write as _};
use std::net::{SocketAddr, TcpListener, TcpStream};

use crate::config::NetworkConfig;
use crate::{CommChannelError, RecvChannel, SendChannel};

pub fn new_listener(config: &NetworkConfig, id: i32) -> io::Result<TcpListener> {
    let mut addr: SocketAddr = config.handshake_socket.parse().unwrap();
    addr.set_port(addr.port() + id as u16);
    TcpListener::bind(addr)
}

pub fn new_server(listener: TcpListener) -> io::Result<(TcpReceiver, TcpSender)> {
    let (stream, _) = listener.accept()?;
    let receiver = TcpReceiver(BufReader::new(stream.try_clone()?));
    let sender = TcpSender(BufWriter::new(stream));
    Ok((receiver, sender))
}

pub fn new_client(config: &NetworkConfig, id: i32) -> io::Result<(TcpSender, TcpReceiver)> {
    let mut addr: SocketAddr = config.handshake_socket.parse().unwrap();
    addr.set_port(addr.port() + id as u16);
    let stream = TcpStream::connect(addr)?;
    let sender = TcpSender(BufWriter::new(stream.try_clone()?));
    let receiver = TcpReceiver(BufReader::new(stream));
    Ok((sender, receiver))
}

pub struct TcpSender(BufWriter<TcpStream>);
pub struct TcpReceiver(BufReader<TcpStream>);

impl SendChannel for TcpSender {
    fn flush(&mut self) -> Result<(), CommChannelError> {
        self.0.flush().map_err(|e| {
            log::error!("flush failed: {e}");
            CommChannelError::IoError
        })
    }

    fn put_bytes(&mut self, src: &[u8]) -> Result<(), CommChannelError> {
        self.0.write_all(src).map_err(|e| {
            log::error!("write failed: {e}");
            CommChannelError::IoError
        })
    }
}

impl RecvChannel for TcpReceiver {
    fn get_bytes(&mut self, dst: &mut [u8]) -> Result<(), CommChannelError> {
        self.0.read_exact(dst).map_err(|e| {
            log::error!("read failed: {e}");
            CommChannelError::IoError
        })
    }
}
