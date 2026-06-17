use std::cell::UnsafeCell;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use network::ringbufferchannel::LocalChannel;
use network::{RecvChannel, SendChannel};

pub struct Channel(UnsafeCell<LocalChannel>);

impl Channel {
    unsafe fn get(&self) -> &mut LocalChannel {
        unsafe { &mut *self.0.get() }
    }
}

unsafe impl Send for Channel {}
unsafe impl Sync for Channel {}

#[test]
fn test_ring_buffer_producer_consumer() {
    let channel = Arc::new(Channel(UnsafeCell::new(LocalChannel::new(
        1024 + network::ringbufferchannel::META_AREA,
    ))));
    let producer_channel = Arc::clone(&channel);
    let consumer_channel = Arc::clone(&channel);

    let barrier = Arc::new(Barrier::new(2)); // Set up a barrier for 2 threads
    let producer_barrier = barrier.clone();
    let consumer_barrier = barrier.clone();

    let test_iters = 1000;

    // Producer thread
    let producer = thread::spawn(move || {
        producer_barrier.wait(); // Wait for both threads to be ready

        for i in 0..test_iters {
            let data = [(i % 256) as u8; 10]; // Simplified data to send
            let producer_channel = unsafe { producer_channel.get() };
            producer_channel.put_bytes(&data).unwrap();
        }

        println!("Producer done");
    });

    // Consumer thread
    let consumer = thread::spawn(move || {
        consumer_barrier.wait(); // Wait for both threads to be ready

        let mut received = 0;
        let mut buffer = [0u8; 10];

        while received < test_iters {
            let len = buffer.len();
            let consumer_channel = unsafe { consumer_channel.get() };
            match consumer_channel.get_bytes(&mut buffer) {
                Ok(()) => {
                    for i in 0..len {
                        assert_eq!(buffer[i], (received % 256) as u8);
                    }

                    received += 1;
                }
                Err(_) => thread::sleep(Duration::from_millis(10)), // Wait if buffer is empty
            }
        }
    });

    // Note: producer must be joined later, since the consumer will reuse the buffer
    consumer.join().unwrap();
    producer.join().unwrap();
}
