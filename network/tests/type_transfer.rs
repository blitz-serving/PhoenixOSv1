use std::sync::{Arc, Barrier};
use std::thread;

use cudasys::cudart::cudaError_t;
use network::ringbufferchannel::SHMChannel;
use network::{RecvChannel, SendChannel};

const VALUES: [cudaError_t; 10] = [
    cudaError_t::cudaSuccess,
    cudaError_t::cudaErrorInvalidValue,
    cudaError_t::cudaErrorMemoryAllocation,
    cudaError_t::cudaErrorInitializationError,
    cudaError_t::cudaErrorCudartUnloading,
    cudaError_t::cudaErrorProfilerDisabled,
    cudaError_t::cudaErrorProfilerNotInitialized,
    cudaError_t::cudaErrorProfilerAlreadyStarted,
    cudaError_t::cudaErrorProfilerAlreadyStopped,
    cudaError_t::cudaErrorInvalidConfiguration,
];

#[test]
fn test_cudaerror() {
    let shm_name = "/stoc";
    let shm_len = 1024;

    let mut consumer_channel = SHMChannel::new_server(shm_name, shm_len).unwrap();
    let mut producer_channel = SHMChannel::new_client(shm_name, shm_len).unwrap();

    let barrier = Arc::new(Barrier::new(2)); // Set up a barrier for 2 threads
    let producer_barrier = barrier.clone();
    let consumer_barrier = barrier.clone();

    let test_iters = 1000;

    // Producer thread
    let producer = thread::spawn(move || {
        producer_barrier.wait(); // Wait for both threads to be ready

        for i in 0..test_iters {
            let var = VALUES[i % VALUES.len()];
            producer_channel.send(&var).unwrap();
            producer_channel.flush().unwrap();
        }

        println!("Producer done");
    });

    // Consumer thread
    let consumer = thread::spawn(move || {
        consumer_barrier.wait(); // Wait for both threads to be ready

        let mut received = 0;

        while received < test_iters {
            let test = VALUES[received % VALUES.len()];
            let mut var = cudaError_t::cudaSuccess;
            consumer_channel.recv_to(&mut var).unwrap();
            assert_eq!(var, test);
            received += 1;
        }
    });

    // Note: producer must be joined later, since the consumer will reuse the buffer
    consumer.join().unwrap();
    producer.join().unwrap();
}
