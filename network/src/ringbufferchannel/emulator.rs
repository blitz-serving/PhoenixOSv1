use super::SHMChannel;
use super::types::NsTimestamp;
use crate::config::CommonConfig;
use crate::{CommChannelError, CommChannelInnerIO, MemRead, MemWrite, RecvChannel, SendChannel};

pub struct EmulatorChannel {
    manager: SHMChannel,
    byte_cnt: usize,
    last_timestamp: NsTimestamp,
    rtt: f64,
    bandwidth: f64,
    start: Option<u64>,
    // begin: NsTimestamp,
}

impl EmulatorChannel {
    pub fn new(manager: SHMChannel, config: &CommonConfig) -> Self {
        // let now = NsTimestamp::now();
        // log::info!("{}:{}", now.sec_timestamp, now.ns_timestamp);
        Self {
            manager,
            byte_cnt: 0,
            last_timestamp: NsTimestamp::new(),
            rtt: config.rtt,
            bandwidth: config.bandwidth,
            start: None,
            // begin: now,
        }
    }

    pub fn inner(&self) -> &SHMChannel {
        &self.manager
    }

    fn calculate_latency(&self, current_bytes: usize) -> f64 {
        let data_size =
            current_bytes + std::mem::size_of::<NsTimestamp>() + std::mem::size_of::<i32>();
        self.rtt * 1000000.0 / 2.0 + (data_size as f64 * 8.0 / self.bandwidth) * 1000000000.0
    }

    pub fn calculate_ts(&mut self, current_bytes: usize) -> NsTimestamp {
        let latency = self.calculate_latency(current_bytes);
        let now_timestamp = NsTimestamp::now();
        let base_timestamp = match now_timestamp > self.get_last_timestamp() {
            true => now_timestamp,
            false => self.get_last_timestamp(),
        };
        let sec = base_timestamp.sec_timestamp
            + (base_timestamp.ns_timestamp as i64 + latency as i64) / 1000000000;
        let ns = (base_timestamp.ns_timestamp + latency as u32) % 1000000000;
        NsTimestamp { sec_timestamp: sec, ns_timestamp: ns }
    }

    #[inline]
    pub fn get_byte_cnt(&self) -> usize {
        self.byte_cnt
    }

    #[inline]
    pub fn set_byte_cnt(&mut self, byte_cnt: usize) {
        self.byte_cnt = byte_cnt;
    }

    #[inline]
    pub fn get_last_timestamp(&self) -> NsTimestamp {
        self.last_timestamp
    }

    #[inline]
    pub fn set_last_timestamp(&mut self, last_timestamp: NsTimestamp) {
        self.last_timestamp = last_timestamp;
    }

    #[inline]
    pub fn get_start(&self) -> Option<u64> {
        self.start
    }

    #[inline]
    pub fn set_start(&mut self, start: Option<u64>) {
        self.start = start;
    }

    pub fn put_bytes(&mut self, src: &mut impl MemRead) -> Result<(), CommChannelError> {
        #[cfg(feature = "log_rperf")]
        if self.get_start() == None {
            self.set_start(Some(measure::rdtscp()));
            // let now = NsTimestamp::now();
            // let elapsed = (now.sec_timestamp - self.begin.sec_timestamp) * 1000000000
            //     + (now.ns_timestamp as i32 - self.begin.ns_timestamp as i32) as i64;
            // log::info!(", {}", elapsed);
        }
        self.set_byte_cnt(self.get_byte_cnt() + src.remaining());
        CommChannelInnerIO::put_bytes(&self.manager, src)
    }

    pub fn get_bytes(&mut self, dst: &mut impl MemWrite) -> Result<(), CommChannelError> {
        CommChannelInnerIO::get_bytes(&self.manager, dst)
    }
}

impl SendChannel for EmulatorChannel {
    fn flush(&mut self) -> Result<(), CommChannelError> {
        SendChannel::flush(&mut self.manager)
    }

    fn put_bytes(&mut self, mut src: &[u8]) -> Result<(), CommChannelError> {
        self.put_bytes(&mut src)
    }
}

impl RecvChannel for EmulatorChannel {
    delegate::delegate! {
        to &mut self.manager {
            #[through(RecvChannel)]
            fn get_bytes(&mut self, dst: &mut [u8]) -> Result<(), CommChannelError>;
            #[through(RecvChannel)]
            fn recv<T: Copy>(&mut self) -> Result<T, CommChannelError>;
            #[through(RecvChannel)]
            fn recv_to<T: Copy>(&mut self, dst: &mut T) -> Result<(), CommChannelError>;
            #[through(RecvChannel)]
            fn recv_slice<T: Copy>(&mut self) -> Result<Box<[T]>, CommChannelError>;
            #[through(RecvChannel)]
            fn recv_slice_to<T: Copy>(&mut self, dst: &mut [T]) -> Result<(), CommChannelError>;
        }
    }
}

impl EmulatorChannel {
    pub fn send_ts(&mut self) -> Result<(), CommChannelError> {
        #[cfg(feature = "log_rperf")]
        {
            if self.get_start() == None {
                self.set_start(Some(measure::rdtscp()));
            }
            let end = measure::rdtscp();
            let elapsed = measure::clock2ns(end - self.get_start().unwrap());
            log::info!(", {}", elapsed / 1000.0);
            let byte_cnt = self.get_byte_cnt();
            log::info!(", {}", byte_cnt);
            self.set_start(None);
        }
        let ts = self.calculate_ts(self.get_byte_cnt());
        self.manager.send(&ts)?;
        self.set_byte_cnt(0);
        self.set_last_timestamp(ts);
        Ok(())
    }

    pub fn recv_ts(&mut self) -> Result<(), CommChannelError> {
        let timestamp: NsTimestamp = self.recv()?;
        while NsTimestamp::now() < timestamp {
            // Busy-waiting
        }
        // let start = NsTimestamp::now();
        // log::info!("gpu_issue, {}:{}", start.sec_timestamp, start.ns_timestamp as f64 / 1000.0);
        Ok(())
    }
}
