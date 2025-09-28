use std::time::UNIX_EPOCH;

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct NsTimestamp {
    pub sec_timestamp: i64,
    pub ns_timestamp: u32,
}

impl NsTimestamp {
    pub fn new() -> NsTimestamp {
        Self::default()
    }
    pub fn now() -> NsTimestamp {
        let now_time = std::time::SystemTime::now();
        let duration_since_epoch = now_time.duration_since(UNIX_EPOCH).expect("Time went backwards");
        let sec = duration_since_epoch.as_secs() as i64;
        let ns = duration_since_epoch.subsec_nanos(); 
        NsTimestamp {
            sec_timestamp: sec,
            ns_timestamp: ns,
        }
    }
}
