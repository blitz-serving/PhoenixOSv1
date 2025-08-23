use std::fs::File;
use std::io::{self, Write};

pub const MEASURE_START: usize = 0;
pub const MEASURE_CSER: usize = 1;
pub const MEASURE_CSEND: usize = 2;
pub const MEASURE_SRECV: usize = 3;
pub const MEASURE_SDSER: usize = 4;
pub const MEASURE_RAW: usize = 5;
pub const MEASURE_SSER: usize = 6;
pub const MEASURE_SSEND: usize = 7;
pub const MEASURE_CRECV: usize = 8;
pub const MEASURE_CDSER: usize = 9;
pub const MEASURE_TOTAL: usize = 10;
const MEASURE_MAX_NUM: usize = 11;

const CLOCK_FREQUENCY: f64 = 2.2;
const ITER_NUM: usize = 10010;

pub struct Timer {
    start_time: Box<[[u64; MEASURE_MAX_NUM]; ITER_NUM]>,
    stop_time:  Box<[[u64; MEASURE_MAX_NUM]; ITER_NUM]>,
    cnt: usize,
    output_file: Option<String>,
}

#[inline]
pub fn rdtscp() -> u64 {
    let mut aux = 0;
    unsafe { core::arch::x86_64::__rdtscp(&raw mut aux) }
}

#[inline]
pub fn clock2ns(clock: u64) -> f64 {
    clock as f64 / CLOCK_FREQUENCY
}

impl Timer {
    
    pub fn new(file_name: String) -> Self {
        let start_time = vec![[0u64; MEASURE_MAX_NUM]; ITER_NUM]
            .into_boxed_slice()
            .try_into()
            .unwrap(); // This converts Box<[T]> to Box<[T; N]>
        let stop_time = vec![[0u64; MEASURE_MAX_NUM]; ITER_NUM]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        Self { start_time, stop_time, cnt: 0, output_file: Some(file_name) }
    }

    pub fn new_null() -> Self {
        let start_time = vec![[0u64; MEASURE_MAX_NUM]; ITER_NUM]
            .into_boxed_slice()
            .try_into()
            .unwrap(); // This converts Box<[T]> to Box<[T; N]>
        let stop_time = vec![[0u64; MEASURE_MAX_NUM]; ITER_NUM]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        Self { start_time, stop_time, cnt: 0, output_file: None }
    }

    #[inline]
    pub fn set(&mut self, id: usize) {
        self.start_time[self.cnt][id] = rdtscp();
    }

    #[inline]
    pub fn stop(&mut self, id: usize) {
        self.stop_time[self.cnt][id] = rdtscp();
    }

    #[inline]
    pub fn get_time(&self, cnt: usize, id: usize) -> u64 {
        self.stop_time[cnt][id] - self.start_time[cnt][id]
    }

    #[inline]
    pub fn plus_cnt(&mut self) {
        self.cnt += 1;
        if self.cnt == ITER_NUM {
            let _ = self.write();
        }
    }

    pub fn write(&self) -> io::Result<()> {
        if let Some(ref file_name) = self.output_file {
            let mut file = File::create(file_name)?;
            for i in 0..ITER_NUM {
                let mut row_result = Vec::new();
                for j in 0..MEASURE_MAX_NUM {
                    row_result.push(self.start_time[i][j].to_string());
                }
                writeln!(file, "{}", &row_result.join(", "))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_test() {
        let mut timer = Timer::new_null();

        timer.set(MEASURE_TOTAL);
        let mut _sum: u64 = 0;
        for i in 0..10000 {
            _sum += i;
        }
        timer.stop(MEASURE_TOTAL);

        let t = timer.get_time(0, MEASURE_TOTAL);
        assert!(t > 0);

        // round 2
        timer.plus_cnt();

        timer.set(MEASURE_TOTAL);                
        let mut _sum: u64 = 0;
        for i in 0..10000 {
            _sum += i;
        }
        timer.stop(MEASURE_TOTAL);        
        let t2 = timer.get_time(1, MEASURE_TOTAL);        
        assert!(t2 > 0);
        let tt = timer.get_time(0, MEASURE_TOTAL);
        assert_eq!(t,tt);
    }
}
