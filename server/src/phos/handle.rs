use std::collections::BTreeMap;
use std::io::{self, IoSlice, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
pub struct HandleManager(BTreeMap<usize, Handle>);

impl HandleManager {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn insert(&mut self, proxy: usize, value: usize) {
        self.0.insert(proxy, Handle { value, payloads: Vec::new() });
    }

    pub fn insert_args(&mut self, proxy: usize, args: Vec<u8>) {
        static TIMESTAMP: AtomicUsize = AtomicUsize::new(1);

        self.0.get_mut(&proxy).unwrap().payloads.push(Payload {
            timestamp: TIMESTAMP.fetch_add(1, Ordering::Relaxed),
            args: args.into_boxed_slice(),
        });
    }

    pub fn get<T>(&mut self, proxy: *mut T, is_destroy: bool) -> *mut T {
        self.get_inner(proxy as usize, is_destroy) as *mut T
    }

    fn get_inner(&mut self, proxy: usize, is_destroy: bool) -> usize {
        if proxy == 0 {
            return 0;
        }
        if is_destroy {
            self.0.remove(&proxy).unwrap().value
        } else {
            self.0.get(&proxy).unwrap().value
        }
    }

    pub fn serialize(&self, output: &mut impl Write) -> io::Result<()> {
        let mut payloads: Vec<_> =
            self.0.values().flat_map(|handle| handle.payloads.iter()).collect();
        payloads.sort_unstable_by_key(|payload| payload.timestamp);
        let mut bufs: Vec<_> =
            payloads.into_iter().map(|payload| IoSlice::new(&payload.args)).collect();
        output.write_all_vectored(bufs.as_mut_slice())
    }
}

struct Handle {
    value: usize,
    payloads: Vec<Payload>,
}

struct Payload {
    timestamp: usize,
    args: Box<[u8]>,
}
