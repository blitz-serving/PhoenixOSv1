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
        self.0.insert(proxy, Handle { value, payloads: Payloads::Empty });
    }

    pub fn insert_args(&mut self, proxy: usize, key: Option<u64>, args: Vec<u8>) {
        static TIMESTAMP: AtomicUsize = AtomicUsize::new(1);

        let payload = Payload {
            timestamp: TIMESTAMP.fetch_add(1, Ordering::Relaxed),
            args: args.into_boxed_slice(),
        };
        self.0.get_mut(&proxy).unwrap().payloads.insert(key, payload);
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
            match self.0.get(&proxy) {
                Some(handle) => handle.value,
                None => panic!("invalid proxy handle: {proxy:#x}"),
            }
        }
    }

    pub fn serialize(&self, output: &mut impl Write) -> io::Result<()> {
        let mut payloads = Vec::new();
        for handle in self.0.values() {
            handle.payloads.extend_refs(&mut payloads);
        }
        payloads.sort_unstable_by_key(|payload| payload.timestamp);
        let mut bufs: Vec<_> =
            payloads.into_iter().map(|payload| IoSlice::new(&payload.args)).collect();
        output.write_all_vectored(bufs.as_mut_slice())
    }
}

struct Handle {
    value: usize,
    payloads: Payloads,
}

enum Payloads {
    Empty,
    Vec(Vec<Payload>),
    Map(BTreeMap<u64, Payload>),
}

impl Payloads {
    fn insert(&mut self, key: Option<u64>, payload: Payload) {
        match key {
            None => match self {
                Payloads::Empty => *self = Payloads::Vec(vec![payload]),
                Payloads::Vec(payloads) => payloads.push(payload),
                Payloads::Map(_) => panic!("expected unkeyed payload"),
            },
            Some(key) => match self {
                Payloads::Empty => {
                    *self = Payloads::Map(BTreeMap::from([(key, payload)]));
                }
                Payloads::Vec(_) => {
                    panic!("expected keyed payload");
                }
                Payloads::Map(payloads) => {
                    payloads.insert(key, payload);
                }
            },
        }
    }

    fn extend_refs<'a>(&'a self, refs: &mut Vec<&'a Payload>) {
        match self {
            Payloads::Empty => {}
            Payloads::Vec(payloads) => refs.extend(payloads.iter()),
            Payloads::Map(payloads) => refs.extend(payloads.values()),
        }
    }
}

struct Payload {
    timestamp: usize,
    args: Box<[u8]>,
}
