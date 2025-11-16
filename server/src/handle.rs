use std::collections::BTreeMap;

#[derive(Default)]
pub struct HandleManager(BTreeMap<usize, Handle>);

struct Handle(usize);

impl HandleManager {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn insert(&mut self, proxy: usize, value: usize) {
        self.0.insert(proxy, Handle(value));
    }

    pub fn get<T>(&mut self, proxy: *mut T, is_destroy: bool) -> *mut T {
        self.get_inner(proxy as usize, is_destroy) as *mut T
    }

    fn get_inner(&mut self, proxy: usize, is_destroy: bool) -> usize {
        if proxy == 0 {
            return 0;
        }
        if is_destroy {
            self.0.remove(&proxy).unwrap().0
        } else {
            self.0.get(&proxy).unwrap().0
        }
    }
}
