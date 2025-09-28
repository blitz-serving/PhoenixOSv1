use std::sync::atomic::{AtomicUsize, Ordering};

pub fn next_handle() -> usize {
    // TODO: reset when support forking
    static VALUE: AtomicUsize = AtomicUsize::new(1);
    VALUE.fetch_add(1, Ordering::Relaxed)
}
