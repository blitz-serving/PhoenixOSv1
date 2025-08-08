#![feature(thread_local)]

mod dl;
mod elf;
#[cfg(not(feature = "passthrough"))]
mod hijack;
#[cfg(feature = "passthrough")]
mod passthrough;
#[cfg(feature = "phos")]
mod phos;

#[small_ctor::ctor]
unsafe fn init() {
    // core_affinity::set_for_current(1);
    let env = env_logger::Env::new().default_filter_or("info");
    env_logger::Builder::from_env(env).format_timestamp(None).init();
}
