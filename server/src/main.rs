#![feature(cast_maybe_uninit)]
#![cfg_attr(feature = "phos", feature(current_thread_id))]
#![cfg_attr(feature = "phos", feature(thread_id_value))]
#![cfg_attr(feature = "phos", feature(write_all_vectored))]

mod control;
mod daemon;
mod dispatcher;
#[cfg(not(feature = "phos"))]
mod handle;
#[cfg(feature = "phos")]
mod phos;
mod worker;

use network::config::Config;
pub use worker::ServerThread;

fn main() {
    let env = env_logger::Env::new().default_filter_or("info");
    env_logger::Builder::from_env(env)
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .init();
    // core_affinity::set_for_current(0);

    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    #[cfg(feature = "phos")]
    if runtime.block_on(phos::cli::run()) {
        return;
    }

    let config = Config::read_env();
    #[cfg(feature = "phos")]
    phos::check_config(&config.common);
    log::info!("Using {:?}", config.common.checked_comm_type());
    let worker_socket =
        std::env::var_os(worker::SERVER_WORKER_SOCKET_ENV).filter(|v| !v.is_empty());
    if let Some(socket) = worker_socket {
        runtime.block_on(worker::server_process(config, socket));
    } else {
        runtime.block_on(daemon::daemon(config));
    }
}
