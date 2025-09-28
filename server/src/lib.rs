#![feature(maybe_uninit_slice)]
#![feature(write_all_vectored)]

mod dispatcher;
mod handle;

#[cfg(feature = "phos")]
mod phos;
#[cfg(feature = "phos")]
pub use phos::check_config;

use cudasys::cudart::{cudaError_t, cudaGetDeviceCount};
use dispatcher::dispatch;

#[cfg(feature = "rdma")]
use network::ringbufferchannel::RDMAChannel;

use network::ringbufferchannel::{EmulatorChannel, SHMChannel};
use network::{Channel, CommChannel, CommChannelError, NetworkConfig, Transportable, tcp};

use log::{error, info};

struct ServerWorker {
    pub id: i32,
    pub channel_sender: Channel,
    pub channel_receiver: Channel,
    pub resources: handle::HandleManager,
    opt_async_api: bool,
    opt_shadow_desc: bool,
}

impl ServerWorker {
    pub fn before_call(&self, func: &'static str) {
        log::debug!(target: func, "[#{}]", self.id);
    }
}

fn create_buffer(
    config: &NetworkConfig,
    id: i32,
    barrier: Option<std::sync::Arc<std::sync::Barrier>>,
) -> (Channel, Channel) {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the server side, the sender's name is stoc_channel_name,
    // receiver's name is ctos_channel_name.
    match config.comm_type.as_str() {
        "shm" => {
            let (receiver, sender) = SHMChannel::new_server_with_id(config, id).unwrap();
            barrier.unwrap().wait();
            if config.emulator {
                return (
                    Channel::new(Box::new(EmulatorChannel::new(sender, config))),
                    Channel::new(Box::new(EmulatorChannel::new(receiver, config))),
                );
            }
            (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
        }
        "tcp" => {
            let (receiver, sender) = tcp::new_server(config, id, &barrier.unwrap()).unwrap();
            (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
        }
        #[cfg(feature = "rdma")]
        "rdma" => {
            assert!(barrier.is_none());
            let (receiver, sender) = RDMAChannel::new_server(config, id);
            (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
        }
        &_ => panic!("Unsupported communication type in config"),
    }
}

fn receive_request<T: CommChannel>(channel_receiver: &mut T) -> Result<i32, CommChannelError> {
    let mut proc_id = 0;
    let () = proc_id.recv(channel_receiver)?;
    Ok(proc_id)
}

pub fn launch_server(
    config: &NetworkConfig,
    id: i32,
    client_pid: u32,
    barrier: Option<std::sync::Arc<std::sync::Barrier>>,
) {
    let (channel_sender, channel_receiver) = create_buffer(config, id, barrier);
    info!("[{}:{}] {} buffer created", std::file!(), std::line!(), config.comm_type);
    let mut max_devices = 0;
    if let cudaError_t::cudaSuccess =
        unsafe { cudaGetDeviceCount(&mut max_devices as *mut ::std::os::raw::c_int) }
    {
        info!("[{}:{}] found {} cuda devices", std::file!(), std::line!(), max_devices);
    } else {
        error!("[{}:{}] failed to find cuda devices", std::file!(), std::line!());
        panic!();
    }

    let mut server = ServerWorker {
        id,
        channel_sender,
        channel_receiver,
        resources: Default::default(),
        opt_async_api: config.opt_async_api,
        opt_shadow_desc: config.opt_shadow_desc,
    };

    #[cfg(feature = "phos")]
    {
        let flag_ptr = server.channel_receiver.flag_ptr().unwrap();
    }

    let mut state = HandleDemoState::Disabled;

    loop {
        match receive_request(&mut server.channel_receiver) {
            Ok(-1) => {
                break;
            }
            Ok(proc_id) => dispatch(proc_id, &mut server),
            #[cfg(feature = "phos")]
            Err(CommChannelError::ShmChannelLocked) => {
                assert!(matches!(
                    receive_request(&mut server.channel_receiver),
                    Err(CommChannelError::ShmChannelLocked),
                )); // assert no remaining async requests
                phos::checkpoint(); // TODO
                log::info!("PhOS checkpoint done");
                phos::clear_flag(&server.channel_receiver);
            }
            #[cfg(feature = "phos")]
            Err(CommChannelError::RestoreEof) => {
                state.finish_restore(&mut server);
            }
            Err(e) => {
                error!("failed to receive request: {e:?}");
                break;
            }
        }
        state.reset_and_restore(&mut server);
    }

    info!("server #{id} (client PID: {client_pid}) terminated");
}

enum HandleDemoState {
    Disabled,
    Counting(u32),
    Restoring { channel_sender: Channel, channel_receiver: Channel },
}

impl HandleDemoState {
    fn reset_and_restore(&mut self, server: &mut ServerWorker) {
        if server.resources.len() <= 3 {
            return;
        }
        match self {
            HandleDemoState::Counting(50..) => {
                log::info!("{}", server.resources.len());
                let mut args = Vec::new();
                log::info!("checkpointing handles...");
                server.resources.serialize(&mut args).unwrap();
                log::info!("resetting all handles...");
                std::mem::take(&mut server.resources);
                let restore_vec = network::restore::RestoreVec::new(args);
                log::info!("start restoring...");
                let channel_sender = std::mem::replace(
                    &mut server.channel_sender,
                    Channel::new(Box::new(network::restore::BlackHole)),
                );
                let channel_receiver = std::mem::replace(
                    &mut server.channel_receiver,
                    Channel::new(Box::new(restore_vec)),
                );
                *self = HandleDemoState::Restoring { channel_sender, channel_receiver };
            }
            HandleDemoState::Counting(n) => {
                *n += 1;
            }
            _ => {}
        }
    }

    fn finish_restore(&mut self, server: &mut ServerWorker) {
        if let HandleDemoState::Restoring { .. } = self {
            let HandleDemoState::Restoring { channel_sender, channel_receiver } =
                std::mem::replace(self, HandleDemoState::Counting(0))
            else {
                unreachable!()
            };
            server.channel_sender = channel_sender;
            server.channel_receiver = channel_receiver;
            log::info!("restoring done");
        }
    }
}
