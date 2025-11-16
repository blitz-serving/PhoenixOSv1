use network::Channel;

use crate::ServerWorker;

pub enum HandleDemoState {
    Disabled,
    Counting(u32),
    Restoring { channel_sender: Channel, channel_receiver: Channel },
}

impl HandleDemoState {
    pub fn reset_and_restore(&mut self, server: &mut ServerWorker) {
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

    pub fn finish_restore(&mut self, server: &mut ServerWorker) {
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
