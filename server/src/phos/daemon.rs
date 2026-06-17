use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use futures::future::join_all;
use network::config::Config;
use network::oob::{CheckpointMode, CheckpointRequest, NewClientThreadRequest, Oob};
use tarpc::context::Context;

use super::call_criu_restore;
use crate::control::RestoreProcessRequest;
use crate::daemon::{Daemon, DaemonInner, StandbyWorker};

#[derive(Clone, Copy)]
pub struct PhosDaemon<'d>(pub &'d Daemon);

const CONCURRENT: bool = true;

impl<'d> Oob for PhosDaemon<'d> {
    async fn new_client_thread(self, context: Context, request: NewClientThreadRequest) -> i32 {
        self.0.new_client_thread(context, request).await
    }

    async fn checkpoint_job(self, context: Context, request: CheckpointRequest) {
        let job_name = request.job_name.clone();
        let ckpt_dir = request.ckpt_dir.clone();
        write_metadata(&ckpt_dir, &self.0.config, &job_name);

        let controls = self.0.inner.lock().await.controls_for_job(&job_name);
        if CONCURRENT {
            join_all(controls.into_iter().map(|control| {
                let request = request.clone();
                async move { control.checkpoint_process(context, request).await }
            }))
            .await
            .into_iter()
            .for_each(Result::unwrap);
        } else {
            for control in controls {
                control.checkpoint_process(context, request.clone()).await.unwrap();
            }
        }
    }

    async fn restore_job(self, context: Context, ckpt_dir: String, cuda_visible_devices: String) {
        let info = RestoreInfo::scan(&ckpt_dir);
        self.0.config.assert_same_server(&info.config);

        let mut inner = self.0.inner.lock().await;
        inner.reserve_ids(info.min_id, info.max_id);

        let mut restores = Vec::with_capacity(info.pid_to_ids.len());
        for (client_pid, ids) in info.pid_to_ids {
            let request = RestoreProcessRequest {
                job_name: info.job_name.clone(),
                ckpt_dir: ckpt_dir.clone(),
                client_pid,
                ids: ids.clone(),
            };
            let child = inner
                .get_or_spawn_child(
                    client_pid,
                    info.job_name.clone(),
                    cuda_visible_devices.clone(),
                    &self.0.config,
                )
                .await;
            let control = child
                .control
                .as_ref()
                .map(|(_, control)| control.clone())
                .expect("restore target is detached");
            child.ids = ids;
            restores.push((control, request));
        }

        if CONCURRENT {
            join_all(restores.into_iter().map(|(control, request)| async move {
                control.restore_process(context, request).await
            }))
            .await
            .into_iter()
            .for_each(Result::unwrap);
        } else {
            for (control, request) in restores {
                control.restore_process(context, request).await.unwrap();
            }
        }

        log::info!("calling criu restore for job {}", info.job_name);
        call_criu_restore(&ckpt_dir);
        log::info!("restore completed for job {}", info.job_name);
    }

    async fn detach(self, context: Context, client_pid: u32) {
        let (worker_pid, control) = {
            let mut inner = self.0.inner.lock().await;
            let worker_pid = inner
                .get_child_mut(client_pid)
                .control
                .as_ref()
                .map(|(pid, _)| *pid)
                .unwrap_or_else(|| panic!("client_pid {client_pid} is already detached"));
            let control = inner.detach(client_pid);
            (worker_pid, control)
        };
        control
            .checkpoint_process(
                context,
                CheckpointRequest {
                    job_name: String::new(),
                    ckpt_dir: String::new(),
                    mode: CheckpointMode::Detach,
                },
            )
            .await
            .unwrap();
        if let Some(client_pids) = control.detach(context).await.unwrap() {
            self.0.inner.lock().await.set_standby_worker(StandbyWorker {
                worker_pid,
                control,
                client_pids,
            });
        }
        log::info!("detach completed for client_pid {client_pid}");
    }

    async fn attach(self, context: Context, client_pid: u32) {
        let mut inner = self.0.inner.lock().await;
        let standby = inner.take_standby_worker(client_pid);
        let child = inner.get_child_for_attach(client_pid);
        let (worker_pid, control, has_standby) = match standby {
            Some(standby) => (standby.worker_pid, standby.control, true),
            None => {
                let (worker_pid, control) =
                    DaemonInner::spawn_worker(&child.cuda_visible_devices, &self.0.config).await;
                (worker_pid, control, false)
            }
        };
        control.attach(context, child.ids[0], client_pid).await.unwrap();
        child.control = Some((worker_pid, control));
        log::info!("attach completed for client_pid {client_pid}, has_standby: {has_standby}");
    }

    async fn standby(self, context: Context, client_pids: Vec<u32>) {
        let request = self.0.inner.lock().await.standby_request(&client_pids);
        let (worker_pid, control) =
            DaemonInner::spawn_worker(&request.cuda_visible_devices, &self.0.config).await;
        control.standby(context, request).await.unwrap();
        self.0.inner.lock().await.set_standby_worker(StandbyWorker {
            worker_pid,
            control,
            client_pids: client_pids.clone(),
        });
        log::info!("standby completed for client_pids {client_pids:?}");
    }
}

fn write_metadata(ckpt_dir: &str, config: &Config, job_name: &str) {
    let ckpt_dir = Path::new(ckpt_dir);
    fs::write(ckpt_dir.join("job_name.txt"), job_name).unwrap();
    config.write_to_file(&ckpt_dir.join("config.toml"));
}

struct RestoreInfo {
    config: Config,
    job_name: String,
    pid_to_ids: BTreeMap<u32, Vec<i32>>,
    min_id: i32,
    max_id: i32,
}

impl RestoreInfo {
    fn scan(ckpt_dir: &str) -> Self {
        let ckpt_dir = Path::new(ckpt_dir);
        let mut pid_to_ids: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut min_id = i32::MAX;
        let mut max_id = 0;
        for entry in fs::read_dir(ckpt_dir).unwrap() {
            let entry = entry.unwrap();
            if !entry.file_type().unwrap().is_dir() {
                continue;
            }
            let name = entry.file_name().into_string().unwrap();
            if let Some((pid, id)) = name.split_once('_')
                && let (Ok(pid), Ok(id)) = (pid.parse(), id.parse())
            {
                pid_to_ids.entry(pid).or_default().push(id);
                min_id = min_id.min(id);
                max_id = max_id.max(id);
            }
        }
        Self {
            config: Config::read_from_file(&ckpt_dir.join("config.toml")),
            job_name: fs::read_to_string(ckpt_dir.join("job_name.txt")).unwrap(),
            pid_to_ids,
            min_id,
            max_id,
        }
    }
}
