#![allow(dead_code)]
/// CPUCR: A simple wrapper for rust-criu (Checkpoint/Restore In Userspace) to checkpoint a CPU process
///
/// Main Interfaces:
///     - `CPUCR::new(criu_path: String) -> Result<CPUCR>`
///         Creates a new CPUCR instance with the specified path to the CRIU executable.
///     - `CPUCR::dump(&mut self, pid: i32, images_dir_fd: i32) -> Result<()>`
///         Performs a checkpoint (dump) of the process with the given pid.
///         `images_dir_fd` is the file descriptor for the directory where checkpoint images will be stored.
///         The process will be stopped after dumping.
///
use rust_criu::Criu;
use std::fs::{self, File};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

pub struct CPUCR {
    pid: i32,
    criu: Criu, // cached, ready-to-use
}

impl CPUCR {
    pub fn new(pid: i32) -> anyhow::Result<Self> {
        let criu_bin = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../third_party/phoenixos-criu/criu/criu");
        // pass a &Path (or &PathBuf) — not a String
        Self::with_criu_path(pid, &criu_bin)
    }

    // Accept any path-like input
    pub fn with_criu_path<P: AsRef<Path>>(pid: i32, criu_path: P) -> anyhow::Result<Self> {
        let criu_path = criu_path.as_ref();
        if !criu_path.exists() {
            anyhow::bail!("CRIU binary not found at {}", criu_path.display());
        }
        // Convert Path -> String for the CRIU API
        let path_str = criu_path.to_string_lossy().into_owned();

        let criu = Criu::new_with_criu_path(path_str)
            .map_err(|e| anyhow::anyhow!("Failed to create Criu instance: {:#?}", e))?;

        Ok(Self { pid, criu })
    }

    /// Forwarding setters: configure the cached CRIU handle any time.
    pub fn set_shell_job(&mut self, on: bool)           { self.criu.set_shell_job(on); }
    pub fn set_leave_running(&mut self, on: bool)       { self.criu.set_leave_running(on); }
    pub fn set_log_level(&mut self, level: i32)         { self.criu.set_log_level(level); }
    pub fn set_log_file<P: Into<String>>(&mut self, p: P){ self.criu.set_log_file(p.into()); }
    pub fn set_pid(&mut self, pid: i32)                 { self.pid = pid; }
    pub fn get_pid(&self) -> i32                          { self.pid }

    /// Path: "images_<pid>"
    pub fn default_img_dir_path(&self) -> PathBuf {
        PathBuf::from(format!("images_{}", self.pid))
    }

    /// Ensure a clean default dir and return a live handle (FD stays valid while `File` is alive).
    pub fn open_default_img_dir(&self, clean : bool) -> anyhow::Result<File> {
        let dir = self.default_img_dir_path();

        if clean { 
            if dir.exists() {
                fs::remove_dir_all(&dir)
                    .map_err(|e| anyhow::anyhow!("Failed to remove '{}': {:#?}", dir.display(), e))?;
            }
            fs::create_dir(&dir)
                .map_err(|e| anyhow::anyhow!("Failed to create '{}': {:#?}", dir.display(), e))?;
        }

        File::open(&dir)
            .map_err(|e| anyhow::anyhow!("Failed to open dir '{}': {:#?}", dir.display(), e))
    }

    /// Dump into the default per-PID dir (keeps the dir `File` local so FD remains valid).
    pub fn dump_to_default(&mut self) -> anyhow::Result<()> {
        let dir = self.open_default_img_dir(true)?;
        self.dump_to_dir(&dir)
    }

    pub fn restore_from_default(&mut self) -> anyhow::Result<()> {
        let dir = self.open_default_img_dir(false)?;
        self.restore_from_dir(&dir)
    }

    /// Core dump using the cached CRIU: only per-call fields are set here.
    pub fn dump_to_dir(&mut self, images_dir: &File) -> anyhow::Result<()> {
        self.criu.set_images_dir_fd(images_dir.as_raw_fd());
        self.criu.set_pid(self.pid);
        self.criu.dump()
            .map_err(|e| anyhow::anyhow!("Failed to dump process: {:#?}", e))
    }

    pub fn restore_from_dir(&mut self, images_dir: &File) -> anyhow::Result<()> {
        self.criu.set_images_dir_fd(images_dir.as_raw_fd());
        self.criu.set_pid(self.pid);
        self.criu
            .restore()
            .map_err(|e| anyhow::anyhow!("Failed to restore process: {:#?}", e))
    }

    /// Escape hatch: allow advanced one-off configuration.
    pub fn criu_mut(&mut self) -> &mut Criu {
        &mut self.criu
    }
}
