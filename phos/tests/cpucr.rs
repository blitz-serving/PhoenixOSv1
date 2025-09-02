use anyhow::{bail, Context};

#[allow(dead_code)]
fn kill_process(pid: i32) -> anyhow::Result<()> {    
    let output = std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .output()
        .context("Failed to spawn `kill`")?;

    if !output.status.success() {
        bail!(
            "kill -9 {} failed (code: {:?}): {}",
            pid,
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

// This test extended from https://github.com/checkpoint-restore/rust-criu/blob/main/src/bin.rs
#[test]
fn test_basic_python_checkpoint_and_restore() { 

    let pid = match std::process::Command::new("tests/piggie").output() {
        Ok(p) => String::from_utf8_lossy(&p.stdout).parse().unwrap_or(0),
        Err(e) => panic!("Starting test process failed ({:#?})", e),
    };    

    let mut cr_cpu = phos::cpucr::CPUCR::new(pid).expect("Failed to create CPUCR instance");
    cr_cpu.dump_to_default().expect("Failed to dump the test piggle process");
    cr_cpu.restore_from_default().expect("Failed to restore the test piggle process");

    //log::info!("Killing test process with pid {} for the clean up", pid);
    //kill_process(pid).expect("Failed to kill the test process");

    //log::info!("Note that the image directory {:?} has nit been cleaned up", cr_cpu.default_img_dir_path());
    println!("Cleaning up");
    std::fs::remove_dir_all(cr_cpu.default_img_dir_path()).expect("Failed to clean up the image directory");  
}