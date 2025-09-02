use clap::{App, Arg, ArgGroup};

mod cpucr; 
use cpucr::CPUCR;

fn test_checkpoint(pid : i32) -> anyhow::Result<()> { 
    let mut cr_cpu = CPUCR::new(pid).expect("Failed to create CPUCR instance");
    cr_cpu.set_leave_running(false); // leave the process running after dump
    cr_cpu.set_shell_job(true); 
    cr_cpu.dump_to_default().expect("Failed to dump process");
    Ok(())
}

fn test_restore(pid : i32) -> anyhow::Result<()> { 
    let mut cr_cpu = CPUCR::new(pid).expect("Failed to create CPUCR instance");
    cr_cpu.set_shell_job(true); 
    cr_cpu.restore_from_default().expect("Failed to restore process");
    Ok(())
}

fn main() {
     env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let matches = App::new("PhoenixOS client")
        .version("1.0")
        .arg(
            Arg::new("pid")
                .long("pid")
                .short('p')
                .takes_value(true)
                .required(true)
                .value_name("PID")
                .help("Process ID to checkpoint/restore"),
        )
        .arg(
            Arg::new("checkpoint")
                .long("checkpoint")
                .help("Checkpoint a process"),
        )
        .arg(
            Arg::new("restore")
                .long("restore")
                .help("Restore a process"),
        )
        .group(
            ArgGroup::new("mode")
                .args(&["checkpoint", "restore"])
                .required(true)     // must pick one
                .multiple(false),   // forbid both
        )
        .get_matches();   

    // Get the PID argument
    let pid: i32 = matches
        .value_of_t("pid")
        .expect("PID must be a valid number");    

    if matches.is_present("checkpoint") {
        test_checkpoint(pid).expect("Checkpoint failed");    
        log::info!("Checkpoint completed for pid {}", pid);
    } else if matches.is_present("restore") {
        test_restore(pid).expect("Restore failed");
        log::info!("Restore completed for pid {}", pid);
    }        
}

    
