use std::process::{Command, exit};
use std::env;
use std::path::Path;

fn build_criu() {
    // Get the directory where build.rs is located (CARGO_MANIFEST_DIR is the root of the project)
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Set the directory of the C/C++ project (relative to the build.rs file)
    let project_dir = Path::new(&manifest_dir).join("../third_party/phoenixos-criu");

    // Ensure the project directory exists
    if !project_dir.exists() {
        eprintln!("Error: Directory {:?} does not exist!", project_dir);
        exit(1);
    }

    let build_project = env::var("BUILD_CRIU").unwrap_or_else(|_| "0".to_string());

    // If the environment variable BUILD_PROJECT is set to "0", skip the build
    if build_project == "0" {
        println!("Not building CRIU"); 
        return; // Skip the build process
    }

    // Run 'make clean' in the specified directory
    let status = Command::new("make")
        .current_dir(&project_dir)
        .arg("clean")
        .status()
        .expect("Failed to run 'make clean'");

    if !status.success() {
        eprintln!("'make clean' failed!");
        exit(1);
    }

    // Run 'make' in the specified directory
    let status = Command::new("make")
        .current_dir(&project_dir)
        .status()
        .expect("Failed to run 'make'");

    if !status.success() {
        eprintln!("'make' failed!");
        exit(1);
    }

    // Tell Cargo to rerun the build script if the files change
    println!("cargo:rerun-if-changed={}", project_dir.display());
}

fn build_test_binaries() {
    std::process::Command::new("gcc")
        .args(["tests/piggie.c", "-o", "tests/piggie"])
        .status()
        .unwrap();   
    println!("cargo:rerun-if-changed=tests/piggie.c");     
}

fn main() {
    build_criu();    
    build_test_binaries();  
}
