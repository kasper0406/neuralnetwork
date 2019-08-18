use std::process::Command;

fn main() {
    if cfg!(matrixlib) {
        println!("cargo:rustc-link-search=native=/home/knielsen/code/matrixlib/");
        println!("cargo:rustc-link-lib=static=matrix");
    }

    // Feature flag this
    let air_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metal", "-c", "src/metalmatrix.metal", "-o", "src/metalmatrix.air"])
        .status()
        .expect("failed to execute metal compiler");
    if !air_status.success() {
        panic!("Failed to compute metalmatrix shader");
    }

    let metal_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metallib", "src/metalmatrix.air", "-o", "src/metalmatrix.metallib"])
        .status()
        .expect("failed to execute metal combine compiler");
    if !metal_status.success() {
        panic!("Failed assemble metalmatrix");
    }
}
