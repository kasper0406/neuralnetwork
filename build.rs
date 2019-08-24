use std::process::Command;

fn main() {
    if cfg!(cudamatrix) {
        println!("cargo:rustc-link-search=native=/home/knielsen/code/matrixlib/");
        println!("cargo:rustc-link-lib=static=matrix");
    }

    if cfg!(feature = "metalmatrix") {
        let air_status = Command::new("xcrun")
            .args(&["-sdk", "macosx", "metal", "-c", "src/matrix/metalmatrix.metal", "-o", "src/matrix/metalmatrix.air"])
            .status()
            .expect("failed to execute metal compiler");
        if !air_status.success() {
            panic!("Failed to compute metalmatrix shader");
        }

        let metal_status = Command::new("xcrun")
            .args(&["-sdk", "macosx", "metallib", "src/matrix/metalmatrix.air", "-o", "src/matrix/metalmatrix.metallib"])
            .status()
            .expect("failed to execute metal combine compiler");
        if !metal_status.success() {
            panic!("Failed assemble metalmatrix");
        }
    }
}
