fn main() {
    println!("cargo:rustc-link-search=native=/home/knielsen/code/matrixlib/");
    println!("cargo:rustc-link-lib=static=matrix");
}