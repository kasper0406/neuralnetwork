# Compiling

By default the library will execute all matrix operations on the CPU. The program can be run the ordinary Rust way using:
```
cargo run --release
```
It is also possible to offload the matrix operations using either CUDA or Metal. This can be enabled by the following respective commands:
```
cargo run --release --features cudamatrix
cargo run --release --features metalmatrix
```

## Testing matrix implementation correctness

In order to have confidence that the different matrix implementations provides similar output (up to float representation weirdness), a `VerifyingMatrixHandle` has been implemented, that uses the CPU implementation to compare the results with respectively the CUDA and Metal implementations.
Currently this is not supported with a feature flag. Instead the code needs to be changed to reference the verifying matrix handle type.
