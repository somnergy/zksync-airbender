# C++ Airbender Example (riscv32im + GCC/newlib)

Basic example of compiling C/C++ code for Airbender.

## Requirements

- CMake 3.20 or newer.
- RISC-V GCC toolchain with newlib (`riscv32-unknown-elf-*` binaries).
  - Tested with the RISC Zero toolchain package that contains
    `riscv32-unknown-elf-gcc` / `riscv32-unknown-elf-g++` /
    `riscv32-unknown-elf-objcopy`.

## Build

```sh
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv32im-gcc.cmake \
  -DRISCV_GCC_TOOLCHAIN_ROOT=<path-to-riscv32im-linux-x86_64>
cmake --build build
```

If the toolchain binaries are already on `PATH`, `RISCV_GCC_TOOLCHAIN_ROOT` can be omitted.

For release builds, add `-DCMAKE_BUILD_TYPE=Release`.

Artifacts are written to the `examples/cpp` directory:

- `app.elf`
- `app.bin`
- `app.text`

## Running

You can run the generated program with `cargo airbender`:

```sh
# From the root of the repo
cargo airbender run examples/cpp/app.bin --input examples/cpp/input.txt
```

## Project Overview

This project contains boilerplate to compile your C/C++ code as an Airbender program.
The entrypoint for your logic is the `workload` function in [main.cpp](src/main.cpp).

Other provided files:

- [start.S](src/start.S): startup and trap entry wiring.
- [airbender_csr.hpp](include/airbender_csr.hpp): wrappers for Airbender CSR operations.
- [quasi_uart.hpp](include/quasi_uart.hpp): quasi-UART writer used for logs.
- [newlib_syscalls.cpp](src/newlib_syscalls.cpp): newlib syscall bridge (`_write`, `_exit`, `_sbrk`, etc.).

Linker scripts used:

- [link.x](../../riscv_common/src/lds/link.x)
- [memory.x](../../riscv_common/src/lds/memory.x)

## Notes

This project is intentionally C/C++-native and does not rely on Rust guest runtime crates.
As a result, it may lag behind Rust examples when platform internals change.

## License

[MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE) at your option.
