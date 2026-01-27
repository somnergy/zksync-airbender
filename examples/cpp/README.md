# C++ Airbender Example (riscv32im + clang)

Basic example of compiling C/C++ code for Airbender.

## Requirements

`clang` v15 or higher. `lld` also must be v15 or higher.

## Build

```sh
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/riscv32im-clang.cmake
cmake --build build
```

By default, it'll use `clang` / `clang++` / `lld` to compile code. `llvm-objcopy` is used to get the required binaries.
They can be overridden via `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` / `CMAKE_ASM_COMPILER` / `CMAKE_LINKER` / `CMAKE_OBJCOPY`, e.g.:

```sh
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv32im-clang.cmake \
  -DCMAKE_C_COMPILER=clang-18 \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_ASM_COMPILER=clang-18 \
  -DCMAKE_LINKER=lld-18 \
  -DCMAKE_OBJCOPY=/usr/bin/llvm-objcopy-18
```

For release builds, add `-DCMAKE_BUILD_TYPE=Release`.

Artifacts are written to the `examples/cpp` directory:

- `app.elf`
- `app.bin`
- `app.text`

## Running

You can use [airbender CLI](../../tools/cli/) to run the project.

Example:

```sh
# From the root of the repo
cd tools/cli

RUST_MIN_STACK=16777216 cargo run run --bin ../../examples/cpp/app.bin --input-file ../../examples/cpp/input.txt
```

The same tool can be used for generating proofs.
See the tool readme to get familiar with its capabilities.

## Project overview

This project contains boilerplate to compile your C/C++ code as an Airbender program.
The entrypoint for your logic is the `workload` function in [main.cpp](src/main.cpp).

Other files that are provided:
- [start.S](src/start.S) - expected initialization sequence for the application (e.g. `_start` function).
- [airbender_csr.hpp](include/airbender_csr.hpp) - convenience wrappers for custom delegations provided by Airbender.
- [quasi_uart.hpp](include/quasi_uart.hpp) - writer device, can be used for implementing debug logs.
- [simple_allocator.cpp](src/simple_allocator.cpp) - basic bump allocator that utilizes the heap section defined by linker.
- [memset.c](src/memset.c) - naive `memset` implementation, alternatively consider [assembly impl](../../riscv_common/src/asm/memset.s).


Linker scripts used:
- [link.x](../scripts/lds/link.x).
- [memory.x](../scripts/lds/memory.x).

## Notes

This project is intentionally C/C++-native, it does not rely on any Rust dependencies.
As a result, it is possible that it will get outdated compared to other Rust examples in this repository.
If you are experiencing any problems with e.g. ability to invoke certain CSRs, please check that C++
definitions match the definitions in Rust; if they don't, please submit a PR.

## License

[MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE) at your option.

## Notes

- No Rust toolchain is required.
- The startup/trap assembly and linker scripts are shared with other examples:
  [`examples/scripts/asm/asm_reduced.S`](../scripts/asm/asm_reduced.S) and [`examples/scripts/lds/{memory.x,link.x}`](../scripts/lds/).
- CSR helpers live in `include/airbender_csr.hpp`.

