// Here we will define planform-specific implementation of basic ops for the base field.
// Platform-specific implementation for extension (if any) will be in the separate files

// we will also fully define addmod/submod/mulmod, that can have different implementation strategies on different platforms,
// especially on circuit-bound ones

#[cfg(not(target_arch = "riscv32"))]
mod basic;

#[cfg(not(target_arch = "riscv32"))]
pub(crate) use self::basic::*;

#[cfg(target_arch = "riscv32")]
mod riscv32;

#[cfg(target_arch = "riscv32")]
pub(crate) use self::riscv32::*;
