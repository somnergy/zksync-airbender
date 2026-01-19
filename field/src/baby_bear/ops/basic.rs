// generic implementation

#[cfg(not(target_arch = "riscv32"))]
#[cfg_attr(not(feature = "no_inline"), inline(always))]
pub(crate) const fn add_mod(a: u32, b: u32) -> u32 {
    let mut sum = a.wrapping_add(b);
    if sum >= crate::baby_bear::base::BabyBearField::ORDER {
        sum -= crate::baby_bear::base::BabyBearField::ORDER;
    }

    sum
}

#[cfg(not(target_arch = "riscv32"))]
#[cfg_attr(not(feature = "no_inline"), inline(always))]
pub(crate) const fn sub_mod(a: u32, b: u32) -> u32 {
    let (mut diff, uf) = a.overflowing_sub(b);
    if uf {
        diff = diff.wrapping_add(crate::baby_bear::base::BabyBearField::ORDER);
    }

    diff
}

#[cfg(not(target_arch = "riscv32"))]
#[cfg_attr(not(feature = "no_inline"), inline(always))]
pub(crate) const fn mul_mod(a: u32, b: u32) -> u32 {
    // TODO: as we are already not interested in lowest 32 bits in a middle of computation,
    // then we can try to eventually specialize it on different platforms

    // Montgomery multiplication
    let mut product = (a as u64).wrapping_mul(b as u64);
    let m = (product as u32).wrapping_mul(crate::baby_bear::base::BabyBearField::MONT_K);
    product = product
        .wrapping_add((m as u64).wrapping_mul(crate::baby_bear::base::BabyBearField::ORDER as u64));
    debug_assert!(product as u32 == 0);
    let mut result = (product >> 32) as u32;
    if result >= crate::baby_bear::base::BabyBearField::ORDER {
        result -= crate::baby_bear::base::BabyBearField::ORDER;
    }
    result
}

#[cfg(not(target_arch = "riscv32"))]
#[cfg_attr(not(feature = "no_inline"), inline(always))]
/// a * b + c
pub(crate) const fn fma_mod(a: u32, b: u32, c: u32) -> u32 {
    // we do not have any interesting tricks here to use
    add_mod(mul_mod(a, b), c)
}
