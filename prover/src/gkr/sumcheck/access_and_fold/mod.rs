use std::mem::MaybeUninit;

use field::{FieldExtension, PrimeField};

// when we will perform sumchecks, we need temporary storage for folded versions. We assume same size of such
// "scratch space" for both the case if input poly is in the base field, or is in the extension,
// and will only perform 2 folding steps with base-field-coefficients representation for polys that are originally
// in base field. We will assert in construction that sizes and alignments are sane
pub struct IntermediatePolysArena<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    arena: Vec<Box<[MaybeUninit<E>]>>,
    size: usize,
    _marker: core::marker::PhantomData<F>,
}

// Representation of ext field poly via base field values:
// - initially we have f(0, x) and f(1, x) as base field values
// - after first folding we need to compute f'(x) = f(0, x) + r * (f(1, x) - f(0, x))
// - we can avoid making extension field value and keep it (f(0, x), f(1, x) - f(0, x)) set of coefficients
// - for simplicify f'(0) = (f(0, 0), f(1, 0) - f(0, 0)), and f'(1) = (f(0, 1), f(1, 1) - f(0, 1))
// - then we can fold again, and get f''(y) = f'(0) + r' * (f'(1) - f'(0)) = f(0, 0) + r * (f(1, 0) - f(0, 0)) + r' * (f(0, 1) + r * (f(1, 1) - f(0, 1)) - f(0, 0) - r * (f(1, 0) - f(0, 0)))
// - that itself can be viewed as bivariate poly over (r, r') with coefficients
// - (f(0, 0)) * 1 + (f(1, 0) - f(0, 0)) * r + (f(0, 1) - f(0, 0)) * r' + (f(0, 0) + f(1, 1) - f(0, 1) - f(1, 0)) * r * r'
// - eventually we will want to just have an explicit extension field value, and making it would require 3 multiplication base (as r * r' can be precomputed) * extension, 2 additions in extension and 1 ext + base

// but we could do even better if we assume max quadratic GKR kernels
// - on first step we will always use f(0, x) and f(1, x) for kernel evaluation itself
// - it would also give us (f(0, x), f(1, x) - f(0, x)) for free
// - then on the next step we would need to compute and use f'(0) and f'(1) - f'(0) => we need (f(0, 0), f(1, 0) - f(0, 0)) and (f(0, 0) - f(0, 1), f(1, 1) - f(0, 1) - (f(1, 0) - f(0, 0)) ) - very simple elementwise
// operations

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> IntermediatePolysArena<F, E> {
    pub fn new(poly_sizes_log2: usize, initial_capacity: usize) -> Self {
        assert!(core::mem::align_of::<E>() >= core::mem::align_of::<F>());
        assert!(core::mem::size_of::<E>() >= core::mem::size_of::<F>() * 4); // we need 4 temporary storage slots for base-field-coefficients representation
                                                                             // for first 2 folding steps

        let storage_size_in_extension_els = 1 << (poly_sizes_log2 - 1);
        let arena = (0..initial_capacity)
            .map(|_| Box::new_uninit_slice(storage_size_in_extension_els))
            .collect();

        Self {
            arena,
            size: storage_size_in_extension_els,
            _marker: core::marker::PhantomData,
        }
    }
}
