use std::collections::BTreeMap;

use cs::definitions::GKRAddress;
use field::FieldExtension;

use super::*;

pub struct PolysStorage<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    pub polys_in_base: F,
    pub polys_in_extension: BTreeMap<GKRAddress, FoldingStorageForExtPoly<E>>,
}

// Poly that is itself in extension when it's an input to some GKR relation and corresponding sumcheck later.
// Some caches (like grand product contribtuions) can be considered so
pub struct FoldingStorageForExtPoly<F: PrimeField> {
    pub evaluations: Vec<F>,
    pub folding_buffers: Vec<Vec<F>>,
}

impl<F: PrimeField> FoldingStorageForExtPoly<F> {
    pub fn new(evaluations: Vec<F>) -> Self {
        let mut size = evaluations.len();
        assert!(size.is_power_of_two());
        let folding_steps = size.trailing_zeros();
        let mut folding_buffers = Vec::with_capacity(folding_steps as usize);
        for _ in 0..folding_steps {
            size /= 2;
            folding_buffers.push(Vec::with_capacity(size));
        }
        assert_eq!(size, 1);

        Self {
            evaluations,
            folding_buffers,
        }
    }

    // TODO: consider using guards

    pub fn start_first_folding_step<'a>(&'a mut self) -> FirstStepInExtensionLazyFolder<'a, F> {
        let size = self.evaluations.len();
        let input = &self.evaluations[..];
        let buffer = &mut self.folding_buffers[0];
        let repeated = buffer.is_empty() == false;
        let buffer = &mut buffer.spare_capacity_mut()[..size / 2];

        FirstStepInExtensionLazyFolder::construct(input, (), buffer, repeated)
    }

    pub fn finish_first_folding_step(&mut self) {
        if self.folding_buffers[0].is_empty() {
            unsafe {
                self.folding_buffers[0].set_len(self.evaluations.len() / 2);
            }
        }
    }

    pub fn start_next_folding_step<'a>(
        &'a mut self,
        step: usize,
    ) -> NextStepsInExtensionLazyFolder<'a, F> {
        assert!(step + 1 <= self.folding_buffers.len());
        // we need 3 buffers here

        todo!();
    }

    pub fn finish_next_folding_step(&mut self, step: usize) {
        if self.folding_buffers[step + 1].is_empty() {
            unsafe {
                self.folding_buffers[step + 1].set_len(self.evaluations.len() / 2);
            }
        }
    }
}
