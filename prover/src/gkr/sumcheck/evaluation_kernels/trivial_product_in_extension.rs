use std::mem::MaybeUninit;

use cs::definitions::GKRAddress;

use super::*;

pub struct SameSizeProductGKRRelation {
    pub cached_sources: [GKRAddress; 2],
}

// Assumes reordering of access implementors, to have lhs at 0 and rhs at 1
pub struct SameSizeProductGKRRelationKernel<
    F: PrimeField,
    E: FieldExtension<F> + PrimeField,
    R: EvaluationRepresentation<F, E>,
> {
    _marker: core::marker::PhantomData<(F, E, R)>,
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField, R: EvaluationRepresentation<F, E>>
    BatchSumcheckEvaluationKernel<F, E, (), R> for SameSizeProductGKRRelationKernel<F, E, R>
{
    fn evaluate<
        S0: EvaluationFormStorage<F, E, ()>,
        S1: EvaluationFormStorage<F, E, R>,
        const FIRST_ROUND: bool,
    >(
        &self,
        index: usize,
        _r0_sources: &[S0],
        r1_sources: &[S1],
        batch_challenge: &E,
    ) -> [E; 2] {
        unsafe {
            let [lhs, rhs] = r1_sources
                .as_chunks::<2>()
                .0
                .iter()
                .next()
                .unwrap_unchecked();
            let ctx = lhs.get_collapse_context();
            let lhs = lhs.get_f0_and_f1_minus_f0(index);
            let rhs = rhs.get_f0_and_f1_minus_f0(index);
            let mut result = [const { MaybeUninit::uninit() }; 2];
            for i in 0..2 {
                let mut product = lhs[i];
                product.repr_mul_assign::<true>(&rhs[i]);
                result[i].write(product.collapse_for_batch_eval(ctx, batch_challenge));
            }

            result.map(|el| el.assume_init())
        }
    }
}
