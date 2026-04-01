use super::*;
use crate::gkr::prover::{apply_row_wise, GKRExternalChallenges};
use crate::gkr::sumcheck::access_and_fold::BaseFieldPolySource;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct GKRInputs {
    pub inputs_in_base: Vec<GKRAddress>,
    pub inputs_in_extension: Vec<GKRAddress>,
    pub outputs_in_base: Vec<GKRAddress>,
    pub outputs_in_extension: Vec<GKRAddress>,
}

#[derive(Clone, Debug, Default)]
pub struct BatchedGKRTermDescriptionConstants<F: PrimeField, E: FieldExtension<F> + Field> {
    pub external_challenges: GKRExternalChallenges<F, E>,
    pub lookup_challenges_multiplicative_part: E,
    pub lookup_challenges_additive_part: E,
    pub constraints_batch_challenge: E,
    pub _marker: core::marker::PhantomData<F>,
}

#[derive(Clone, Debug, Default)]
pub struct BatchedGKRTermDescription<F: PrimeField, E: FieldExtension<F> + Field> {
    pub quadratic_part_base_by_base: BTreeMap<GKRAddress, BTreeMap<GKRAddress, E>>,
    pub quadratic_part_base_by_ext: BTreeMap<GKRAddress, BTreeMap<GKRAddress, E>>,
    pub quadratic_part_ext_by_ext: BTreeMap<GKRAddress, BTreeMap<GKRAddress, E>>,
    pub linear_part_base: BTreeMap<GKRAddress, E>,
    pub linear_part_ext: BTreeMap<GKRAddress, E>,
    pub constant_term: E,
    pub output_in_base: Option<GKRAddress>,
    pub output_in_extension: Option<GKRAddress>,
    pub _marker: core::marker::PhantomData<F>,
}

pub trait BatchedGKRKernel<F: PrimeField, E: FieldExtension<F> + Field> {
    fn num_challenges(&self) -> usize;
    fn get_inputs(&self) -> GKRInputs;
    fn evaluate_forward_over_storage(
        &self,
        storage: &mut GKRStorage<F, E>,
        expected_output_layer: usize,
        input_trace_len: usize,
        worker: &Worker,
    );
    fn evaluate_over_storage<const N: usize>(
        &self,
        storage: &mut GKRStorage<F, E>,
        step: usize,
        batch_challenges: &[E],
        folding_challenges: &[E],
        accumulator: &mut [[E; 2]],
        total_sumcheck_rounds: usize,
        last_evaluations: &mut BTreeMap<GKRAddress, [E; N]>,
        worker: &Worker,
    );

    fn terms(
        &self,
        _challenge_constants: &BatchedGKRTermDescriptionConstants<F, E>,
    ) -> Vec<BatchedGKRTermDescription<F, E>> {
        unimplemented!(
            "Not implemented yet for {:?}",
            core::any::type_name::<Self>()
        );
    }
}

pub fn evaluate_single_input_kernel_with_base_inputs<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    K: SingleInputTypeBatchSumcheckEvaluationKernel<F, E, OUT>,
    const N: usize,
    const OUT: usize,
>(
    kernel: &K,
    inputs: &GKRInputs,
    storage: &mut GKRStorage<F, E>,
    step: usize,
    challenges: &[E],
    folding_challenges: &[E],
    accumulator: &mut [[E; 2]],
    total_sumcheck_rounds: usize,
    last_evaluations: &mut BTreeMap<GKRAddress, [E; N]>,
    worker: &Worker,
) {
    assert_eq!(challenges.len(), kernel.num_challenges());

    let work_size = accumulator.len();
    assert!(work_size.is_power_of_two());
    match step {
        0 => {
            let sources = storage.get_for_sumcheck_round_0(inputs);
            let base_field_inputs = &sources.base_field_inputs;
            assert!(sources.extension_field_inputs.is_empty());
            assert_eq!(
                sources.base_field_outputs.len(),
                inputs.outputs_in_base.len()
            );
            assert_eq!(
                sources.extension_field_outputs.len(),
                inputs.outputs_in_extension.len()
            );
            if sources.base_field_outputs.is_empty() == false {
                let outputs = &sources.base_field_outputs;
                apply_row_wise::<F, _>(
                    vec![],
                    vec![accumulator],
                    work_size,
                    worker,
                    |_, mut ext_dest, chunk_start, chunk_size| {
                        assert_eq!(ext_dest.len(), 1);
                        let accumulator = ext_dest.pop().unwrap();
                        for index in 0..chunk_size {
                            let absolute_index = chunk_start + index;
                            let value = kernel.evaluate_first_round(
                                absolute_index,
                                base_field_inputs,
                                outputs,
                                challenges,
                                &(),
                                &(),
                            );
                            for i in 0..2 {
                                accumulator[index][i].add_assign(&value[i]);
                            }
                        }
                    },
                );
            } else if sources.extension_field_outputs.is_empty() == false {
                let outputs = &sources.extension_field_outputs;
                apply_row_wise::<F, _>(
                    vec![],
                    vec![accumulator],
                    work_size,
                    worker,
                    |_, mut ext_dest, chunk_start, chunk_size| {
                        assert_eq!(ext_dest.len(), 1);
                        let accumulator = ext_dest.pop().unwrap();
                        for index in 0..chunk_size {
                            let absolute_index = chunk_start + index;
                            let value = kernel.evaluate_first_round(
                                absolute_index,
                                base_field_inputs,
                                outputs,
                                challenges,
                                &(),
                                &(),
                            );
                            for i in 0..2 {
                                accumulator[index][i].add_assign(&value[i]);
                            }
                        }
                    },
                );
            } else {
                let outputs: &[BaseFieldPolySource<F>] = &[][..];
                apply_row_wise::<F, _>(
                    vec![],
                    vec![accumulator],
                    work_size,
                    worker,
                    |_, mut ext_dest, chunk_start, chunk_size| {
                        assert_eq!(ext_dest.len(), 1);
                        let accumulator = ext_dest.pop().unwrap();
                        for index in 0..chunk_size {
                            let absolute_index = chunk_start + index;
                            let value = kernel.evaluate_first_round(
                                absolute_index,
                                base_field_inputs,
                                outputs,
                                challenges,
                                &(),
                                &(),
                            );
                            for i in 0..2 {
                                accumulator[index][i].add_assign(&value[i]);
                            }
                        }
                    },
                );
            }
        }
        1 => {
            let sources = storage.get_for_sumcheck_round_1(inputs, folding_challenges);
            let inputs = &sources.base_field_inputs;
            assert!(sources.extension_field_inputs.is_empty());
            apply_row_wise::<F, _>(
                vec![],
                vec![accumulator],
                work_size,
                worker,
                |_, mut ext_dest, chunk_start, chunk_size| {
                    assert_eq!(ext_dest.len(), 1);
                    let accumulator = ext_dest.pop().unwrap();
                    let ctx = inputs[0].get_collapse_context();
                    for index in 0..chunk_size {
                        let absolute_index = chunk_start + index;
                        let value =
                            kernel.evaluate::<_, _, false>(absolute_index, inputs, challenges, ctx);
                        for i in 0..2 {
                            accumulator[index][i].add_assign(&value[i]);
                        }
                    }
                },
            );
        }
        2 => {
            let sources = storage.get_for_sumcheck_round_2(inputs, folding_challenges);
            let inputs = &sources.base_field_inputs;
            assert!(sources.extension_field_inputs.is_empty());
            apply_row_wise::<F, _>(
                vec![],
                vec![accumulator],
                work_size,
                worker,
                |_, mut ext_dest, chunk_start, chunk_size| {
                    assert_eq!(ext_dest.len(), 1);
                    let accumulator = ext_dest.pop().unwrap();
                    let ctx = inputs[0].get_collapse_context();
                    for index in 0..chunk_size {
                        let absolute_index = chunk_start + index;
                        let value =
                            kernel.evaluate::<_, _, false>(absolute_index, inputs, challenges, ctx);
                        for i in 0..2 {
                            accumulator[index][i].add_assign(&value[i]);
                        }
                    }
                },
            );
        }
        i if i + 1 == total_sumcheck_rounds => {
            assert!(i >= 3);

            let sources = storage.get_for_sumcheck_round_3_and_beyond(inputs, folding_challenges);
            {
                let inputs = &sources.base_field_inputs;
                assert!(sources.extension_field_inputs.is_empty());
                apply_row_wise::<F, _>(
                    vec![],
                    vec![accumulator],
                    work_size,
                    worker,
                    |_, mut ext_dest, chunk_start, chunk_size| {
                        assert_eq!(ext_dest.len(), 1);
                        let accumulator = ext_dest.pop().unwrap();
                        let ctx = inputs[0].get_collapse_context();
                        for index in 0..chunk_size {
                            let absolute_index = chunk_start + index;
                            let value = kernel.evaluate::<_, _, true>(
                                absolute_index,
                                inputs,
                                challenges,
                                ctx,
                            );
                            for i in 0..2 {
                                accumulator[index][i].add_assign(&value[i]);
                            }
                        }
                    },
                );
            }

            sources.collect_last_values(inputs, last_evaluations);
        }
        3.. => {
            let sources = storage.get_for_sumcheck_round_3_and_beyond(inputs, folding_challenges);
            let inputs = &sources.base_field_inputs;
            assert!(sources.extension_field_inputs.is_empty());
            apply_row_wise::<F, _>(
                vec![],
                vec![accumulator],
                work_size,
                worker,
                |_, mut ext_dest, chunk_start, chunk_size| {
                    assert_eq!(ext_dest.len(), 1);
                    let accumulator = ext_dest.pop().unwrap();
                    let ctx = inputs[0].get_collapse_context();
                    for index in 0..chunk_size {
                        let absolute_index = chunk_start + index;
                        let value =
                            kernel.evaluate::<_, _, false>(absolute_index, inputs, challenges, ctx);
                        for i in 0..2 {
                            accumulator[index][i].add_assign(&value[i]);
                        }
                    }
                },
            );
        }
    }
}
