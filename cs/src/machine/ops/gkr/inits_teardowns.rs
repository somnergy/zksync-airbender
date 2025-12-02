use field::{Field, FieldExtension, PrimeField};
use crate::cs::gkr_circuit::*;

fn define_circuit<F: PrimeField, FEXT: FieldExtension<F>>(ram_challenges: (FEXT, [FEXT; RAM_WIDTH - 1]), lookup_challenges: (FEXT, [FEXT; LOOKUP_WIDTH - 1]), start_time: u64, end_time: u64) -> FullGKRCircuit<F, FEXT, 2> {
    let mut cs = GKRLayerCircuit::<F, FEXT, 2>::new();
    cs.initialise_finalise_inits_and_teardowns(ram_challenges, lookup_challenges, start_time, end_time)
}