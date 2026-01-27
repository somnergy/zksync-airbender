use ::cs::cs::placeholder::Placeholder;
use ::cs::cs::witness_placer::WitnessTypeSet;
use ::cs::cs::witness_placer::{
    WitnessComputationCore, WitnessComputationalField, WitnessComputationalI32,
    WitnessComputationalInteger, WitnessComputationalU8, WitnessComputationalU16,
    WitnessComputationalU32, WitnessMask,
};
use ::field::Mersenne31Field;
use ::prover::witness_proxy::WitnessProxy;

include!("./generated.rs");
