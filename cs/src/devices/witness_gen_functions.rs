use crate::tables::TableDriver;
use crate::tables::TableType;
use crate::types::*;
use field::PrimeField;

// expects
// in dependencies:
// - src1
// - src2
// in constants:
// is_sub flag
pub(crate) fn unconditional_add_sub_witness_gen_function_for_non_constant_operations<
    F: PrimeField,
>(
    inputs: WitnessGenSource<'_, F>,
    mut outputs: WitnessGenDest<'_, F>,
    constants: &[F],
    _table_driver: &TableDriver<F>,
    table_type: TableType,
) {
    let is_sub = constants[0].as_boolean();
    let a = Register::get_u32_from_source(inputs, 0);
    let b = Register::get_u32_from_source(inputs, 2);

    let (result, of) = if is_sub {
        a.overflowing_sub(b)
    } else {
        a.overflowing_add(b)
    };

    Register::set_u32_to_destination(result, &mut outputs, 0);
    outputs[2] = F::from_boolean(of);
}

// expects
// in dependencies:
// - src2
// in constants:
// is_sub flag
// src1
pub(crate) fn unconditional_add_sub_witness_gen_function_for_constant_src1<F: PrimeField>(
    inputs: WitnessGenSource<'_, F>,
    mut outputs: WitnessGenDest<'_, F>,
    constants: &[F],
    _table_driver: &TableDriver<F>,
    table_type: TableType,
) {
    let is_sub = constants[0].as_boolean();
    let a_low = constants[1].as_u32_reduced() as u16;
    let a_high = constants[2].as_u32_reduced() as u16;
    let a = ((a_high as u32) << 16) | (a_low as u32);

    let b = Register::get_u32_from_source(inputs, 0);

    let (c, of_flag) = match is_sub {
        false => a.overflowing_add(b),
        true => a.overflowing_sub(b),
    };
    Register::set_u32_to_destination(c, &mut outputs, 0);
    outputs[2] = F::from_boolean(of_flag);
}

// expects
// in dependencies:
// - src1
// in constants:
// is_sub flag
// src2
pub(crate) fn unconditional_add_sub_witness_gen_function_for_constant_src2<F: PrimeField>(
    inputs: WitnessGenSource<'_, F>,
    mut outputs: WitnessGenDest<'_, F>,
    constants: &[F],
    _table_driver: &TableDriver<F>,
    table_type: TableType,
) {
    let is_sub = constants[0].as_boolean();
    let b_low = constants[1].as_u32_reduced() as u16;
    let b_high = constants[2].as_u32_reduced() as u16;
    let b = ((b_high as u32) << 16) | (b_low as u32);

    let a = Register::get_u32_from_source(inputs, 0);

    let (c, of_flag) = match is_sub {
        false => a.overflowing_add(b),
        true => a.overflowing_sub(b),
    };
    Register::set_u32_to_destination(c, &mut outputs, 0);
    outputs[2] = F::from_boolean(of_flag);
}
