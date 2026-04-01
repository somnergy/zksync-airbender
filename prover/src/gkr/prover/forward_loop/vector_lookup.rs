use cs::definitions::gkr::NoFieldVectorLookupRelation;

use super::*;

pub(crate) fn materialize_decoder_lookup_minus_setup<
    F: PrimeField,
    E: FieldExtension<F> + Field,
>(
    decoder_predicate_address: GKRAddress,
    decoder_relation: &NoFieldVectorLookupRelation,
    multiplicity_address: GKRAddress,
    outputs: [GKRAddress; 2],
    gkr_storage: &mut GKRStorage<F, E>,
    witness_trace: &mut GKRFullWitnessTrace<F, Global, Global>,
    trace_len: usize,
    preprocessed_generic_lookup: &[E],
    lookup_challenges_multiplicative_part: E,
    lookup_challenges_additive_part: E,
    decoder_lookup_fill_value: E,
    offset_for_decoder_table: u32,
    worker: &Worker,
) {
    assert_eq!(
        decoder_relation.lookup_set_index,
        DECODER_LOOKUP_FORMAL_SET_INDEX
    );
    let mut num_destination = Box::<[E], Global>::new_uninit_slice(trace_len);
    let mut den_destination = Box::<[E], Global>::new_uninit_slice(trace_len);
    let mapping_ref = {
        // println!("Mapping decoder lookup");
        assert!(witness_trace.generic_lookup_mapping.len() > 0);
        witness_trace.generic_lookup_mapping.last().unwrap()
    };
    let decoder_predicate = gkr_storage.get_base_layer(decoder_predicate_address);
    let multiplicity = gkr_storage.get_base_layer(multiplicity_address);

    apply_row_wise::<F, _>(
        vec![],
        vec![&mut num_destination, &mut den_destination],
        trace_len,
        worker,
        |_, ext_dest, chunk_start, chunk_size| {
            assert_eq!(ext_dest.len(), 2);
            let [num_dest, den_dest] = ext_dest.try_into().unwrap();
            for i in 0..chunk_size {
                let row = chunk_start + i;
                let mapping_index = mapping_ref[row];
                let mapped_value = preprocessed_generic_lookup[mapping_index as usize];
                let decoder_predicate = decoder_predicate[row];
                let multiplicity_value = multiplicity[row];
                let setup_value = preprocessed_generic_lookup
                    .get(row)
                    .copied()
                    .unwrap_or(decoder_lookup_fill_value);

                // a/(b + gamma) - c/(d + gamma) -> (a*(d+gamma) - c*(b+gamma)), (b+gamma) * (d+gamma)

                let mut b = mapped_value;
                b.add_assign(&lookup_challenges_additive_part);

                let mut d = setup_value;
                d.add_assign(&lookup_challenges_additive_part);

                let mut num = d;
                num.mul_assign_by_base(&decoder_predicate);

                let mut t = b;
                t.mul_assign_by_base(&multiplicity_value);

                num.sub_assign(&t);

                let mut den = b;
                den.mul_assign(&d);

                num_dest[i].write(num);
                den_dest[i].write(den);

                #[cfg(feature = "gkr_self_checks")]
                {
                    let decoder_mask_value = decoder_predicate.as_boolean();
                    if decoder_mask_value {
                        assert!(mapping_index >= offset_for_decoder_table, "decoder lookup should have mapping index {} >= decoder table offset {}, and is not zero in padding", mapping_index, offset_for_decoder_table);
                    } else {
                        assert_eq!(
                            mapping_index, 0,
                            "decoder lookup should have mapping index zero in padding"
                        );
                    }

                    let naive_eval = {
                        let mut result = E::from_base(evaluate_linear_relation_at_row(
                            &decoder_relation.columns[0],
                            gkr_storage,
                            row,
                        ));
                        let mut challenge = lookup_challenges_multiplicative_part;
                        for rel in decoder_relation.columns[1..].iter() {
                            let mut t = challenge;
                            t.mul_assign_by_base(&evaluate_linear_relation_at_row(
                                rel,
                                gkr_storage,
                                row,
                            ));
                            result.add_assign(&t);

                            challenge.mul_assign(&lookup_challenges_multiplicative_part);
                        }

                        result
                    };

                    if decoder_mask_value {
                        if naive_eval != mapped_value {
                            for (idx, rel) in decoder_relation.columns.iter().enumerate() {
                                let v = evaluate_linear_relation_at_row(rel, gkr_storage, row);
                                println!("Column {} = {}", idx, v);
                            }
                        }
                        assert_eq!(
                            naive_eval, mapped_value,
                            "decoder lookup diverged at row {} for relation {:?}",
                            row, decoder_relation
                        );
                    } else {
                        if naive_eval != decoder_lookup_fill_value {
                            for (idx, rel) in decoder_relation.columns.iter().enumerate() {
                                let v = evaluate_linear_relation_at_row(rel, gkr_storage, row);
                                println!("Column {} = {}", idx, v);
                            }
                        }
                        assert_eq!(
                            naive_eval, decoder_lookup_fill_value,
                            "decoder lookup diverged at filling row {} for relation {:?}",
                            row, decoder_relation
                        );
                    }
                }
            }
        },
    );

    for (output, destination) in outputs
        .into_iter()
        .zip([num_destination, den_destination].into_iter())
    {
        let destination = unsafe { destination.assume_init() };
        output.assert_as_layer(1);
        gkr_storage.insert_extension_at_layer(1, output, ExtensionFieldPoly::new(destination));
    }
}
