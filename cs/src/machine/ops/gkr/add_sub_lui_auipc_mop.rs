use field::{Field, FieldExtension, PrimeField};
use crate::cs::gkr_circuit::*;
use core::array::from_fn;
use crate::{constraint::{Constraint, Term}, types::{Num, Register}};

fn define_circuit<F: PrimeField, FEXT: FieldExtension<F>>(ram_challenges: (FEXT, [FEXT; RAM_WIDTH - 1]), lookup_challenges: (FEXT, [FEXT; LOOKUP_WIDTH - 1])) -> FullGKRCircuit<F, FEXT, 2> {
    let mut cs = GKRLayerCircuit::<F, FEXT, 2>::new();
    let GKROpcodeFamilyDecoding {
        pc,
        rs1_idx,
        rs2_idx,
        rd_idx,
        not_rdx0,
        immediate,
        f3: _f3,
        mask
    } = cs.initialise(ram_challenges, lookup_challenges);

    let (_rs1_read_time, rs1_reg) = {
        let addr = [Constraint::from(rs1_idx), Constraint::from(0)];
        cs.read::<true>(Constraint::from(1), addr)
    };
    let (_rs2_read_time, rs2_reg) = {
        let addr = [Constraint::from(rs2_idx), Constraint::from(0)];
        cs.read::<true>(Constraint::from(1), addr)
    };
    let [is_add, is_addi, is_sub, is_lui, is_auipc, is_addmod, is_submod, is_mulmod] = cs.decompose_bitmask(Constraint::from(mask));
    
    // add:   rs1 + rs2 == out + 2^32 of
    // addi:  rs1 + imm == out + 2^32 of
    // sub:   rs1 - rs2 == out - 2^32 of <=> out + rs2 == rs1 + 2^32 of
    // lui:   0   + imm == out + 2^32 of
    // auipc: pc  + imm == out + 2^32 of
    // addmod/submod/mulmod: out == modop(rs1, rs2) /\ out - (2^31-1) == tmp - 2^32 <=> tmp + (2^31-1) == out + 2^32
    let out = cs.new_u32();
    let tmp = cs.new_u32();
    let ofs = [cs.new_boolean(), cs.new_boolean()];
    let modulo_reg = {
        const MODULO: u32 = (1<<31) - 1;
        const MODULO_LOW: u16 = MODULO as u16;
        const MODULO_HIGH: u16 = (MODULO >> 16) as u16;
        Register([Num::Constant(F::from_u64_unchecked(MODULO_LOW as u64)), Num::Constant(F::from_u64_unchecked(MODULO_HIGH as u64))])
    };
    let a = from_fn(|i| 
        (Constraint::from(is_add) + Constraint::from(is_addi)) * Constraint::from(rs1_reg.0[i])
        + Constraint::from(is_sub) * Constraint::from(out.0[i])
        + Constraint::from(is_auipc) * Constraint::from(pc.0[i])
        + (Constraint::from(is_addmod) + Constraint::from(is_submod) + Constraint::from(is_mulmod)) * Constraint::from(tmp.0[i])
    );
    let b = from_fn(|i|
        (Constraint::from(is_add) + Constraint::from(is_sub)) * Constraint::from(rs2_reg.0[i])
        + (Constraint::from(is_addi) + Constraint::from(is_lui) + Constraint::from(is_auipc)) * Constraint::from(immediate.0[i])
        + (Constraint::from(is_addmod) + Constraint::from(is_submod) + Constraint::from(is_mulmod)) * Constraint::from(modulo_reg.0[i])
    );
    let c = from_fn(|i|
        (Constraint::from(is_add) + Constraint::from(is_addi) + Constraint::from(is_lui) + Constraint::from(is_auipc) + Constraint::from(is_addmod) + Constraint::from(is_submod) + Constraint::from(is_mulmod)) * Constraint::from(out.0[i])
        + Constraint::from(is_sub) * Constraint::from(rs1_reg.0[i])
    );
    let of_low = Constraint::from(ofs[0]);
    let of_high_special = (Constraint::from(is_add) + Constraint::from(is_addi) + Constraint::from(is_sub) + Constraint::from(is_sub) + Constraint::from(is_lui) + Constraint::from(is_auipc)) * Constraint::from(ofs[1])
        + (Constraint::from(is_addmod) + Constraint::from(is_submod) + Constraint::from(is_mulmod));
    cs.enforce_addition::<16>(a, b, c, [of_low, of_high_special]);
    // don't forget addmod/submod/mulmod: out == modop(rs1, rs2)
    let rs1mod = Constraint::from(rs1_reg.0[0]) + Term::from(1<<16)*Term::from(rs1_reg.0[1]);
    let rs2mod = Constraint::from(rs2_reg.0[0]) + Term::from(1<<16)*Term::from(rs2_reg.0[1]);
    let outmod = Constraint::from(out.0[0]) + Term::from(1<<16)*Term::from(out.0[1]);
    let addmod = rs1mod.clone() + rs2mod.clone();
    let submod = rs1mod.clone() - rs2mod.clone();
    let mulmod = cs.new_variable_from_constraint(rs1mod * rs2mod);
    cs.enforce_constraint(
        Constraint::from(is_addmod) * (outmod.clone() - addmod)
        + Constraint::from(is_submod) * (outmod.clone() - submod)
        + Constraint::from(is_mulmod) * (outmod - Term::from(mulmod))
    );





    {
        let addr: [Constraint<_>; 2] = [Constraint::from(rd_idx), Constraint::from(0)];
        let masked_out = from_fn(|i|
            Term::from(out.0[i]) * Term::from(not_rdx0)
        );
        cs.write::<true>(None, Constraint::from(1), addr, masked_out);
    }
    let pc_next = {
        let pc = pc.0.map(Constraint::from);
        let four = [4,0].map(Constraint::from);
        let ofs = [cs.new_boolean(), cs.new_boolean()];
        cs.get_addition::<16>(pc, four, ofs)
    };
    cs.finalise(pc_next)
}