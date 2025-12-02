use core::{array::from_fn, ops::MulAssign};
use std::marker::PhantomData;
use common_constants::TIMESTAMP_STEP;
use field::{Field, FieldExtension, PrimeField};

use crate::{constraint::{Constraint, Term}, definitions::Variable, tables::TableType, types::{Boolean, Num, Register}};

pub struct FullGKRCircuit<F: PrimeField, FEXT: FieldExtension<F>, const GATE_DEGREE: usize> 
{
    // layers: Vec<GKRIntermediateLayer>,
    _marker: (PhantomData<F>, PhantomData<FEXT>)
}

// struct GKRIntermediateLayer {
//     unique_gates: Vec<(GKRSelector, Constraint<FEXT>)> // (wiring, gate)
// }

// struct GKRSelector(Constraint<FEXT>);


fn lift_constraint<F: PrimeField, FEXT: FieldExtension<F>>(expression: Constraint<F>) -> Constraint<FEXT> {
    fn lift_term<F: PrimeField, FEXT: FieldExtension<F>>(term: Term<F>) -> Term<FEXT> {
        match term {
            Term::Constant(base_constant) => Term::Constant(FEXT::from_base(base_constant)),
            Term::Expression { coeff: base_coeff, inner, degree } => Term::Expression { coeff: FEXT::from_base(base_coeff), inner, degree }
        }
    }
    let fext_terms = expression.terms.into_iter().map(|term| lift_term(term)).collect();
    Constraint { terms: fext_terms }
}

fn lift_compress_randomise_tuple<F: PrimeField, FEXT: FieldExtension<F>, const N: usize, const NMINUS1: usize>(tuple: [Constraint<F>; N], challenges: (FEXT, [FEXT; NMINUS1])) -> Constraint<FEXT> 
{
    let _: () = assert!(NMINUS1 == N - 1);
    let mut lifted_tuple = tuple.map(lift_constraint);
    let (accumulation_challenge, compression_challenges) = challenges;
    for (expression, alpha) in lifted_tuple[1..].iter_mut().zip(compression_challenges) {
        expression.scale(alpha);
    }
    let compressed_tuple = lifted_tuple.into_iter().reduce(|acc, expr| acc + expr).unwrap();
    let gamma = Constraint::from_field(accumulation_challenge);
    gamma + compressed_tuple
}

// fn mask_ram_tuple<F: PrimeField>(tuple: [Constraint<F>; RAM_WIDTH], time_mask: Boolean, value_mask: Boolean) -> [Constraint<F>; RAM_WIDTH] {
//     let tmask = Term::from(time_mask);
//     let vmask = Term::from(value_mask);
//     let [addr_low, addr_high, is_reg, time_low, time_high, val_low, val_high] = tuple;
//     [addr_low, addr_high, is_reg, time_low * tmask, time_high * tmask, val_low * vmask, val_high * vmask]
// }

const MAX_CYCLE_UNIQUE_RAM_ACCESSES: usize = 4;
pub const LOOKUP_WIDTH: usize = 11;
pub const RAM_WIDTH: usize = 7;
const STATE_PERMUTATION_ADDR_LOW: u64 = 32;
const TIMESTAMP_LIMB_BITS: usize = 19;

#[derive(Debug)]
struct GKRPlaceholderMemoryState<F: PrimeField> {
    execute: Boolean,
    time: Register<F>, // u19 limbs
    pc: Register<F>,
    cycle_read_times: [Option<Register<F>>; MAX_CYCLE_UNIQUE_RAM_ACCESSES], // u19 limbs
    cycle_read_values: Vec<Register<F>>, // can be much more than MAX_CYCLE_UNIQUE_RAM_ACCESSES
    // TODO: add offset variables here too...
}

#[derive(Debug)]
struct GKRPlaceholderState<F: PrimeField> {
    memory: GKRPlaceholderMemoryState<F>,
    table: [Variable; LOOKUP_WIDTH],
    multiplicity: Variable,
}

pub struct GKROpcodeFamilyDecoding<F: PrimeField> {
    pub pc: Register<F>,
    pub rs1_idx: Variable,
    pub rs2_idx: Variable,
    pub rd_idx: Variable,
    pub not_rdx0: Boolean,
    pub immediate: Register<F>,
    pub f3: Variable,
    pub mask: Variable
}

#[derive(Debug, Default)]
pub struct GKRLayerCircuit<F: PrimeField, FEXT: FieldExtension<F>, const GATE_DEGREE: usize> 
{
    // public_input: Option<Num<F>>, // test
    placeholder_state: Option<GKRPlaceholderState<F>>,
    total_vars: u64,
    total_cycle_unique_reads: usize,
    total_cycle_unique_writes: usize,
    total_zerocheck_gates: Vec<Constraint<FEXT>>,
    total_copy_gates: Vec<Variable>,
    total_readset_gates: Vec<Constraint<FEXT>>,
    total_writeset_gates: Vec<Constraint<FEXT>>,
    total_lookupset_gates: Vec<(Option<Constraint<FEXT>>, Constraint<FEXT>)>, // (numerator, denominator)
    total_tableset_gates: Vec<(Constraint<FEXT>, Constraint<FEXT>)>,
    lookup_challenges: (FEXT, [FEXT; LOOKUP_WIDTH - 1]), // LogUp
    ram_challenges: (FEXT, [FEXT; RAM_WIDTH - 1]), // 2 Shuffles Make A RAM
}

impl<F: PrimeField, FEXT: FieldExtension<F>, const GATE_DEGREE: usize> GKRLayerCircuit<F, FEXT, GATE_DEGREE> 
{
    const TIME_STEP: usize = {
        assert!(MAX_CYCLE_UNIQUE_RAM_ACCESSES.count_ones() == 1);
        MAX_CYCLE_UNIQUE_RAM_ACCESSES
    };

    pub fn new() -> Self {
        Self::default()
    }

    pub fn initialise(&mut self, ram_challenges: (FEXT, [FEXT; RAM_WIDTH - 1]), lookup_challenges: (FEXT, [FEXT; LOOKUP_WIDTH - 1])) -> GKROpcodeFamilyDecoding<F> {
        // NB: no need to range check (pc, time) as they're part of memory ReadSet, which is range checked through Memory Argument
        self.ram_challenges = ram_challenges;
        self.lookup_challenges = lookup_challenges;

        // append copy and booleancheck for execute
        let execute = self.new_boolean(); // from external witness generation
        self.append_copygate(execute.get_variable().unwrap());
        
        // append readset for pc/time
        let time = Register([self.new_variable(), self.new_variable()].map(Num::Var)); // from external witness generation
        let pc = Register([self.new_variable(), self.new_variable()].map(Num::Var)); // from external witness generation
        let [state_addr_low, state_addr_high] = [Constraint::from(STATE_PERMUTATION_ADDR_LOW), Constraint::from(0)];
        let is_register = Constraint::from(1);
        let [time_low, time_high] = time.0.map(Constraint::from);
        let [pc_low, pc_high] = pc.0.map(Constraint::from);
        self.append_readset([state_addr_low, state_addr_high, is_register, time_low, time_high, pc_low.clone(), pc_high.clone()]);
        
        // append tableset for setup/multiplicity
        let table = from_fn(|_| self.new_variable()); // from external witness generation
        let multiplicity = self.new_variable(); // from external witness generation
        self.append_tableset(table, multiplicity);

        // append lookupset for decoder
        // all vars range checked by lookup
        let table_id = Constraint::from(TableType::OpcodeFamilyDecoder.to_num());
        let rs1_idx = self.new_variable();
        let rs2_idx = self.new_variable();
        let rd_idx = self.new_variable();
        let not_rdx0 = Boolean::Is(self.new_variable());
        let immediate = Register([self.new_variable(), self.new_variable()].map(Num::Var));
        let [immediate_low, immediate_high] = immediate.0.map(Constraint::from);
        let f3 = self.new_variable();
        let mask = self.new_variable();
        self.append_lookupset([table_id, pc_low, pc_high, Constraint::from(rs1_idx), Constraint::from(rs2_idx), Constraint::from(rd_idx), Constraint::from(not_rdx0), immediate_low, immediate_high, Constraint::from(f3), Constraint::from(mask)]);

        // store placeholders
        // TODO: we need to add cycle read_addresses
        // TODO: we need to add main cycle pc/time read elements..
        self.placeholder_state = Some(GKRPlaceholderState {
            // these values need to go in RAM transcript
            // we make a first pass external witness gen and commitment for them
            memory: GKRPlaceholderMemoryState {
                execute,
                time,
                pc,
                cycle_read_times: [None; MAX_CYCLE_UNIQUE_RAM_ACCESSES], // we do not have any yet
                cycle_read_values: vec![], // we do not have any yet
            },
            table, // already public input tbh..
            multiplicity,
        });

        GKROpcodeFamilyDecoding {
            pc,
            rs1_idx,
            rs2_idx,
            rd_idx,
            not_rdx0,
            immediate,
            f3,
            mask
        }
        
    }

    pub fn initialise_finalise_inits_and_teardowns(&mut self, ram_challenges: (FEXT, [FEXT; RAM_WIDTH - 1]), lookup_challenges: (FEXT, [FEXT; LOOKUP_WIDTH - 1]), start_time: u64, end_time: u64) -> FullGKRCircuit<F, FEXT, GATE_DEGREE> {
        assert!(end_time < (1<<(TIMESTAMP_LIMB_BITS*2)));
        assert!(end_time > start_time);

        self.ram_challenges = ram_challenges;
        self.lookup_challenges = lookup_challenges;

        // append writeset for init
        // we always range check writeset
        const TIME_MASK: u64 = (1<<TIMESTAMP_LIMB_BITS) - 1;
        let [init_addr_low, init_addr_high] = self.new_u32().0.map(Constraint::from);
        let is_register = Constraint::from(1);
        let [init_time_low, init_time_high] = Register([Num::Constant(F::from_u64_unchecked(start_time & TIME_MASK)), Num::Constant(F::from_u64_unchecked(start_time >> TIMESTAMP_LIMB_BITS))]).0.map(Constraint::from);
        let [init_val_low, init_val_high] = Register([Num::Constant(F::ZERO), Num::Constant(F::ZERO)]).0.map(Constraint::from);
        self.append_readset([init_addr_low.clone(), init_addr_high.clone(), is_register, init_time_low, init_time_high, init_val_low, init_val_high]);

        // append readset for teardown
        // no need to range check readset
        let [teardown_addr_low, teardown_addr_high] = [init_addr_low, init_addr_high];
        let is_register = Constraint::from(1);
        let [teardown_time_low, teardown_time_high] = Register([Num::Constant(F::from_u64_unchecked(end_time & TIME_MASK)), Num::Constant(F::from_u64_unchecked(end_time >> TIMESTAMP_LIMB_BITS))]).0.map(Constraint::from);
        let [teardown_val_low, teardown_val_high] = Register([self.new_variable(), self.new_variable()].map(Num::Var)).0.map(Constraint::from);
        self.append_writeset([teardown_addr_low, teardown_addr_high, is_register, teardown_time_low, teardown_time_high, teardown_val_low, teardown_val_high]);

        // append row-wise init_addr inequality
        // self.enforce_inequality(init_addr1, init_addr2, Boolean::Constant(true));
        todo!(); // gotta do it in the finaliser/composer
        
        // append tableset for setup/multiplicity
        let table = from_fn(|_| self.new_variable()); // from external witness generation
        let multiplicity = self.new_variable(); // from external witness generation
        self.append_tableset(table, multiplicity);

        // store placeholders
        // TODO: we need to add main cycle read_value == teardown_val
        // TODO: we need to add cycle read_addresses == teardown_addr
        self.placeholder_state = Some(GKRPlaceholderState {
            // these values need to go in RAM transcript
            // we make a first pass external witness gen and commitment for them
            memory: GKRPlaceholderMemoryState {
                execute: Boolean::Constant(false), // not present in inits_teardowns
                time: Register([Num::Constant(F::ZERO), Num::Constant(F::ZERO)]), // not present in inits_teardowns
                pc: Register([Num::Constant(F::ZERO), Num::Constant(F::ZERO)]), // not present in inits_teardowns,
                cycle_read_times: [None; MAX_CYCLE_UNIQUE_RAM_ACCESSES], // not present in inits_teardowns,
                cycle_read_values: vec![], // not present in inits_teardowns,
            },
            table, // already public input tbh..
            multiplicity,
        });  

        todo!()
    }

    pub fn finalise(&mut self, pc_next: [Constraint<F>; 2]) -> FullGKRCircuit<F, FEXT, GATE_DEGREE> {
        // pc_next must be already range checked!
        // append writeset for pc/time
        let [state_addr_low, state_addr_high] = [Constraint::from(STATE_PERMUTATION_ADDR_LOW), Constraint::from(0)];
        let is_register = Constraint::from(1);
        let time = self.placeholder_state.as_ref().expect("did not init circuit").memory.time.0.map(Constraint::from);
        let time_bump = [Constraint::from(TIMESTAMP_STEP), Constraint::from(0)];
        let time_ofs_nowrap = [self.new_boolean(), Boolean::Constant(false)];
        let [time_next_low, time_next_high] = self.get_addition::<TIMESTAMP_LIMB_BITS>(time, time_bump, time_ofs_nowrap);
        let [pc_next_low, pc_next_high] = pc_next;
        self.append_writeset([state_addr_low, state_addr_high, is_register, time_next_low, time_next_high, pc_next_low, pc_next_high]);

        // don't forget to assert total number of lookups...
        // assert!(self.total_lookupset_gates.len() * CONST < F::CHARACTERISTICS);
        // generate_ram_and_logup_compression_layers(self)
        todo!()
    }

    // ... elementary gates

    pub fn append_zerocheck(&mut self, expression: Constraint<F>) {
        assert!(expression.degree() <= GATE_DEGREE);
        let lifted_expression = lift_constraint(expression);
        self.total_zerocheck_gates.push(lifted_expression);
    }

    fn append_copygate(&mut self, var: Variable) {
        // assert!(expression.degree() <= GATE_DEGREE);
        // let lifted_expression = lift_constraint(expression);
        self.total_copy_gates.push(var);
    }

    fn append_readset(&mut self, tuple: [Constraint<F>; RAM_WIDTH]) {
        assert!(tuple.iter().all(|con| con.degree() <= GATE_DEGREE));
        let readset_contribution = lift_compress_randomise_tuple(tuple, self.ram_challenges);
        self.total_readset_gates.push(readset_contribution);
    }

    fn append_writeset(&mut self, tuple: [Constraint<F>; RAM_WIDTH]) {
        // WARN: ensure range checked tuple and timestamp inequaity with the read.
        assert!(tuple.iter().all(|con| con.degree() <= GATE_DEGREE));
        let writeset_contribution = lift_compress_randomise_tuple(tuple, self.ram_challenges);
        self.total_writeset_gates.push(writeset_contribution);
    }

    fn append_lookupset(&mut self, tuple: [Constraint<F>; LOOKUP_WIDTH]) {
        assert!(tuple.iter().all(|con| con.degree() <= GATE_DEGREE));
        let lookupset_contribution = lift_compress_randomise_tuple(tuple, self.lookup_challenges);
        self.total_lookupset_gates.push((None, lookupset_contribution));
    }

    fn append_tableset(&mut self, tuple: [Variable; LOOKUP_WIDTH], multiplicity: Variable) {
        let tableset_contribution = lift_compress_randomise_tuple(tuple.map(Constraint::from), self.lookup_challenges);
        let lifted_multiplicity = lift_constraint(Constraint::from(multiplicity));
        self.total_tableset_gates.push((lifted_multiplicity, tableset_contribution));
    }

    // ... basic constraints

    pub fn new_variable(&mut self) -> Variable {
        let var = Variable(self.total_vars);
        self.total_vars += 1;
        var
    }

    pub fn booleancheck(&mut self, expression: Constraint<F>) {
        self.append_zerocheck(expression.clone() * (Constraint::from(1) - expression));
    }

    pub fn rangecheck<const LIMB_BITS: usize>(&mut self, expression: Constraint<F>) {
        let mut tuple = from_fn(|_| Constraint::from(0));
        let table_id = match LIMB_BITS {
            16 => Constraint::from(TableType::U16.to_num()),
            19 => Constraint::from(TableType::U19.to_num()),
            _ => unreachable!()
        };
        tuple[0] = table_id;
        tuple[1] = expression;
        self.append_lookupset(tuple);
    }

    pub fn new_boolean(&mut self) -> Boolean {
        let var = self.new_variable();
        self.booleancheck(Constraint::from(var));
        Boolean::Is(var)
    }

    pub fn new_variable_from_constraint(&mut self, expression: Constraint<F>) -> Variable {
        let var = self.new_variable();
        self.append_zerocheck(Constraint::from(var) - expression);
        var
    }

    pub fn new_u16(&mut self) -> Variable {
        let var = self.new_variable();
        self.rangecheck::<16>(Constraint::from(var));
        var
    }

    pub fn new_u32(&mut self) -> Register<F> {
        let limbs = [self.new_u16(), self.new_u16()].map(Num::Var);
        Register(limbs)
    }

    pub fn read<const READONLY: bool>(&mut self, is_register: Constraint<F>, address: [Constraint<F>; 2]) -> (Register<F>, Register<F>) {
        // NB: no need to range check ReadSet, it's implied by the RAM argument
        // NB: this read will be masked to 1 when cycle's execute==0
        assert!(self.total_cycle_unique_reads < MAX_CYCLE_UNIQUE_RAM_ACCESSES);
        let read_time = Register([self.new_variable(), self.new_variable()].map(Num::Var));
        let read_value = Register([self.new_variable(), self.new_variable()].map(Num::Var));
        self.placeholder_state.as_mut().expect("did not init circuit").memory.cycle_read_times[self.total_cycle_unique_reads] = Some(read_time);
        self.placeholder_state.as_mut().expect("did not init circuit").memory.cycle_read_values.push(read_value.clone());
        self.append_readset([address[0].clone(), address[1].clone(), is_register.clone(), Constraint::from(read_time.0[0]), Constraint::from(read_time.0[1]), Constraint::from(read_value.0[0]), Constraint::from(read_value.0[1])]);
        self.total_cycle_unique_reads += 1;

        if READONLY {
            self.write::<false>(Some(read_time), is_register, address, read_value.0.map(Constraint::from));
        }
        
        (read_time, read_value)
    }

    fn _read_from_ptr<const N: usize>(&mut self, indexes: [Constraint<F>; N]) -> [Register<F>; N] {
        todo!()
    }

    pub fn write<const WRITEONLY: bool>(&mut self, read_time: Option<Register<F>>, is_register: Constraint<F>, address: [Constraint<F>; 2], write_value: [Constraint<F>; 2]) {
        // WARN: ensure addr/value are range checked and masked (eg. if rd==x0 then value must be 0)
        // WARN: ensure writes are performed in the same order as reads
        // NB: this read will be masked to 1 when cycle's execute==0
        let read_time = match (WRITEONLY, read_time) {
            (true, None) => self.read::<false>(is_register.clone(), address.clone()).0,
            (false, Some(read_time)) => read_time,
            _ => unreachable!()
        };

        assert!(self.total_cycle_unique_writes < MAX_CYCLE_UNIQUE_RAM_ACCESSES);
        let cycle_time = self.placeholder_state.as_ref().expect("did not init circuit").memory.time;
        let cycle_time_offset = self.total_cycle_unique_writes;
        let write_time = [Constraint::from(cycle_time.0[0]) + Constraint::from(cycle_time_offset as u64), Constraint::from(cycle_time.0[1])];  
        self.append_writeset([address[0].clone(), address[1].clone(), is_register, write_time[0].clone(), write_time[1].clone(), write_value[0].clone(), write_value[1].clone()]);
        self.total_cycle_unique_writes += 1;

        // timestamp inequality: read_time < write_time (when exe==1, checked later)
        self.enforce_inequality::<TIMESTAMP_LIMB_BITS>(read_time.0.map(Constraint::from), write_time, Boolean::Constant(true));
    }

    fn _write_to_ptr<const N: usize>(&mut self, indexes: [Constraint<F>; N]) -> [Register<F>; N] {
        todo!()
    }

    // .. advanced constraints

    pub fn enforce_constraint(&mut self, expression: Constraint<F>) {
        self.append_zerocheck(expression);
    }

    pub fn enforce_addition<const LIMB_BITS: usize>(&mut self, a: [Constraint<F>; 2], b: [Constraint<F>; 2], c: [Constraint<F>; 2], ofs: [Constraint<F>; 2]) {
        // assume all range checked
        let [a_low, a_high] = a;
        let [b_low, b_high] = b;
        let [c_low, c_high] = c;
        let [of_low, of_high] = ofs;
        // aL + bL == cL + 2^LIMB_WIDTH ofL
        // aH + bH + ofL == cH + 2^LIMB_WIDTH ofH
        self.append_zerocheck(a_low + b_low - c_low - Term::from(1 << LIMB_BITS)*of_low.clone());
        self.append_zerocheck(a_high + b_high + of_low - c_high - Term::from(1 << LIMB_BITS)*of_high);
    }

    pub fn enforce_subtraction<const LIMB_BITS: usize>(&mut self, a: [Constraint<F>; 2], b: [Constraint<F>; 2], c: [Constraint<F>; 2], ofs: [Constraint<F>; 2]) {
        // assume all range checked
        // A - B == C - 2^(LIMB_WIDTH*2) OF --> C + B == A + 2^(LIMB_WIDTH*2) OF
        self.enforce_addition::<LIMB_BITS>(c, b, a, ofs);
    }

    pub fn get_addition<const LIMB_BITS: usize>(&mut self, a: [Constraint<F>; 2], b: [Constraint<F>; 2], ofs: [Boolean; 2]) -> [Constraint<F>; 2] {
        let [a_low, a_high] = a;
        let [b_low, b_high] = b;
        let [of_low, of_high] = ofs.map(Term::from);

        // aL + bL == cL + 2^LIMB_WIDTH ofL
        // aH + bH + ofL == cH + 2^LIMB_WIDTH ofH
        // we're using the 1-less-chunk-trick here, see Z3 script
        let c_low = a_low + b_low - Term::from(1 << LIMB_BITS)*of_low;
        let c_high = a_high + b_high + of_low - Term::from(1 << LIMB_BITS)*of_high;
        self.rangecheck::<LIMB_BITS>(c_low.clone());
        self.rangecheck::<LIMB_BITS>(c_high.clone());
        [c_low, c_high]
    }

    pub fn get_subtraction<const LIMB_BITS: usize>(&mut self, a: [Constraint<F>; 2], b: [Constraint<F>; 2], ofs: [Boolean; 2]) -> [Constraint<F>; 2] {
        let [a_low, a_high] = a;
        let [b_low, b_high] = b;
        let [of_low, of_high] = ofs.map(Term::from);

        // aL - bL == cL - 2^19 ofL
        // aH - bH - ofL == cH - 2^19 ofH
        // we're using the 1-less-chunk-trick here, see Z3 script
        let c_low = a_low - b_low + Term::from(1<<19)*of_low;
        let c_high = a_high - b_high - of_low + Term::from(1<<19)*of_high;
        self.rangecheck::<LIMB_BITS>(c_low.clone());
        self.rangecheck::<LIMB_BITS>(c_high.clone());
        [c_low, c_high]
    }

    pub fn enforce_inequality<const LIMB_BITS: usize>(&mut self, a: [Constraint<F>; 2], b: [Constraint<F>; 2], is_lessthan: Boolean) {
        // set less_than to constant 1 if you want strict (a < b)
        let of_low = self.new_boolean();
        let _output = self.get_subtraction::<LIMB_BITS>(a, b, [of_low, is_lessthan]);
    }

    // ensure the addition of word limbs cannot overflow the field
    pub fn is_zero<const N: usize>(&mut self, word_limbs: [Constraint<F>; N]) -> Boolean {
        // either there is an inverse, or it's zero
        // (1 - word*inv)*word == 0
        let inv = Term::from(self.new_variable());
        let compressed_word = word_limbs.into_iter().reduce(|acc, con| acc + con).unwrap();
        let zflag = self.new_variable_from_constraint(Constraint::from(1) - compressed_word.clone() * inv);
        self.append_zerocheck(Term::from(zflag) * compressed_word);
        Boolean::Is(zflag)
    }

    pub fn is_eq<const LIMB_BITS: usize>(&mut self, a: [Constraint<F>; 2], b: [Constraint<F>; 2]) -> Boolean {
        let ofs = [self.new_boolean(), self.new_boolean()];
        let c = self.get_subtraction::<LIMB_BITS>(a, b, ofs);
        self.is_zero(c)
    }

    pub fn decompose_bitmask<const N: usize>(&mut self, expression: Constraint<F>) -> [Boolean; N] {
        let bitmask = from_fn(|_| self.new_boolean());
        let mut composition = Constraint::from(0);
        for i in 0..N {
            composition += Term::from(1 << i)*Term::from(bitmask[i]);
        }
        self.append_zerocheck(composition - expression);
        bitmask
    }









}





fn generate_ram_and_logup_compression_layers<F: PrimeField, FEXT: FieldExtension<F>, const GATE_DEGREE: usize, const LOG2_WITNESS_SIZE: usize>(base_layer: GKRLayerCircuit<F, FEXT, GATE_DEGREE>) -> Vec<Vec<(Constraint<FEXT>, Constraint<FEXT>)>> {
    // 1. compress gates as much as possible
    // 2. compile dense/sparse selectors
    // 3. (optional) apply masking

    // first we try to compress first layer a bit more (to optimise), [DENSE WIRES]
    // then we can try to go ahead and derive next layers for the cycle, [SPARSE WIRES]
    // until cycle is complete, then there is masking
    // then we can continue as usual except we no longer worry about masking [SPARSE WIRES]

    // ideally each time we move onto a new layer, we have computed wiring too..
    // ... so technically now we need to compute wires for the first layer
    dbg!(base_layer.total_copy_gates.len());
    dbg!(base_layer.total_readset_gates.len());
    dbg!(base_layer.total_writeset_gates.len());
    dbg!(base_layer.total_tableset_gates.len());
    dbg!(base_layer.total_lookupset_gates.len());
    dbg!(base_layer.total_zerocheck_gates.len());
    todo!();
}


fn compress_multiplicative_contributions<F: PrimeField, FEXT: FieldExtension<F>, const GATE_DEGREE: usize>(gates: Vec<Constraint<FEXT>>) -> Vec<Constraint<FEXT>> {
    assert!(GATE_DEGREE == 2); // for now we keep it simple
    let mut deg0_gate = Constraint::from_field(FEXT::ONE);
    let mut deg1_gates = Vec::with_capacity(gates.len());
    let mut deg2_gates = Vec::with_capacity(gates.len());
    for gate in gates {
        match gate.degree() {
            0 => deg0_gate *= gate,
            1 => deg1_gates.push(gate),
            2 => deg2_gates.push(gate),
            _ => unreachable!()
        }
    }

    let (deg1_pairs, deg1_tail) = deg1_gates.as_chunks::<2>();
    for [left, right] in deg1_pairs {
        deg2_gates.push(left.clone() * right.clone());
    }
    for gate in deg1_tail {
        deg2_gates.push(gate.clone());
    }
    if let Some(gate) = deg2_gates.first_mut() {
        gate.mul_assign(deg0_gate);
    }
    deg2_gates
}

fn compress_rational_additive_contributions<F: PrimeField, FEXT: FieldExtension<F>, const GATE_DEGREE: usize>(gates: Vec<(Constraint<FEXT>, Constraint<FEXT>)>) -> Vec<(Constraint<FEXT>, Constraint<FEXT>)> {
    assert!(GATE_DEGREE == 2); // for now we keep it simple
    let mut deg0_gate = Constraint::from_field(FEXT::ONE);
    let mut deg1_gates = Vec::with_capacity(gates.len());
    let mut deg2_gates = Vec::with_capacity(gates.len());
    for (gate_num, gate_den) in gates {
        assert!(gate_num.degree() <= gate_den.degree());
        match gate_den.degree() {
            0 => deg0_gate *= {
                let mut num = gate_num.as_constant();
                let den = gate_den.as_constant();
                num.mul_assign(&den.inverse().unwrap());
                Constraint::from_field(num)
            },
            1 => deg1_gates.push((gate_num, gate_den)),
            2 => deg2_gates.push((gate_num, gate_den)),
            _ => unreachable!()
        }
    }

    let (deg1_pairs, deg1_tail) = deg1_gates.as_chunks::<2>();
    for [(left_num, left_den), (right_num, right_den)] in deg1_pairs {
        // a1/b1 + a2/b2 == (a1b2 + a2b1)/b1b2
        let gate_num = left_num.clone() * right_den.clone() + right_num.clone() * left_den.clone();
        let gate_den = left_den.clone() * right_den.clone();
        deg2_gates.push((gate_num, gate_den));
    }
    for gate in deg1_tail {
        deg2_gates.push(gate.clone());
    }
    if let Some((gate_num, _gate_den)) = deg2_gates.first_mut() {
        gate_num.mul_assign(deg0_gate);
    }
    deg2_gates
}