use std::{ptr::addr_of_mut, mem::offset_of};

use common_constants::TimestampScalar;
use dynasmrt::{dynasm, x64, DynamicLabel, DynasmApi, DynasmLabelApi};
use riscv_decode::Instruction;

use crate::vm::*;

#[cfg(test)]
mod tests;

// Register use and mapping

// - x10-x15 (RV) are stored in r10-r15 (X86)
// - RDI holds a pointer to backing array for snapshot itself
// - RSI will contain a pointer to the special unsized(!) structure that begins with register timestamps, then counters, then unsized RAM backing slice
// - r8 holds a timestamp (0 mod 4 in the cycle)
// - r9 holds a number of elements in the snapshot

// The prologue saves all callee-saved registers
// This allows us to use all but rbp and rsp
// Using rbp would mess with debuggers
// Using rsp would cause signal handlers to write to some random location
// instead of the stack.
macro_rules! prologue {
    ($ops:ident) => {
        dynasm!($ops
            ; push rbp
            ; mov rbp, rsp

            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            ; sub rsp, 8
        )
    };
}

macro_rules! epilogue {
    ($ops:ident) => {
        dynasm!($ops
            ; add rsp, 8
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; leave
            ; ret

            // handler for full trace chunk
            ; ->trace_buffer_full:
            ; xor r9, r9
            ; push rax
            ; push rcx
            ; push rdx
            ;; before_call!($ops)
            ; mov rax, QWORD Context::<N, [Register; 1 << 28]>::receive_trace as _
            ; mov rdi, rsi
            ; mov rsi, r9
            ; sub rsp, 8
            ; call rax
            ; add rsp, 8
            ;; after_call!($ops)
            ; pop rdx
            ; pop rcx
            ; pop rax
            ; ret
        )
    };
}

macro_rules! before_call {
    ($ops:ident) => {
        dynasm!($ops
            ; push rsi
            ; push rdi
            ; push r8
            ; push r9
            ; push r10
            ; push r11

            ; sub rsp, 16 * 16
            ; movdqu [rsp + 0], xmm0
            ; movdqu [rsp + 16], xmm1
            ; movdqu [rsp + 32], xmm2
            ; movdqu [rsp + 48], xmm3
            ; movdqu [rsp + 64], xmm4
            ; movdqu [rsp + 80], xmm5
            ; movdqu [rsp + 96], xmm6
            ; movdqu [rsp + 112], xmm7
            ; movdqu [rsp + 128], xmm8
            ; movdqu [rsp + 144], xmm9
            ; movdqu [rsp + 160], xmm10
            ; movdqu [rsp + 176], xmm11
            ; movdqu [rsp + 192], xmm12
            ; movdqu [rsp + 208], xmm13
            ; movdqu [rsp + 224], xmm14
            ; movdqu [rsp + 240], xmm15
        )
    }
}

macro_rules! after_call {
    ($ops:ident) => {
        dynasm!($ops
            ; movdqu  xmm0, [rsp + 0]
            ; movdqu  xmm1, [rsp + 16]
            ; movdqu  xmm2, [rsp + 32]
            ; movdqu  xmm3, [rsp + 48]
            ; movdqu  xmm4, [rsp + 64]
            ; movdqu  xmm5, [rsp + 80]
            ; movdqu  xmm6, [rsp + 96]
            ; movdqu  xmm7, [rsp + 112]
            ; movdqu  xmm8, [rsp + 128]
            ; movdqu  xmm9, [rsp + 144]
            ; movdqu  xmm10, [rsp + 160]
            ; movdqu  xmm11, [rsp + 176]
            ; movdqu  xmm12, [rsp + 192]
            ; movdqu  xmm13, [rsp + 208]
            ; movdqu  xmm14, [rsp + 224]
            ; movdqu  xmm15, [rsp + 240]
            ; add rsp, 16 * 16

            ; pop r11
            ; pop r10
            ; pop r9
            ; pop r8
            ; pop rdi
            ; pop rsi
        )
    }
}

// The following functions are used to specify a mapping
// between RISC-V and x86 registers
//
// For registers with no dedicated x86 register,
// register writes go via rax and reads via rdx
// rcx also doesn't contain a register because it must be used for bitshifts
//
// x10 - x15 are assiged to r10 - r15
// rbx is for x9

const SCRATCH_REGISTER: u8 = x64::Rq::RCX as u8;

fn rv_to_gpr(x: u32) -> Option<u8> {
    use x64::Rq::*;
    assert!(x < 32);

    Some(
        (match x {
            9 => RBX,
            10 => R10,
            11 => R11,
            12 => R12,
            13 => R13,
            14 => R14,
            15 => R15,
            _ => return None,
        }) as u8,
    )
}

fn destination_gpr(x: u32) -> u8 {
    rv_to_gpr(x).unwrap_or(x64::Rq::RAX as u8)
}

fn store_result(ops: &mut x64::Assembler, x: u32) {
    assert!(x != 0);
    assert!(x < 32);

    if rv_to_gpr(x).is_none() {
        let x = x as u8;
        let high_or_low = x & 1;
        let register = x >> 1;
        dynasm!(ops
            ; pinsrd Rx(register), eax, high_or_low as i8
        )
    }
}

/// Returns the general purpose register that now holds the value of the
/// RISC-V register `x`.
/// Do not use in quick succession; the first value will get overwritten.
fn load(ops: &mut x64::Assembler, x: u32) -> u8 {
    rv_to_gpr(x).unwrap_or_else(|| {
        if x == 0 {
            dynasm!(ops
                ; xor edx, edx
            );
        } else {
            let x = x as u8;
            let high_or_low = x & 1;
            let register = x >> 1;
            dynasm!(ops
                ; pextrd edx, Rx(register), high_or_low as i8
            );
        }

        x64::Rq::RDX as u8
    })
}

/// Loads the RISC-V register `x` into the specified register.
fn load_into(ops: &mut x64::Assembler, x: u32, destination: u8) {
    if let Some(gpr) = rv_to_gpr(x) {
        if destination != gpr {
            dynasm!(ops
                ; mov Rd(destination), Rd(gpr)
            );
        }
    } else {
        if x == 0 {
            dynasm!(ops
                ; xor Rd(destination), Rd(destination)
            );
        } else {
            let x = x as u8;
            let high_or_low = x & 1;
            let register = x >> 1;
            dynasm!(ops
                ; pextrd Rd(destination), Rx(register), high_or_low as i8
            );
        }
    }
}

fn load_abelian(ops: &mut x64::Assembler, x: u32, y: u32, destination: u8) -> u8 {
    let a = rv_to_gpr(x);
    let b = rv_to_gpr(y);
    if a == Some(destination) {
        load(ops, y)
    } else if b == Some(destination) {
        load(ops, x)
    } else {
        load_into(ops, x, destination);
        load(ops, y)
    }
}

macro_rules! print_registers {
    ($ops:ident) => {
        dynasm!($ops
            ; sub rsp, 32 * 4
            ; mov DWORD [rsp], 0
        );
        for i in 1..32 {
            let reg = load(&mut $ops, i);
            dynasm!($ops
                ; mov [rsp + 4 * i as i32], Rd(reg)
            );
        }

        dynasm!($ops
            ; mov rcx, rsp

            ; push rdi
            ; push rsi
            ; push r8
            ; push r9

            ; mov rax, QWORD print_registers as _
            ; mov rdi, rcx
            ; call rax

            ; pop r9
            ; pop r8
            ; pop rsi
            ; pop rdi
        );

        for i in 1..32 {
            let out = destination_gpr(i);
            dynasm!($ops
                ; mov Rd(out), [rsp + 4 * i as i32]
            );
            store_result(&mut $ops, i);
        }
        dynasm!($ops
            ; add rsp, 32 * 4
        );
    };
}

// We will measure trace chunk in a number of memory accesses and not in a fixed number of cycles that did pass between them
const TRACE_CHUNK_LEN: usize = 1 << 20;

macro_rules! increment_trace {
    ($ops:ident) => {
        dynasm!($ops
            ; inc r9
            ; cmp r9, TRACE_CHUNK_LEN as _
            ; jne >skip
            ; call ->trace_buffer_full
            ; skip:
        );
    };
}

macro_rules! pre_bump_timestamp_and_touch {
    ($ops:ident, $d:expr, $r:expr) => {
        dynasm!($ops
            ; add r8, $d
            ; mov [rsi + 8*($r as i32)], r8
        );
    };
}

macro_rules! touch_register_and_increment_timestamp {
    ($ops:ident, $r:expr) => {
        dynasm!($ops
            ; mov [rsi + 8*($r as i32)], r8
            ; inc r8
        );
    };
}

macro_rules! touch_register_and_bump_timestamp {
    ($ops:ident, $r:expr, $d:expr) => {
        dynasm!($ops
            ; mov [rsi + 8*($r as i32)], r8
            ; add r8, $d
        );
    };
}

macro_rules! bump_timestamp {
    ($ops:ident, $r:expr) => {
        dynasm!($ops
            ; add r8, $r
        );
    };
}

// macro_rules! touch_register {
//     ($ops:ident, $r:expr) => {
//         dynasm!($ops
//             ; mov [r8 + 4*r9], Rd($r)
//             ;; increment_trace!($ops)
//         );
//     };
// }

// macro_rules! trace_register {
//     ($ops:ident, $r:expr) => {
//         dynasm!($ops
//             ; mov [r8 + 4*r9], Rd($r)
//             ;; increment_trace!($ops)
//         );
//     };
// }

// macro_rules! trace_zero {
//     ($ops:ident) => {
//          dynasm!($ops
//             ; mov DWORD [r8 + 4*r9], 0
//             ;; increment_trace!($ops)
//         );
//     };
// }

macro_rules! emit_runtime_error {
    ($ops:ident) => {
        dynasm!($ops
            ; jmp ->exit_with_error
        )
    };
}

pub fn run_alternative_simulator<'a, N: NonDeterminismCSRSource<RamWithRomRegion< {common_constants::rom::ROM_SECOND_WORD_BITS } >>>(
    program: &[u32],
    non_determinism_source: &'a mut N,
    initial_memory: &[u32],
) {
    assert!(initial_memory.len() <= common_constants::rom::ROM_WORD_SIZE);
    let mut ops = x64::Assembler::new().unwrap();
    let start = ops.offset();

    dynasm!(ops
        ; ->start:
        ;; prologue!(ops)
        ; vzeroall
        ; xor r10, r10
        ; xor r11, r11
        ; xor r12, r12
        ; xor r13, r13
        ; xor r14, r14
        ; xor r15, r15

        ; mov r8, INITIAL_TIMESTAMP as _
        ; xor r9, r9
    );

    // Static jump targets for JAL and branch instructions
    let instruction_labels = (0..program.len())
        .map(|_| ops.new_dynamic_label())
        .collect::<Vec<_>>();

    // Jump target array for Jalr
    // Records the position of each RISC-V instruction relative to the start
    let mut jump_offsets = vec![];

    for (i, raw_instruction) in program.iter().enumerate() {
        let pc = i as u32 * 4;

        dynasm!(ops
            ; =>instruction_labels[i]
        );
        jump_offsets.push(ops.offset().0);

        let Ok(instruction) = riscv_decode::decode(*raw_instruction) else {
            emit_runtime_error!(ops);
            continue;
        };

        let mut instruction_emitted = false;

        // Pure instructions
        if matches!(
            instruction,
            Instruction::Addi(_)
                | Instruction::Andi(_)
                | Instruction::Ori(_)
                | Instruction::Xori(_)
                | Instruction::Slti(_)
                | Instruction::Sltiu(_)
                | Instruction::Slli(_)
                | Instruction::Srli(_)
                | Instruction::Srai(_)
                | Instruction::Lui(_)
                | Instruction::Auipc(_)
                | Instruction::Add(_)
                | Instruction::Sub(_)
                | Instruction::Slt(_)
                | Instruction::Sltu(_)
                | Instruction::And(_)
                | Instruction::Or(_)
                | Instruction::Xor(_)
                | Instruction::Sll(_)
                | Instruction::Srl(_)
                | Instruction::Sra(_)
                | Instruction::Lb(_)
                | Instruction::Lbu(_)
                | Instruction::Lh(_)
                | Instruction::Lhu(_)
                | Instruction::Lw(_)
                | Instruction::Mul(_)
                | Instruction::Mulh(_)
                | Instruction::Mulhu(_)
                | Instruction::Mulhsu(_)
        ) {
            let rd = (raw_instruction >> 7) & 0x1F;
            let out = destination_gpr(rd);
            // Instructions that just compute a result are NOPs if they write to x0, and formally touch x0 twice on read
            if rd == 0 {
                pre_bump_timestamp_and_touch!(ops, 2, 0);
                bump_timestamp!(ops, 2);
                continue;
            }

            match instruction {
                // Arithmetic
                Instruction::Addi(parts) => {
                    let source = load(&mut ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; lea Rd(out), [Rd(source) + sign_extend::<12>(parts.imm())]
                    );
                }
                Instruction::Andi(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; and Rd(out), sign_extend::<12>(parts.imm())
                    );
                }
                Instruction::Ori(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; or Rd(out), sign_extend::<12>(parts.imm())
                    );
                }
                Instruction::Xori(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; xor Rd(out), sign_extend::<12>(parts.imm())
                    );
                }
                Instruction::Slti(parts) => {
                    let source = load(&mut ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; cmp Rd(source), sign_extend::<12>(parts.imm())
                        ; setl Rb(out)
                        ; movzx Rd(out), Rb(out)
                    );
                }
                Instruction::Sltiu(parts) => {
                    let source = load(&mut ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; cmp Rd(source), sign_extend::<12>(parts.imm())
                        ; setb Rb(out)
                        ; movzx Rd(out), Rb(out)
                    );
                }
                Instruction::Slli(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; shl Rd(out), parts.shamt() as i8
                    );
                }
                Instruction::Srli(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; shr Rd(out), parts.shamt() as i8
                    );
                }
                Instruction::Srai(parts) => {
                    load_into(&mut ops, parts.rs1(), out);
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; sar Rd(out), parts.shamt() as i8
                    );
                }
                Instruction::Lui(parts) => {
                    pre_bump_timestamp_and_touch!(ops, 2, 0);
                    dynasm!(ops
                        ; mov Rd(out), parts.imm() as i32
                    );
                }
                Instruction::Auipc(parts) => {
                    pre_bump_timestamp_and_touch!(ops, 2, 0);
                    dynasm!(ops
                        ; mov Rd(out), (pc + parts.imm()) as i32
                    );
                }
                Instruction::Add(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    let other = load_abelian(&mut ops, parts.rs1(), parts.rs2(), out);
                    dynasm!(ops
                        ; add Rd(out), Rd(other)
                    );
                }
                Instruction::Sub(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), SCRATCH_REGISTER);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; sub Rd(out), Rd(SCRATCH_REGISTER)
                    );
                }
                Instruction::Slt(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), SCRATCH_REGISTER);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; cmp Rd(out), Rd(SCRATCH_REGISTER)
                        ; setl Rb(out)
                        ; movzx Rd(out), Rb(out)
                    );
                }
                Instruction::Sltu(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), SCRATCH_REGISTER);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; cmp Rd(out), Rd(SCRATCH_REGISTER)
                        ; setb Rb(out)
                        ; movzx Rd(out), Rb(out)
                    );
                }
                Instruction::And(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    let other = load_abelian(&mut ops, parts.rs1(), parts.rs2(), out);
                    dynasm!(ops
                        ; and Rd(out), Rd(other)
                    );
                }
                Instruction::Or(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    let other = load_abelian(&mut ops, parts.rs1(), parts.rs2(), out);
                    dynasm!(ops
                        ; or Rd(out), Rd(other)
                    );
                }
                Instruction::Xor(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    let other = load_abelian(&mut ops, parts.rs1(), parts.rs2(), out);
                    dynasm!(ops
                        ; xor Rd(out), Rd(other)
                    );
                }
                Instruction::Sll(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), x64::Rq::RCX as u8);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; and rcx, 0x1f
                        ; shl Rd(out), cl
                    );
                }
                Instruction::Srl(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), x64::Rq::RCX as u8);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; and rcx, 0x1f
                        ; shr Rd(out), cl
                    );
                }
                Instruction::Sra(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), x64::Rq::RCX as u8);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; and rcx, 0x1f
                        ; sar Rd(out), cl
                    );
                }

                // Loads
                Instruction::Lb(parts) => {
                    emit_runtime_error!(ops);
                    // let address = load(&mut ops, parts.rs1());
                    // dynasm!(ops
                    //     ; movsx Rq(out), Rd(address)
                    //     ; movsx Rd(out), BYTE [rsi + Rq(out) + sign_extend::<12>(parts.imm())]
                    // );
                }
                Instruction::Lbu(parts) => {
                    emit_runtime_error!(ops);
                    // let address = load(&mut ops, parts.rs1());
                    // dynasm!(ops
                    //     ; movsx Rq(out), Rd(address)
                    //     ; movzx Rd(out), BYTE [rsi + Rq(out) + sign_extend::<12>(parts.imm())]
                    // );
                }
                Instruction::Lh(parts) => {
                    emit_runtime_error!(ops);
                    // // TODO: exception on misalignment
                    // let address = load(&mut ops, parts.rs1());
                    // dynasm!(ops
                    //     ; movsx Rq(out), Rd(address)
                    //     ; movsx Rd(out), WORD [rsi + Rq(out) + sign_extend::<12>(parts.imm())]
                    // );
                }
                Instruction::Lhu(parts) => {
                    emit_runtime_error!(ops);
                    // // TODO: exception on misalignment
                    // let address = load(&mut ops, parts.rs1());
                    // dynasm!(ops
                    //     ; movsx Rq(out), Rd(address)
                    //     ; movzx Rd(out), WORD [rsi + Rq(out) + sign_extend::<12>(parts.imm())]
                    // );
                }
                Instruction::Lw(parts) => {
                    // TODO: exception on misalignment
                    let address = load(&mut ops, parts.rs1());
                    dynasm!(ops
                        ; movsx Rq(SCRATCH_REGISTER), Rd(address)
                    );
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    dynasm!(ops
                        ; mov Rd(out), [rsi + (MEM_BACKING_OFFSET as i32) + 8 + 4 * Rq(SCRATCH_REGISTER) + 4 * sign_extend::<12>(parts.imm())] // load old value into destination
                        ; mov rdx, [rsi + (MEM_BACKING_OFFSET as i32) + 4 * Rq(SCRATCH_REGISTER) + 4 * sign_extend::<12>(parts.imm())] // reuse RDX for read timestamp
                        ; mov [rsi + (MEM_BACKING_OFFSET as i32) + 4 * Rq(SCRATCH_REGISTER) + 4 * sign_extend::<12>(parts.imm())], r8 // update timestamp
                        ; mov [rdi + r9 * 4], Rd(out) // write value into trace
                        ; mov [rdi + r9 * 8 + (TRACE_CHUNK_TIMESTAMPS_OFFSET as i32)], rdx // write old value into trace
                    );
                    bump_timestamp!(ops, 1);
                    increment_trace!(ops);
                }

                // Multiplication
                Instruction::Mul(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    let other = load_abelian(&mut ops, parts.rs1(), parts.rs2(), out);
                    dynasm!(ops
                        ; imul Rd(out), Rd(other)
                    );
                }
                Instruction::Mulh(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs1(), x64::Rq::RAX as u8);
                    let other = load(&mut ops, parts.rs2());
                    dynasm!(ops
                        ; imul Rd(other)
                    );
                    if out != x64::Rq::RDX as u8 {
                        dynasm!(ops
                            ; mov Rd(out), edx
                        );
                    }
                }
                Instruction::Mulhu(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs1(), x64::Rq::RAX as u8);
                    let other = load(&mut ops, parts.rs2());
                    dynasm!(ops
                        ; mul Rd(other)
                    );
                    if out != x64::Rq::RDX as u8 {
                        dynasm!(ops
                            ; mov Rd(out), edx
                        );
                    }
                }
                Instruction::Mulhsu(parts) => {
                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());
                    load_into(&mut ops, parts.rs2(), SCRATCH_REGISTER);
                    load_into(&mut ops, parts.rs1(), out);
                    dynasm!(ops
                        ; movsx Rq(out), Rd(out)
                        ; imul Rq(out), Rq(SCRATCH_REGISTER)
                        ; shr Rq(out), 32
                    );
                }
                _ => unreachable!(),
            }

            touch_register_and_bump_timestamp!(ops, rd, 2);
            // trace_register!(ops, out);
            store_result(&mut ops, rd);
            continue;
        }

        match instruction {
            // Control transfer instructions
            Instruction::Jal(parts) => {
                let rd = (raw_instruction >> 7) & 0x1F;
                let out = destination_gpr(rd);
                // No reads (so read x0 twice)
                if rd != 0 {
                    pre_bump_timestamp_and_touch!(ops, 1, 0);
                    dynasm!(ops
                        ; mov Rd(out), (pc + 4) as i32
                    );
                    store_result(&mut ops, rd);
                    pre_bump_timestamp_and_touch!(ops, 1, rd);
                } else {
                    pre_bump_timestamp_and_touch!(ops, 2, 0);
                }

                bump_timestamp!(ops, 2);

                let offset = sign_extend::<21>(parts.imm());
                let jump_target = pc as i32 + offset;
                if offset == 0 {
                    // An infinite loop is used to signal end of execution
                    dynasm!(ops
                        ; jmp ->quit
                    );
                } else if jump_target % 4 != 0 {
                    panic!("Unaligned just destination");
                    // emit_runtime_error!(ops)
                } else {
                    if let Some(&label) = instruction_labels.get((jump_target / 4) as usize) {
                        dynasm!(ops
                            ; jmp =>label
                        );
                    } else {
                        panic!("Unknown just destination");
                        // emit_runtime_error!(ops)
                    }
                }
            }
            Instruction::Jalr(parts) => {
                let rd = (raw_instruction >> 7) & 0x1F;
                let out = destination_gpr(rd);
                let offset = sign_extend::<12>(parts.imm());
                touch_register_and_increment_timestamp!(ops, parts.rs1());
                load_into(&mut ops, parts.rs1(), SCRATCH_REGISTER);
                dynasm!(ops
                    ; add Rd(SCRATCH_REGISTER), offset
                    // Must be aligned to an instruction but no need to test the least significant bit,
                    // as it is set to zero according to the specification
                    ; test Rd(SCRATCH_REGISTER), 2
                    ; jnz >misaligned
                    ; shr Rd(SCRATCH_REGISTER), 2
                    ; lea rdx, [->jump_offsets]
                    ; mov rax, [rdx + Rq(SCRATCH_REGISTER) * 8]
                    ; lea rdx, [->start]
                    ; add rdx, rax
                );

                // Return address may not be written into register before jump target is computed,
                // otherwise it could affect the jump target.
                if rd != 0 {
                    touch_register_and_increment_timestamp!(ops, 0);
                    dynasm!(ops
                        ; mov Rd(out), (pc + 4) as i32
                    );
                    touch_register_and_bump_timestamp!(ops, rd, 2);
                    store_result(&mut ops, rd);
                } else {
                    pre_bump_timestamp_and_touch!(ops, 1, 0);
                    bump_timestamp!(ops, 2);
                }

                dynasm!(ops
                    ; jmp rdx
                    ; misaligned:
                    ;; emit_runtime_error!(ops)
                );
            }
            Instruction::Beq(parts)
            | Instruction::Bne(parts)
            | Instruction::Blt(parts)
            | Instruction::Bltu(parts)
            | Instruction::Bge(parts)
            | Instruction::Bgeu(parts) => {
                let jump_target = pc as i32 + sign_extend::<13>(parts.imm());
                if jump_target % 4 != 0 {
                    panic!("Unaligned just destination");
                    // emit_runtime_error!(ops);
                } else {
                    let a = load(&mut ops, parts.rs1());
                    load_into(&mut ops, parts.rs2(), SCRATCH_REGISTER);

                    touch_register_and_increment_timestamp!(ops, parts.rs1());
                    touch_register_and_increment_timestamp!(ops, parts.rs2());

                    touch_register_and_bump_timestamp!(ops, 0, 2);

                    if let Some(&label) = instruction_labels.get((jump_target / 4) as usize) {
                        dynasm!(ops
                            ; cmp Rd(a), Rd(SCRATCH_REGISTER)
                        );
                        match instruction {
                            Instruction::Beq(_) => {
                                dynasm!(ops
                                    ; je =>label
                                );
                            }
                            Instruction::Bne(_) => {
                                dynasm!(ops
                                    ; jne =>label
                                );
                            }
                            Instruction::Blt(_) => {
                                dynasm!(ops
                                    ; jl =>label
                                );
                            }
                            Instruction::Bltu(_) => {
                                dynasm!(ops
                                    ; jb =>label
                                );
                            }
                            Instruction::Bge(_) => {
                                dynasm!(ops
                                    ; jge =>label
                                );
                            }
                            Instruction::Bgeu(_) => {
                                dynasm!(ops
                                    ; jae =>label
                                );
                            }
                            _ => unreachable!(),
                        }
                    } else {
                        emit_runtime_error!(ops)
                    }

                    // print_registers!(ops);
                }
            }

            // Stores
            Instruction::Sb(parts) => {
                emit_runtime_error!(ops);
                // let address = load(&mut ops, parts.rs1());
                // dynasm!(ops
                //     ; movsx Rq(SCRATCH_REGISTER), Rd(address)
                // );
                // let value = load(&mut ops, parts.rs2());
                // dynasm!(ops
                //     ; mov [rsi + Rq(SCRATCH_REGISTER) + sign_extend::<12>(parts.imm())], Rb(value)
                // );
                // trace_zero!(ops);
            }
            Instruction::Sh(parts) => {
                emit_runtime_error!(ops);
                // // TODO: exception on misalignment
                // let address = load(&mut ops, parts.rs1());
                // dynasm!(ops
                //     ; movsx Rq(SCRATCH_REGISTER), Rd(address)
                // );
                // let value = load(&mut ops, parts.rs2());
                // dynasm!(ops
                //     ; mov [rsi + Rq(SCRATCH_REGISTER) + sign_extend::<12>(parts.imm())], Rw(value)
                // );
                // trace_zero!(ops);
            }
            Instruction::Sw(parts) => {
                // TODO: exception on misalignment
                let address = load(&mut ops, parts.rs1());
                dynasm!(ops
                    ; movsx Rq(SCRATCH_REGISTER), Rd(address)
                );
                let value = load(&mut ops, parts.rs2());
                touch_register_and_increment_timestamp!(ops, parts.rs1());
                touch_register_and_increment_timestamp!(ops, parts.rs2());
                dynasm!(ops
                    // this sequence of operations is: read old value and timestamp, save it, write new value and timestamp
                    ; mov eax, [rsi + (MEM_BACKING_OFFSET as i32) + 8 + 4 * Rq(SCRATCH_REGISTER) + 4 * sign_extend::<12>(parts.imm())] // load old value into RAX
                    ; mov [rsi + (MEM_BACKING_OFFSET as i32) + 8 + 4 * Rq(SCRATCH_REGISTER) + 4 * sign_extend::<12>(parts.imm())], Rd(value) // store new value
                    ; mov Rq(value), [rsi + (MEM_BACKING_OFFSET as i32) + 4 * Rq(SCRATCH_REGISTER) + 4 * sign_extend::<12>(parts.imm())] // reuse Rq(value) for read timestamp
                    ; mov [rsi + (MEM_BACKING_OFFSET as i32) + 4 * Rq(SCRATCH_REGISTER) + 4 * sign_extend::<12>(parts.imm())], r8 // update timestamp
                    ; mov [rdi + r9 * 4], eax // write old value into trace
                    ; mov [rdi + r9 * 8 + (TRACE_CHUNK_TIMESTAMPS_OFFSET as i32)], Rq(value) // write timestamp value into trace
                );
                bump_timestamp!(ops, 2);
                increment_trace!(ops);
            }

            Instruction::Csrrw(parts) => {
                todo!();
            }

            // Instruction::Csrrw(parts) => match parts.csr() {
            //     todo!();

            //     NON_DETERMINISM_CSR => {
            //         if rd != 0 {
            //             before_call!(ops);
            //             dynasm!(ops
            //                 ; mov rax, QWORD Context::<N>::read_nondeterminism as _
            //                 ; call rax
            //             );
            //             after_call!(ops);
            //             dynasm!(ops
            //                 ; mov Rd(out), eax
            //             );
            //             trace_register!(ops, out);
            //             store_result(&mut ops, rd);
            //         } else {
            //             trace_zero!(ops);
            //         }
            //         if parts.rs1() != 0 {
            //             load_into(&mut ops, parts.rs1(), SCRATCH_REGISTER);
            //             before_call!(ops);
            //             dynasm!(ops
            //                 ; mov rax, QWORD Context::<N>::write_nondeterminism as _
            //                 ; mov esi, Rd(SCRATCH_REGISTER)
            //                 ; call rax
            //             );
            //             after_call!(ops);
            //         }
            //     }
            //     csr => {
            //         let function = match csr {
            //             BLAKE2_ROUND_FUNCTION_WITH_EXTENDED_CONTROL_ACCESS_ID => {
            //                 Context::<N>::process_csr::<
            //                     BLAKE2_ROUND_FUNCTION_WITH_EXTENDED_CONTROL_ACCESS_ID,
            //                 > as _
            //             }
            //             U256_OPS_WITH_CONTROL_ACCESS_ID => {
            //                 Context::<N>::process_csr::<U256_OPS_WITH_CONTROL_ACCESS_ID> as _
            //             }
            //             x => {
            //                 emit_runtime_error!(ops);
            //                 continue;
            //             }
            //         };

            //         dynasm!(ops
            //             ; sub rsp, 32 * 4
            //             ; mov DWORD [rsp], 0
            //         );
            //         for i in 1..32 {
            //             let reg = load(&mut ops, i);
            //             dynasm!(ops
            //                 ; mov [rsp + 4 * i as i32], Rd(reg)
            //             );
            //         }

            //         load_into(&mut ops, parts.rs1(), x64::Rq::RCX as u8);

            //         dynasm!(ops
            //             ; mov rdx, rsp

            //             ; push rdi
            //             ; push rsi
            //             ; push r8
            //             ; push r9

            //             ; mov rax, QWORD function
            //             ; mov esi, ecx
            //             ; call rax

            //             ; pop r9
            //             ; pop r8
            //             ; pop rsi
            //             ; pop rdi
            //         );

            //         for i in 1..32 {
            //             let out = destination_gpr(i);
            //             dynasm!(ops
            //                 ; mov Rd(out), [rsp + 4 * i as i32]
            //             );
            //             store_result(&mut ops, i);
            //         }
            //         dynasm!(ops
            //             ; add rsp, 32 * 4
            //         );

            //         if rd != 0 {
            //             dynasm!(ops
            //                 ; mov Rd(out), eax
            //             );
            //             trace_register!(ops, out);
            //             store_result(&mut ops, rd);
            //         } else {
            //             trace_zero!(ops);
            //         }
            //     }
            // },

            _ => {
                emit_runtime_error!(ops)
            }
        }
    }

    // dynasm!(ops
    //     ; ->jump_offsets:
    //     ; .bytes jump_offsets.into_iter().flat_map(|x| x.to_le_bytes())

    //     ; ->exit_with_error:
    //     ; mov rax, QWORD print_complaint as _
    //     ; call rax
    //     ; ->quit:
    //     ; mov rax, r8
    //     ; mov rdx, r9
    // );

    dynasm!(ops
        ; ->jump_offsets:
        ; .bytes jump_offsets.into_iter().flat_map(|x| x.to_le_bytes())

        ; ->exit_with_error:
        ; mov rax, QWORD print_complaint as _
        ; call rax
        ; ->quit:
    );

    dynasm!(ops
        ; sub rsp, 32 * 4
        ; mov DWORD [rsp], 0
    );
    for i in 1..32 {
        let reg = load(&mut ops, i);
        dynasm!(ops
            ; mov [rsp + 4 * i as i32], Rd(reg)
        );
    }

    dynasm!(ops
        ; mov rcx, rsp

        ; push rax
        ; push rcx
        ; push rdx
        ;; before_call!(ops)
        ; mov rax, QWORD Context::<N, [Register; 1 << 28]>::receive_final_regs_trace as _
        ; mov rdi, rsi
        ; mov rsi, rcx
        ; sub rsp, 8
        ; call rax
        ; add rsp, 8
        ;; after_call!(ops)
        ; pop rdx
        ; pop rcx
        ; pop rax
    );

    for i in 1..32 {
        let out = destination_gpr(i);
        dynasm!(ops
            ; mov Rd(out), [rsp + 4 * i as i32]
        );
        store_result(&mut ops, i);
    }
    dynasm!(ops
        ; add rsp, 32 * 4
    );

    // prepare return values (u64, u64)
    dynasm!(ops
        ; mov rax, r8
        ; mov rdx, r9
    );

    epilogue!(ops);

    let code = ops.finalize().unwrap();

    // let assembly = unsafe {
    //     core::slice::from_raw_parts(code.ptr(start), code.len())
    // };
    // view_assembly(&assembly[..100], start.0);

    let mut context = unsafe {
        let mut context: Box<Context<'a, N, [Register; 1 << 28]>> = Box::new_uninit().assume_init();
        addr_of_mut!((*context).reg_timestamps).write([0; 32]);
        addr_of_mut!((*context).counters).write(DelegationsAndFamiliesCounters::default());
        addr_of_mut!((*context).non_determinism_source).write(non_determinism_source);
        addr_of_mut!((*context).trace_len).write(0);

        let mut initial_memory: Vec<_> = initial_memory.iter().map(|el| Register {
            value: *el,
            timestamp: 0,
        }).collect();
        core::ptr::copy_nonoverlapping(initial_memory.as_ptr(), addr_of_mut!((*context).memory_backing).cast(), initial_memory.len());
        for i in initial_memory.len()..1 << 28 {
            core::ptr::write(addr_of_mut!((*context).memory_backing).cast::<Register>().add(i), Register::default());
        }

        context
    };

    // let mut context = unsafe {
    //     let mut context: Box<Context<'a, N, [u32; 1 << 28]>> = Box::new_uninit().assume_init();
    //     addr_of_mut!((*context).reg_timestamps).write([0; 32]);
    //     addr_of_mut!((*context).counters).write(DelegationsAndFamiliesCounters::default());
    //     addr_of_mut!((*context).non_determinism_source).write(non_determinism_source);
    //     addr_of_mut!((*context).trace_len).write(0);

    //     core::ptr::copy_nonoverlapping(initial_memory.as_ptr(), addr_of_mut!((*context).memory_backing).cast(), initial_memory.len());
    //     for i in initial_memory.len()..1 << 28 {
    //         core::ptr::write(addr_of_mut!((*context).memory_backing).cast::<u32>().add(i), 0u32);
    //     }

    //     context
    // };

    // panic!("{}", MEM_BACKING_OFFSET);

    let context_ref_mut = &mut *context;
    let mut trace: Box<TraceChunk<TRACE_CHUNK_LEN>> = unsafe {
        let trace = Box::new_uninit().assume_init();

        trace
    };

    let before = std::time::Instant::now();

    // let run_program: extern "sysv64" fn(*mut Register, &mut Context<N, [u32]>) -> (u64, u64) =
    //     unsafe { std::mem::transmute(code.ptr(start)) };

    let run_program: extern "sysv64" fn(&mut TraceChunk<TRACE_CHUNK_LEN>, &mut Context<N, [Register]>) -> (u64, u64) =
        unsafe { std::mem::transmute(code.ptr(start)) };

    let (final_timestamp, remaining_trace) = run_program(trace.as_mut(), context_ref_mut);
    assert_eq!(final_timestamp % TIMESTAMP_STEP, 0);
    use common_constants::*;
    // println!(
    //     "execution of {} instructions took {:?}",
    //     (final_timestamp - INITIAL_TIMESTAMP) / TIMESTAMP_STEP,
    //     before.elapsed()
    // );
    let num_instructions = (final_timestamp - INITIAL_TIMESTAMP) / TIMESTAMP_STEP;
    println!(
        "Frequency is {} MHz over {} instructions",
        (num_instructions as f64)/(before.elapsed().as_micros() as f64),
        num_instructions
    );
    dbg!(remaining_trace);

    // let remaining_trace = run_program(trace.as_mut_ptr(), context);
    // println!(
    //     "execution of {} instructions took {:?}",
    //     context.trace_len as u64 + remaining_trace,
    //     before.elapsed()
    // );
}

#[repr(C)]
struct Context<'a, N: NonDeterminismCSRSource<RamWithRomRegion<{common_constants::rom::ROM_SECOND_WORD_BITS }>>, B: ?Sized> {
    reg_timestamps: [TimestampScalar; 32],
    counters: DelegationsAndFamiliesCounters,
    non_determinism_source: &'a mut N,
    trace_len: usize,
    memory_backing: B,
}

#[repr(C, align(8))]
struct TraceChunk<const CHUNK_SIZE: usize> {
    values: [u32; CHUNK_SIZE],
    timestamps: [TimestampScalar; CHUNK_SIZE],
}

const TRACE_CHUNK_TIMESTAMPS_OFFSET: usize = const {
    offset_of!(TraceChunk<TRACE_CHUNK_LEN>, timestamps)
};

const MEM_BACKING_OFFSET: usize = const {
    offset_of!(Context<'static, (), [Register; 1 << 28]>, memory_backing)
};

// const MEM_BACKING_OFFSET: usize = const {
//     offset_of!(Context<'static, (), [u32; 1 << 28]>, memory_backing)
// };

impl<'a, N: NonDeterminismCSRSource<RamWithRomRegion<{common_constants::rom::ROM_SECOND_WORD_BITS }>>, B: ?Sized> Context<'a, N, B> {
    extern "sysv64" fn read_nondeterminism(&mut self) -> u32 {
        self.non_determinism_source.read()
    }
    extern "sysv64" fn write_nondeterminism(&mut self, value: u32) {
        todo!();
        // self.non_determinism_source
            // .write_with_memory_access(&self.memory, value);
    }
    extern "sysv64" fn process_csr<const CSR_NUMBER: u32>(
        &mut self,
        registers: &mut [u32; 32],
    ) -> u32 {
        todo!();
    }

    extern "sysv64" fn receive_trace(&mut self, timestamp: u64) {
        use common_constants::*;
        println!("{} cycles passes", (timestamp - INITIAL_TIMESTAMP) / TIMESTAMP_STEP);
        // let trace: &[u32; TRACE_LEN] = unsafe { &*trace.cast() };
        // self.trace_len += TRACE_LEN;
    }

    extern "sysv64" fn receive_final_regs_trace(&mut self, registers: *mut [u32; 32]) {
        unsafe {
            println!("Final registers = {:?}", registers.as_ref());
        }
    }

    // extern "sysv64" fn receive_trace(&mut self, trace: *const ()) {
    //     // let trace: &[u32; TRACE_LEN] = unsafe { &*trace.cast() };
    //     // self.trace_len += TRACE_LEN;
    // }
}

extern "sysv64" fn print_registers(registers: &mut [u32; 32]) {
    println!("{registers:?}");
}

extern "sysv64" fn print_complaint() {
    println!("Runtime error!")
}

fn sign_extend<const SOURCE_BITS: u8>(x: u32) -> i32 {
    let shift = 32 - SOURCE_BITS;
    i32::from_ne_bytes((x << shift).to_ne_bytes()) >> shift
}

fn view_assembly(assembly: &[u8], start: usize) {
    /// Print register names
    fn reg_names(cs: &Capstone, regs: &[RegId]) -> String {
        let names: Vec<String> = regs.iter().map(|&x| cs.reg_name(x).unwrap()).collect();
        names.join(", ")
    }

    /// Print instruction group names
    fn group_names(cs: &Capstone, regs: &[InsnGroupId]) -> String {
        let names: Vec<String> = regs.iter().map(|&x| cs.group_name(x).unwrap()).collect();
        names.join(", ")
    }

    use capstone::*;
    use capstone::arch::*;

    let cs = Capstone::new()
        .x86()
        .mode(arch::x86::ArchMode::Mode64)
        .syntax(arch::x86::ArchSyntax::Att)
        .detail(true)
        .build()
        .expect("Failed to create Capstone object");

    let insns = cs.disasm_all(assembly, start as u64)
        .expect("Failed to disassemble");
    println!("Found {} instructions", insns.len());
    for i in insns.as_ref() {
        println!();
        println!("{}", i);

        let detail: InsnDetail = cs.insn_detail(&i).expect("Failed to get insn detail");
        let arch_detail: ArchDetail = detail.arch_detail();
        let ops = arch_detail.operands();

        let output: &[(&str, String)] = &[
            ("insn id:", format!("{:?}", i.id().0)),
            ("bytes:", format!("{:?}", i.bytes())),
            ("read regs:", reg_names(&cs, detail.regs_read())),
            ("write regs:", reg_names(&cs, detail.regs_write())),
            ("insn groups:", group_names(&cs, detail.groups())),
        ];

        for &(ref name, ref message) in output.iter() {
            println!("{:4}{:12} {}", "", name, message);
        }

        println!("{:4}operands: {}", "", ops.len());
        for op in ops {
            println!("{:8}{:?}", "", op);
        }
    }
}