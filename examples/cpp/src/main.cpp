#include <stdint.h>

#include "airbender_csr.hpp"
#include "quasi_uart.hpp"

extern "C" {
// Boundaries of the heap
extern uint32_t _sheap;
extern uint32_t _eheap;

// Boundaries of the stack
extern uint32_t _sstack;
extern uint32_t _estack;
}

extern "C" void eh_personality() {}

struct MachineTrapFrame {
    uint32_t registers[32];
};

extern "C" [[noreturn]] void _start_rust() __attribute__((section(".init.rust")));
extern "C" uint32_t _machine_start_trap_rust(MachineTrapFrame*) __attribute__((section(".trap.rust")));


static constexpr uint32_t kModulus = 7919;

[[noreturn]] static void workload() {
    uint32_t n = airbender::csr_read_word();

    airbender::QuasiUART uart;
    uart.write_cstr("Computing Fibonacci number...");

    uint32_t *val = new uint32_t(2);

    uint32_t a = 0;
    uint32_t b = 1;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t c = (a + b) % kModulus;
        a = b;
        b = c;
    }

    uint32_t out[8] = {b, n, 0, 0, 0, 0, 0, 0};
    airbender::finish_success(out);
}

extern "C" [[noreturn]] void _start_rust() {
    workload();
}

extern "C" uint32_t _machine_start_trap_rust(MachineTrapFrame*) {
    while (true) {
    }
}
