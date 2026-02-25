#include "app.hpp"

#include <array>
#include <cstdlib>
#include <stdio.h>
#include <stdint.h>

#include "airbender_csr.hpp"

namespace {

// =============================================================================
// App contract and demo mode
// =============================================================================
//
// This sample consumes one u32 input word and commits an 8-word output. For
// starter-template UX we also reserve one sentinel input that intentionally
// calls std::exit(0) to demonstrate that libc exits are treated as failures
// unless the program explicitly commits output via finish_success().
constexpr uint32_t kModulus = 7919;
constexpr uint32_t kExitDemoSentinel = 0xFFFFFFFFu;

// =============================================================================
// Pure computation routine
// =============================================================================
uint32_t fibonacci_mod(uint32_t n) {
    uint32_t a = 0;
    uint32_t b = 1;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = (a + b) % kModulus;
        a = b;
        b = c;
    }
    return b;
}

} // namespace

[[noreturn]] void app_entrypoint() {
    // =============================================================================
    // 1) Read input and log through standard streams
    // =============================================================================
    //
    // stdout/stderr are bridged by newlib_syscalls.cpp, so application code can
    // use stdio (`printf`, `fputs`) without touching quasi-UART directly.
    const uint32_t n = airbender::csr_read_word();
    printf("Computing Fibonacci number with n=%u\n", static_cast<unsigned>(n));

    // =============================================================================
    // 2) Optional failure-contract demonstration
    // =============================================================================
    if (n == kExitDemoSentinel) {
        fputs("Exit-demo sentinel observed; calling std::exit(0).\n", stderr);
        // TODO: Replace this sentinel with a dedicated demo input mode once the
        // template grows a richer input format than a single u32 word.
        std::exit(0);
    }

    // =============================================================================
    // 3) Commit explicit success output for x10..x17
    // =============================================================================
    const uint32_t fib = fibonacci_mod(n);
    std::array<uint32_t, 8> output = {fib, n, 0, 0, 0, 0, 0, 0};

    fputs("Computation finished; committing output words.\n", stdout);
    fflush(stdout);
    fflush(stderr);

    airbender::finish_success(output.data());
}
