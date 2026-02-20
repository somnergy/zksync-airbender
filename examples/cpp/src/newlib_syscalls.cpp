#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>

#include "airbender_csr.hpp"
#include "quasi_uart.hpp"

extern "C" {
extern uint32_t _sheap;
extern uint32_t _eheap;
}

namespace {

constexpr int kStdinFd = 0;
constexpr int kStdoutFd = 1;
constexpr int kStderrFd = 2;
constexpr intptr_t kSbrkFailure = -1;

uint8_t* g_program_break = reinterpret_cast<uint8_t*>(&_sheap);

bool is_supported_fd(int fd) {
    return fd == kStdinFd || fd == kStdoutFd || fd == kStderrFd;
}

} // namespace

extern "C" int _write(int fd, const char* buf, int len) {
    // newlib funnels stdio writes through this hook. We only surface stdout/stderr
    // via quasi-UART because that is what the guest runtime can expose today.
    if (fd != kStdoutFd && fd != kStderrFd) {
        errno = EBADF;
        return -1;
    }
    if (len < 0) {
        errno = EINVAL;
        return -1;
    }
    if (len == 0) {
        return 0;
    }

    airbender::QuasiUART uart;
    uart.write_str(buf, static_cast<size_t>(len));
    return len;
}

extern "C" int _read(int fd, char*, int) {
    // TODO: Map fd=0 reads to Airbender's input stream when byte-oriented stdin is required.
    if (fd != kStdinFd) {
        errno = EBADF;
    } else {
        errno = ENOSYS;
    }
    return -1;
}

extern "C" int _close(int fd) {
    if (!is_supported_fd(fd)) {
        errno = EBADF;
        return -1;
    }
    return 0;
}

extern "C" int _fstat(int fd, struct stat* st) {
    if (!is_supported_fd(fd)) {
        errno = EBADF;
        return -1;
    }
    st->st_mode = S_IFCHR;
    return 0;
}

extern "C" int _isatty(int fd) {
    if (!is_supported_fd(fd)) {
        errno = EBADF;
        return 0;
    }
    return 1;
}

extern "C" int _lseek(int fd, int, int) {
    if (!is_supported_fd(fd)) {
        errno = EBADF;
        return -1;
    }
    return 0;
}

extern "C" int _getpid() {
    return 1;
}

extern "C" int _kill(int, int) {
    errno = EINVAL;
    return -1;
}

extern "C" void* _sbrk(ptrdiff_t increment) {
    // We keep a monotonic program break over the linker-defined heap range.
    // Shrinking is intentionally omitted for now because this guest does not
    // rely on returning heap pages to the runtime.
    if (increment < 0) {
        errno = ENOMEM;
        return reinterpret_cast<void*>(kSbrkFailure);
    }

    uint8_t* program_break = g_program_break;
    uint8_t* heap_end = reinterpret_cast<uint8_t*>(&_eheap);
    size_t available = static_cast<size_t>(heap_end - program_break);
    size_t requested = static_cast<size_t>(increment);

    if (requested > available) {
        errno = ENOMEM;
        return reinterpret_cast<void*>(kSbrkFailure);
    }

    void* previous_break = program_break;
    g_program_break = program_break + requested;
    return previous_break;
}

extern "C" [[noreturn]] void _exit(int status) {
    // Match Airbender's success/error termination model for libc-driven exits.
    if (status == 0) {
        uint32_t output[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        airbender::finish_success(output);
    }
    airbender::finish_error();
}
