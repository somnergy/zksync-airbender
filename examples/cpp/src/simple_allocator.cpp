#include <stdint.h>
#include <stddef.h>

const size_t MEMORY_POOL_SIZE = 1024 * 1024; // 1 MB
char global_memory_pool[MEMORY_POOL_SIZE];
size_t allocated_bytes = 0;

void* operator new(size_t size) {
    if (allocated_bytes + size <= MEMORY_POOL_SIZE) {
        void* ptr = global_memory_pool + allocated_bytes;
        allocated_bytes += size;
        return ptr;
    }
    return nullptr; 
}

void operator delete(void* ptr) noexcept {
    // We do not free memory
}

void* operator new[](size_t size) {
    return operator new(size);
}

void operator delete[](void* ptr) noexcept {
    operator delete(ptr);
}
