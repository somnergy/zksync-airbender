#include <stddef.h>
#include <stdint.h>

void *memset(void *dest, int value, size_t count) {
    uint8_t *ptr = (uint8_t *)dest;
    uint8_t v = (uint8_t)value;
    for (size_t i = 0; i < count; ++i) {
        ptr[i] = v;
    }
    return dest;
}
