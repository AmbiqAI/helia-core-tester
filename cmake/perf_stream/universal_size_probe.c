#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

typedef struct
{
    const char *name;
    const void *address;
} hct_symbol_ref_t;

static const hct_symbol_ref_t g_hct_symbol_refs[] = {
#include "kernel_symbol_refs.inc"
};

volatile uintptr_t g_hct_symbol_registry_anchor;
volatile size_t g_hct_symbol_registry_count;

size_t hct_universal_symbol_count(void)
{
    return sizeof(g_hct_symbol_refs) / sizeof(g_hct_symbol_refs[0]);
}

int main(void)
{
    uintptr_t checksum = 0u;
    const size_t count = hct_universal_symbol_count();

    for (size_t index = 0; index < count; ++index)
    {
        checksum ^= ((uintptr_t)g_hct_symbol_refs[index].address >> 2) + (uintptr_t)index;
    }

    g_hct_symbol_registry_count = count;
    g_hct_symbol_registry_anchor = checksum;

    for (;;)
    {
    }
}
