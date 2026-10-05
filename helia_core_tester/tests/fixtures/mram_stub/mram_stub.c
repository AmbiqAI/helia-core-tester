/* Host MRAM stub: plain RAM, logged. */
#include <string.h>

#include "hct_mram_stub.h"

__attribute__((aligned(16))) uint8_t hct_stub_mram[HCT_STUB_MRAM_BYTES];
uintptr_t hct_stub_mram_end = 0u;
hct_mram_stub_t hct_mram_stub;

/* Assembler names; Mach-O adds "_". */
#define HCT_STR_(x) #x
#define HCT_STR(x) HCT_STR_(x)
#define HCT_ASM_NAME(name) HCT_STR(__USER_LABEL_PREFIX__) #name
#define HCT_ALIAS(name) ".globl " HCT_ASM_NAME(name) "\n.set " HCT_ASM_NAME(name) ", " HCT_ASM_NAME(hct_stub_mram) "\n"

/* Linker symbols: empty data, image ends here. */
__asm__(HCT_ALIAS(_init_data) HCT_ALIAS(_sdata) HCT_ALIAS(_edata)
        HCT_ALIAS(_init_data_sram) HCT_ALIAS(_ssdata) HCT_ALIAS(_sedata));

int am_hal_mram_main_program(uint32_t key, uint32_t *src, uint32_t *dst, uint32_t words)
{
    const uint8_t *begin = hct_stub_mram;
    hct_mram_stub.program_calls += 1u;
    if (key != AM_HAL_MRAM_PROGRAM_KEY || (const uint8_t *)dst < begin ||
        (const uint8_t *)(dst + words) > begin + HCT_STUB_MRAM_BYTES || hct_mram_stub.fail_program != 0)
    {
        hct_mram_stub.fail_program = 0;
        return 1;
    }
    memcpy(dst, src, (size_t)words * 4u);
    if (hct_mram_stub.corrupt_program != 0 && words > 0u)
    {
        hct_mram_stub.corrupt_program = 0;
        ((uint8_t *)dst)[0] ^= 0xFFu;
    }
    return 0;
}

void SCB_InvalidateDCache_by_Addr(volatile void *addr, int32_t bytes)
{
    hct_mram_stub.invalidate_calls += 1u;
    hct_mram_stub.last_invalidate = addr;
    hct_mram_stub.last_invalidate_bytes = bytes;
}
