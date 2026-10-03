/* Host stub of the MRAM HAL. */
#ifndef HCT_MRAM_STUB_H
#define HCT_MRAM_STUB_H

#include <stdint.h>

#define AM_HAL_MRAM_PROGRAM_KEY 0x12344321u

/* Fake MRAM: image ends at its start. */
#define HCT_STUB_MRAM_BYTES (640u * 1024u)
extern uint8_t hct_stub_mram[];
/* Tests shrink this to force overlap. */
extern uintptr_t hct_stub_mram_end;
#define HCT_MRAM_END hct_stub_mram_end

/* Stub behaviour and call log. */
typedef struct
{
    uint32_t program_calls;
    uint32_t invalidate_calls;
    /* Non-zero: next program call fails. */
    int fail_program;
    /* Non-zero: next program flips a byte. */
    int corrupt_program;
    const volatile void *last_invalidate;
    int32_t last_invalidate_bytes;
} hct_mram_stub_t;

extern hct_mram_stub_t hct_mram_stub;

int am_hal_mram_main_program(uint32_t key, uint32_t *src, uint32_t *dst, uint32_t words);
void SCB_InvalidateDCache_by_Addr(volatile void *addr, int32_t bytes);

#endif
