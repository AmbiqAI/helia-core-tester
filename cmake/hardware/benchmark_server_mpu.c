#include "benchmark_server_mpu.h"

#include "am_mcu_apollo.h"
#include "benchmark_server_session.h"

/* Armv8-M only; v7-M keeps the default map. */
#if defined(__MPU_PRESENT) && (__MPU_PRESENT == 1U) && \
    (defined(__ARM_ARCH_8M_MAIN__) || defined(__ARM_ARCH_8_1M_MAIN__))
#define HCT_MPU_XN 1
#else
#define HCT_MPU_XN 0
#endif

#if HCT_MPU_XN
/* MAIR slots; match the default memory map. */
#define HCT_ATTR_WBWA 0u
#define HCT_ATTR_WT 1u
/* MRAM image end, as the linker scripts. */
#if defined(AM_PART_APOLLO330P)
#define HCT_MRAM_LIMIT 0x005FFFFFu
#else
#define HCT_MRAM_LIMIT 0x007FFFFFu
#endif
#define HCT_ITCM_LIMIT 0x0003FFFFu

/* First plain .rodata input: rodata start. */
__attribute__((section(".rodata"), aligned(32), used)) const uint32_t hct_rodata_start = 0u;

typedef struct
{
    uint32_t rbar;
    uint32_t rlar;
} hct_region_t;

static hct_region_t s_regions[5];
static uint32_t s_region_count;

/* SDK regions could overlap or share MAIR. */
static bool mpu_in_use(uint32_t count)
{
    uint32_t index;
    if ((MPU->CTRL & MPU_CTRL_ENABLE_Msk) != 0u)
    {
        return true;
    }
    for (index = 0u; index < count; ++index)
    {
        MPU->RNR = index;
        if ((MPU->RLAR & MPU_RLAR_EN_Msk) != 0u)
        {
            return true;
        }
    }
    return false;
}

static void add_region(uint32_t base, uint32_t limit, uint32_t ro, uint32_t xn, uint32_t attr)
{
    s_regions[s_region_count].rbar = ARM_MPU_RBAR(base, ARM_MPU_SH_NON, ro, 1u, xn);
    s_regions[s_region_count].rlar = ARM_MPU_RLAR(limit, attr);
    s_region_count += 1u;
}

bool hct_mpu_protect(void)
{
    const uint32_t count = (MPU->TYPE & MPU_TYPE_DREGION_Msk) >> MPU_TYPE_DREGION_Pos;
    uint32_t index;
    if (count < sizeof(s_regions) / sizeof(s_regions[0]) || mpu_in_use(count))
    {
        return false;
    }
    /* SRAM: data only. */
    add_region(0x20000000u, 0x3FFFFFFFu, 0u, 1u, HCT_ATTR_WBWA);
    /* MRAM after code: rodata, data, weights. */
    add_region((uint32_t)(uintptr_t)&hct_rodata_start, HCT_MRAM_LIMIT, 0u, 1u, HCT_ATTR_WT);
    /* ITCM: boot-copied HAL code, read-only. */
    add_region(0x00000000u, HCT_ITCM_LIMIT, 1u, 0u, HCT_ATTR_WT);
    /* External RAM windows: data only. */
    add_region(0x60000000u, 0x7FFFFFFFu, 0u, 1u, HCT_ATTR_WBWA);
    add_region(0x80000000u, 0x9FFFFFFFu, 0u, 1u, HCT_ATTR_WT);
    __DMB();
    ARM_MPU_SetMemAttr(HCT_ATTR_WBWA, ARM_MPU_ATTR(ARM_MPU_ATTR_MEMORY_(1, 1, 1, 1), ARM_MPU_ATTR_MEMORY_(1, 1, 1, 1)));
    ARM_MPU_SetMemAttr(HCT_ATTR_WT, ARM_MPU_ATTR(ARM_MPU_ATTR_MEMORY_(1, 0, 1, 0), ARM_MPU_ATTR_MEMORY_(1, 0, 1, 0)));
    for (index = 0u; index < s_region_count; ++index)
    {
        ARM_MPU_SetRegion(index, s_regions[index].rbar, s_regions[index].rlar);
    }
    /* Enables MemManage too. */
    ARM_MPU_Enable(MPU_CTRL_PRIVDEFENA_Msk);
    return true;
}

bool hct_mpu_intact(void)
{
    uint32_t index;
    if (s_region_count == 0u)
    {
        return true;
    }
    if ((MPU->CTRL & (MPU_CTRL_ENABLE_Msk | MPU_CTRL_PRIVDEFENA_Msk)) != (MPU_CTRL_ENABLE_Msk | MPU_CTRL_PRIVDEFENA_Msk) ||
        (SCB->SHCSR & SCB_SHCSR_MEMFAULTENA_Msk) == 0u)
    {
        return false;
    }
    for (index = 0u; index < s_region_count; ++index)
    {
        MPU->RNR = index;
        if (MPU->RBAR != s_regions[index].rbar || MPU->RLAR != s_regions[index].rlar)
        {
            return false;
        }
    }
    return true;
}

/* IT/ICI bits and exception number. */
#define HCT_XPSR_CLEAR ((3u << 25) | (0x3Fu << 10) | 0x1FFu)

/* Frame: r0-r3, r12, lr, pc, xpsr. */
void hct_mem_fault(uint32_t *frame, uint32_t exc_return)
{
    const uint32_t cfsr = SCB->CFSR;
    /* Unwind only thread-mode kernel faults. */
    if (!hct_fault_armed() || (exc_return & (1u << 3)) == 0u)
    {
        for (;;)
        {
        }
    }
    SCB->CFSR = cfsr & SCB_CFSR_MEMFAULTSR_Msk;
    frame[0] = (uint32_t)((cfsr & SCB_CFSR_IACCVIOL_Msk) != 0u ? HCT_STATUS_EXEC_FROM_RAM : HCT_STATUS_PROTECTED_WRITE);
    frame[6] = (uint32_t)(uintptr_t)hct_fault_unwind & ~1u;
    frame[7] = (frame[7] & ~HCT_XPSR_CLEAR) | (1u << 24);
}

__attribute__((naked)) void MemManage_Handler(void)
{
    __asm volatile(
        "mov r1, lr\n"
        "tst lr, #4\n"
        "ite eq\n"
        "mrseq r0, msp\n"
        "mrsne r0, psp\n"
        "b hct_mem_fault\n");
}
#else
bool hct_mpu_protect(void)
{
    return false;
}

bool hct_mpu_intact(void)
{
    return true;
}
#endif
