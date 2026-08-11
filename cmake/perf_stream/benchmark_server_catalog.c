#include "benchmark_server_catalog.h"

#ifndef HCT_BENCHMARK_SERVER_BOARD_ID
#define HCT_BENCHMARK_SERVER_BOARD_ID "apollo510_evb"
#endif

#ifndef HCT_BENCHMARK_SERVER_TARGET_CPU
#define HCT_BENCHMARK_SERVER_TARGET_CPU "cortex-m55"
#endif

static const hct_kernel_catalog_entry_t g_hct_kernel_catalog[] = {
    {1u, "arm_abs_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {2u, "arm_convolve_s8", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 64u},
    {3u, "arm_add_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {4u, "arm_sub_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {5u, "arm_mul_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {6u, "arm_minimum_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {7u, "arm_maximum_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
};

static const uint8_t g_hct_kernel_catalog_hash[32] = {
    0x0eu, 0x6du, 0x27u, 0xbau, 0x52u, 0x92u, 0x36u, 0x07u,
    0x23u, 0x5fu, 0x5eu, 0x5bu, 0x7bu, 0xffu, 0x5eu, 0xc5u,
    0x43u, 0x3fu, 0x68u, 0xfcu, 0xd0u, 0x68u, 0xa9u, 0x02u,
    0x05u, 0x87u, 0xc6u, 0x04u, 0x29u, 0x99u, 0x97u, 0x1eu,
};

const hct_kernel_catalog_entry_t *hct_benchmark_server_catalog(size_t *count)
{
    if (count != NULL)
    {
        *count = sizeof(g_hct_kernel_catalog) / sizeof(g_hct_kernel_catalog[0]);
    }
    return g_hct_kernel_catalog;
}

const uint8_t *hct_benchmark_server_catalog_hash(void)
{
    return g_hct_kernel_catalog_hash;
}

const char *hct_benchmark_server_board_id(void)
{
    return HCT_BENCHMARK_SERVER_BOARD_ID;
}

const char *hct_benchmark_server_target_cpu(void)
{
    return HCT_BENCHMARK_SERVER_TARGET_CPU;
}

const char *hct_benchmark_server_build_id(void)
{
    return "hct-benchmark-server-v0";
}

uint32_t hct_benchmark_server_capability_flags(void)
{
    return HCT_CAP_CASE_STREAMING
         | HCT_CAP_CORRECTNESS
         | HCT_CAP_PERFORMANCE
         | HCT_CAP_RTT_TRANSPORT
         | HCT_CAP_KERNEL_CATALOG
         | HCT_CAP_ABS_S8;
}
