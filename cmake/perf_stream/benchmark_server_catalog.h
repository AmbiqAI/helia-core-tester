#ifndef HCT_BENCHMARK_SERVER_CATALOG_H
#define HCT_BENCHMARK_SERVER_CATALOG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    uint32_t kernel_id;
    const char *canonical_name;
    const char *operator_family;
    uint16_t api_version;
    const char *supported_dtype;
    uint16_t adapter_schema_version;
    bool stateless;
    bool repeated_invocation_safe;
    bool mutates_input;
    uint32_t scratch_bytes;
} hct_kernel_catalog_entry_t;

enum
{
    HCT_CAP_CASE_STREAMING = (1u << 0),
    HCT_CAP_CORRECTNESS = (1u << 1),
    HCT_CAP_PERFORMANCE = (1u << 2),
    HCT_CAP_RTT_TRANSPORT = (1u << 3),
    HCT_CAP_KERNEL_CATALOG = (1u << 4),
    HCT_CAP_ABS_S8 = (1u << 5),
    /* Armv8.1-M PMU event counters are available (__PMU_PRESENT == 1). Absent on
     * DWT-only cores such as Cortex-M4, where SAMPLE_RESULT only carries cycles. */
    HCT_CAP_PMU_ARMV8M = (1u << 6)
};

const hct_kernel_catalog_entry_t *hct_benchmark_server_catalog(size_t *count);
const uint8_t *hct_benchmark_server_catalog_hash(void);
const char *hct_benchmark_server_board_id(void);
const char *hct_benchmark_server_target_cpu(void);
const char *hct_benchmark_server_build_id(void);
uint32_t hct_benchmark_server_capability_flags(void);
/* Number of 16-bit PMU event-counter slots (__PMU_NUM_EVENTCNT, 8 on Cortex-M55);
 * 0 when HCT_CAP_PMU_ARMV8M is not set. */
uint8_t hct_benchmark_server_pmu_counter_slots(void);

#ifdef __cplusplus
}
#endif

#endif
