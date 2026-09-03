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
    HCT_CAP_ABS_S8 = (1u << 5)
};

const hct_kernel_catalog_entry_t *hct_benchmark_server_catalog(size_t *count);
const uint8_t *hct_benchmark_server_catalog_hash(void);
const char *hct_benchmark_server_board_id(void);
const char *hct_benchmark_server_target_cpu(void);
const char *hct_benchmark_server_build_id(void);
uint32_t hct_benchmark_server_capability_flags(void);

#ifdef __cplusplus
}
#endif

#endif
