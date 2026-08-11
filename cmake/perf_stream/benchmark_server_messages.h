#ifndef HCT_BENCHMARK_SERVER_MESSAGES_H
#define HCT_BENCHMARK_SERVER_MESSAGES_H

#include <stddef.h>
#include <stdint.h>

#include "hctp_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HCT_BENCHMARK_SERVER_TRANSFER_MODE_CASE_STREAMING 1u
#define HCT_BENCHMARK_SERVER_OUTPUT_MODE_FULL 1u
#define HCT_BENCHMARK_SERVER_TRANSPORT_RTT 1u

hctp_status_t hct_build_hello_frame(uint32_t session_id,
                                    uint32_t sequence_id,
                                    uint32_t max_frame_payload,
                                    uint32_t runtime_arena_capacity,
                                    uint8_t *frame_bytes,
                                    size_t frame_capacity,
                                    size_t *frame_length);

hctp_status_t hct_build_catalog_frame(uint32_t session_id,
                                      uint32_t sequence_id,
                                      uint8_t *frame_bytes,
                                      size_t frame_capacity,
                                      size_t *frame_length);

#ifdef __cplusplus
}
#endif

#endif
