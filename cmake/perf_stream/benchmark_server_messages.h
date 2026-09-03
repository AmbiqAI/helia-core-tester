#ifndef HCT_BENCHMARK_SERVER_MESSAGES_H
#define HCT_BENCHMARK_SERVER_MESSAGES_H

#include <stdbool.h>
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

/* F008: paginated catalog frame builder. Emits the entries starting at start_index that
 * fit within HCT_CATALOG_CHUNK_MAX_PAYLOAD_BYTES, reporting next_index (the index to
 * resume from) and is_final (true once the returned chunk covers the last entry). The
 * frame carries HCTP_FLAG_MORE on every non-final chunk. Callers must loop until
 * is_final is true. */
hctp_status_t hct_build_catalog_frame_chunk(uint32_t session_id,
                                            uint32_t sequence_id,
                                            size_t start_index,
                                            uint8_t *frame_bytes,
                                            size_t frame_capacity,
                                            size_t *frame_length,
                                            size_t *next_index,
                                            bool *is_final);

#ifdef __cplusplus
}
#endif

#endif
