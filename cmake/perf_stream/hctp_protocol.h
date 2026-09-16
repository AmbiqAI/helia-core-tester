#ifndef HCTP_PROTOCOL_H
#define HCTP_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HCTP_HEADER_SIZE 32u
#define HCTP_MAGIC_U32 0x31544348u
/* v3: TARGET_INFO / KERNEL_CATALOG / SESSION_PLAN vocabulary with compact message ids
 * and TARGET_INFO advertising the session limits (max cases per plan, max PMU passes)
 * next to the PMU slot count and receive-buffer bound. Must match
 * helia_core_tester/perf_stream/hctp.py. */
#define HCTP_SUPPORTED_VERSION 3u
#define HCTP_DEFAULT_MAX_PAYLOAD (64u * 1024u)

#define HCTP_FLAG_NONE 0u
/* F008: set on every non-final KERNEL_CATALOG chunk when the catalog is too large for a
 * single frame's payload; the host keeps decoding/accumulating chunks until it receives
 * one without this flag set. */
#define HCTP_FLAG_MORE (1u << 0)

typedef enum
{
    HCTP_STATUS_OK = 0,
    HCTP_STATUS_INVALID_ARGUMENT = -1,
    HCTP_STATUS_INVALID_MAGIC = -2,
    HCTP_STATUS_UNSUPPORTED_VERSION = -3,
    HCTP_STATUS_HEADER_CRC_MISMATCH = -4,
    HCTP_STATUS_PAYLOAD_CRC_MISMATCH = -5,
    HCTP_STATUS_OVERSIZED_PAYLOAD = -6,
    HCTP_STATUS_TRUNCATED_FRAME = -7
} hctp_status_t;

typedef enum
{
    /* HCTP v3 message ids, in protocol order (target -> host unless noted). */
    HCTP_MSG_TARGET_INFO = 1,
    HCTP_MSG_TARGET_INFO_ACK = 2,   /* host -> target */
    HCTP_MSG_KERNEL_CATALOG = 3,
    HCTP_MSG_SESSION_PLAN = 4,      /* host -> target */
    HCTP_MSG_REQUEST_CASE = 5,
    HCTP_MSG_CASE_META = 6,         /* host -> target */
    HCTP_MSG_REQUEST_BLOB = 7,
    HCTP_MSG_BLOB_CHUNK = 8,        /* host -> target */
    HCTP_MSG_CASE_READY = 9,
    HCTP_MSG_RUN_CORRECTNESS = 10,  /* host -> target */
    HCTP_MSG_CORRECTNESS_RESULT = 11,
    HCTP_MSG_OUTPUT_BEGIN = 12,
    HCTP_MSG_OUTPUT_CHUNK = 13,
    HCTP_MSG_OUTPUT_END = 14,
    HCTP_MSG_CORRECTNESS_ACK = 15,  /* host -> target */
    HCTP_MSG_RUN_PERFORMANCE = 16,  /* host -> target */
    HCTP_MSG_SAMPLE_RESULT = 17,
    HCTP_MSG_CASE_COMPLETE = 18,
    HCTP_MSG_SESSION_COMPLETE = 19,
    HCTP_MSG_ERROR = 20
} hctp_message_type_t;

typedef struct
{
    uint32_t magic;
    uint16_t protocol_version;
    uint16_t message_type;
    uint32_t flags;
    uint32_t session_id;
    uint32_t sequence_id;
    uint32_t payload_length;
    uint32_t payload_crc32;
    uint32_t header_crc32;
} hctp_frame_header_t;

typedef struct
{
    hctp_frame_header_t header;
    const uint8_t *payload;
} hctp_frame_view_t;

uint32_t hctp_crc32(const uint8_t *data, size_t length);
void hctp_encode_header(uint8_t out_header[HCTP_HEADER_SIZE], const hctp_frame_header_t *header_without_crc);
hctp_status_t hctp_decode_header(const uint8_t *header_bytes, size_t header_length, uint32_t max_payload, hctp_frame_header_t *out_header);
hctp_status_t hctp_decode_frame(const uint8_t *frame_bytes, size_t frame_length, uint32_t max_payload, hctp_frame_view_t *out_frame);

#ifdef __cplusplus
}
#endif

#endif
