#ifndef HCTP_PROTOCOL_H
#define HCTP_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HCTP_HEADER_SIZE 32u
#define HCTP_MAGIC_U32 0x31544348u
#define HCTP_SUPPORTED_VERSION 1u
#define HCTP_DEFAULT_MAX_PAYLOAD (64u * 1024u)

#define HCTP_FLAG_NONE 0u

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
    HCTP_MSG_HELLO = 1,
    HCTP_MSG_HELLO_ACK = 2,
    HCTP_MSG_LOAD_PLAN = 3,
    HCTP_MSG_CASE_META = 4,
    HCTP_MSG_BLOB_CHUNK = 5,
    HCTP_MSG_RUN_CORRECTNESS = 6,
    HCTP_MSG_CORRECTNESS_ACK = 7,
    HCTP_MSG_RUN_PERFORMANCE = 8,
    HCTP_MSG_ACK = 9,
    HCTP_MSG_NACK = 10,
    HCTP_MSG_ABORT_CASE = 11,
    HCTP_MSG_RESET_SESSION = 12,
    HCTP_MSG_PING = 13,
    HCTP_MSG_CAPABILITIES = 14,
    HCTP_MSG_REQUEST_CASE = 15,
    HCTP_MSG_REQUEST_BLOB = 16,
    HCTP_MSG_CASE_READY = 17,
    HCTP_MSG_CORRECTNESS_RESULT = 18,
    HCTP_MSG_OUTPUT_BEGIN = 19,
    HCTP_MSG_OUTPUT_CHUNK = 20,
    HCTP_MSG_OUTPUT_END = 21,
    HCTP_MSG_SAMPLE_RESULT = 22,
    HCTP_MSG_CASE_COMPLETE = 23,
    HCTP_MSG_SESSION_COMPLETE = 24,
    HCTP_MSG_ERROR = 25,
    HCTP_MSG_LOG = 26,
    HCTP_MSG_PONG = 27
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
