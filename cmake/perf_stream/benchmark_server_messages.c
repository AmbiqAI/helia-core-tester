#include "benchmark_server_messages.h"

#include <string.h>

#include "benchmark_server_catalog.h"

static hctp_status_t write_u8(uint8_t *buffer, size_t capacity, size_t *offset, uint8_t value)
{
    if (*offset + 1u > capacity)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    buffer[*offset] = value;
    *offset += 1u;
    return HCTP_STATUS_OK;
}

static hctp_status_t write_u16(uint8_t *buffer, size_t capacity, size_t *offset, uint16_t value)
{
    if (*offset + 2u > capacity)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    buffer[*offset + 0u] = (uint8_t)(value & 0xFFu);
    buffer[*offset + 1u] = (uint8_t)((value >> 8) & 0xFFu);
    *offset += 2u;
    return HCTP_STATUS_OK;
}

static hctp_status_t write_u32(uint8_t *buffer, size_t capacity, size_t *offset, uint32_t value)
{
    if (*offset + 4u > capacity)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    buffer[*offset + 0u] = (uint8_t)(value & 0xFFu);
    buffer[*offset + 1u] = (uint8_t)((value >> 8) & 0xFFu);
    buffer[*offset + 2u] = (uint8_t)((value >> 16) & 0xFFu);
    buffer[*offset + 3u] = (uint8_t)((value >> 24) & 0xFFu);
    *offset += 4u;
    return HCTP_STATUS_OK;
}

static hctp_status_t write_fixed(uint8_t *buffer, size_t capacity, size_t *offset, const uint8_t *value, size_t size)
{
    if (*offset + size > capacity)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    memcpy(&buffer[*offset], value, size);
    *offset += size;
    return HCTP_STATUS_OK;
}

static hctp_status_t write_text(uint8_t *buffer, size_t capacity, size_t *offset, const char *value)
{
    const size_t length = strlen(value);
    hctp_status_t status;
    if (length > 0xFFFFu)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    status = write_u16(buffer, capacity, offset, (uint16_t)length);
    if (status != HCTP_STATUS_OK)
    {
        return status;
    }
    return write_fixed(buffer, capacity, offset, (const uint8_t *)value, length);
}

static hctp_status_t wrap_frame(uint16_t message_type,
                                uint32_t session_id,
                                uint32_t sequence_id,
                                uint8_t *payload,
                                size_t payload_length,
                                uint8_t *frame_bytes,
                                size_t frame_capacity,
                                size_t *frame_length)
{
    hctp_frame_header_t header;

    if (frame_capacity < HCTP_HEADER_SIZE + payload_length)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }

    header.magic = HCTP_MAGIC_U32;
    header.protocol_version = HCTP_SUPPORTED_VERSION;
    header.message_type = message_type;
    header.flags = HCTP_FLAG_NONE;
    header.session_id = session_id;
    header.sequence_id = sequence_id;
    header.payload_length = (uint32_t)payload_length;
    header.payload_crc32 = hctp_crc32(payload, payload_length);
    header.header_crc32 = 0u;

    hctp_encode_header(frame_bytes, &header);
    memcpy(frame_bytes + HCTP_HEADER_SIZE, payload, payload_length);
    *frame_length = HCTP_HEADER_SIZE + payload_length;
    return HCTP_STATUS_OK;
}

hctp_status_t hct_build_hello_frame(uint32_t session_id,
                                    uint32_t sequence_id,
                                    uint32_t max_frame_payload,
                                    uint32_t runtime_arena_capacity,
                                    uint8_t *frame_bytes,
                                    size_t frame_capacity,
                                    size_t *frame_length)
{
    uint8_t payload[256];
    size_t offset = 0u;
    hctp_status_t status;

    status = write_text(payload, sizeof(payload), &offset, hct_benchmark_server_build_id());
    if (status != HCTP_STATUS_OK) return status;
    status = write_fixed(payload, sizeof(payload), &offset, hct_benchmark_server_catalog_hash(), 32u);
    if (status != HCTP_STATUS_OK) return status;
    status = write_u32(payload, sizeof(payload), &offset, max_frame_payload);
    if (status != HCTP_STATUS_OK) return status;
    status = write_u32(payload, sizeof(payload), &offset, runtime_arena_capacity);
    if (status != HCTP_STATUS_OK) return status;
    status = write_u8(payload, sizeof(payload), &offset, HCT_BENCHMARK_SERVER_TRANSFER_MODE_CASE_STREAMING);
    if (status != HCTP_STATUS_OK) return status;
    status = write_u8(payload, sizeof(payload), &offset, HCT_BENCHMARK_SERVER_OUTPUT_MODE_FULL);
    if (status != HCTP_STATUS_OK) return status;
    status = write_text(payload, sizeof(payload), &offset, hct_benchmark_server_board_id());
    if (status != HCTP_STATUS_OK) return status;
    status = write_text(payload, sizeof(payload), &offset, hct_benchmark_server_target_cpu());
    if (status != HCTP_STATUS_OK) return status;
    status = write_u8(payload, sizeof(payload), &offset, HCT_BENCHMARK_SERVER_TRANSPORT_RTT);
    if (status != HCTP_STATUS_OK) return status;
    status = write_u32(payload, sizeof(payload), &offset, hct_benchmark_server_capability_flags());
    if (status != HCTP_STATUS_OK) return status;

    return wrap_frame(HCTP_MSG_HELLO, session_id, sequence_id, payload, offset, frame_bytes, frame_capacity, frame_length);
}

hctp_status_t hct_build_catalog_frame(uint32_t session_id,
                                      uint32_t sequence_id,
                                      uint8_t *frame_bytes,
                                      size_t frame_capacity,
                                      size_t *frame_length)
{
    uint8_t payload[512];
    size_t offset = 0u;
    size_t count = 0u;
    size_t index;
    const hct_kernel_catalog_entry_t *entries = hct_benchmark_server_catalog(&count);
    hctp_status_t status = write_u16(payload, sizeof(payload), &offset, (uint16_t)count);
    if (status != HCTP_STATUS_OK) return status;

    for (index = 0u; index < count; ++index)
    {
        status = write_u32(payload, sizeof(payload), &offset, entries[index].kernel_id);
        if (status != HCTP_STATUS_OK) return status;
        status = write_text(payload, sizeof(payload), &offset, entries[index].canonical_name);
        if (status != HCTP_STATUS_OK) return status;
        status = write_text(payload, sizeof(payload), &offset, entries[index].operator_family);
        if (status != HCTP_STATUS_OK) return status;
        status = write_u16(payload, sizeof(payload), &offset, entries[index].api_version);
        if (status != HCTP_STATUS_OK) return status;
        status = write_text(payload, sizeof(payload), &offset, entries[index].supported_dtype);
        if (status != HCTP_STATUS_OK) return status;
        status = write_u16(payload, sizeof(payload), &offset, entries[index].adapter_schema_version);
        if (status != HCTP_STATUS_OK) return status;
        status = write_u8(payload, sizeof(payload), &offset, entries[index].stateless ? 1u : 0u);
        if (status != HCTP_STATUS_OK) return status;
        status = write_u8(payload, sizeof(payload), &offset, entries[index].repeated_invocation_safe ? 1u : 0u);
        if (status != HCTP_STATUS_OK) return status;
        status = write_u8(payload, sizeof(payload), &offset, entries[index].mutates_input ? 1u : 0u);
        if (status != HCTP_STATUS_OK) return status;
        status = write_u32(payload, sizeof(payload), &offset, entries[index].scratch_bytes);
        if (status != HCTP_STATUS_OK) return status;
    }

    return wrap_frame(HCTP_MSG_CAPABILITIES, session_id, sequence_id, payload, offset, frame_bytes, frame_capacity, frame_length);
}
