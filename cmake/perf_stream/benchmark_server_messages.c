#include "benchmark_server_messages.h"

#include <stdbool.h>
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
                                uint32_t flags,
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
    header.flags = flags;
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

    return wrap_frame(HCTP_MSG_HELLO, session_id, sequence_id, HCTP_FLAG_NONE, payload, offset, frame_bytes, frame_capacity, frame_length);
}

/* F008: bounded max payload size per CAPABILITIES chunk (well under the write buffer's
 * 512-byte scratch capacity and the firmware's 1 KiB flush buffer / 32 KiB outbox), so a
 * 126-entry catalog is always paginated into multiple frames instead of overflowing a
 * single one. */
#define HCT_CATALOG_CHUNK_MAX_PAYLOAD_BYTES 256u

static hctp_status_t write_catalog_entry(uint8_t *payload, size_t capacity, size_t *offset, const hct_kernel_catalog_entry_t *entry)
{
    hctp_status_t status = write_u32(payload, capacity, offset, entry->kernel_id);
    if (status != HCTP_STATUS_OK) return status;
    status = write_text(payload, capacity, offset, entry->canonical_name);
    if (status != HCTP_STATUS_OK) return status;
    status = write_text(payload, capacity, offset, entry->operator_family);
    if (status != HCTP_STATUS_OK) return status;
    status = write_u16(payload, capacity, offset, entry->api_version);
    if (status != HCTP_STATUS_OK) return status;
    status = write_text(payload, capacity, offset, entry->supported_dtype);
    if (status != HCTP_STATUS_OK) return status;
    status = write_u16(payload, capacity, offset, entry->adapter_schema_version);
    if (status != HCTP_STATUS_OK) return status;
    status = write_u8(payload, capacity, offset, entry->stateless ? 1u : 0u);
    if (status != HCTP_STATUS_OK) return status;
    status = write_u8(payload, capacity, offset, entry->repeated_invocation_safe ? 1u : 0u);
    if (status != HCTP_STATUS_OK) return status;
    status = write_u8(payload, capacity, offset, entry->mutates_input ? 1u : 0u);
    if (status != HCTP_STATUS_OK) return status;
    return write_u32(payload, capacity, offset, entry->scratch_bytes);
}

hctp_status_t hct_build_catalog_frame_chunk(uint32_t session_id,
                                            uint32_t sequence_id,
                                            size_t start_index,
                                            uint8_t *frame_bytes,
                                            size_t frame_capacity,
                                            size_t *frame_length,
                                            size_t *next_index,
                                            bool *is_final)
{
    /* Scratch payload buffer sized to HCT_CATALOG_CHUNK_MAX_PAYLOAD_BYTES plus headroom
     * for the leading entry_count field; the chunk loop below never lets the *committed*
     * payload exceed HCT_CATALOG_CHUNK_MAX_PAYLOAD_BYTES, so this bound is never hit in
     * practice -- it's just large enough that a single write_catalog_entry() attempt
     * (which is rolled back on overflow) can't itself overrun the scratch buffer. */
    uint8_t payload[HCT_CATALOG_CHUNK_MAX_PAYLOAD_BYTES + 128u];
    size_t offset = 0u;
    size_t count = 0u;
    size_t index = start_index;
    size_t entry_count_offset;
    uint16_t chunk_entry_count = 0u;
    const hct_kernel_catalog_entry_t *entries = hct_benchmark_server_catalog(&count);
    hctp_status_t status;

    if (frame_bytes == NULL || frame_length == NULL || next_index == NULL || is_final == NULL)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    if (start_index > count)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }

    /* Reserve space for entry_count now; its final value is patched in once we know how
     * many entries fit in this chunk. */
    entry_count_offset = offset;
    status = write_u16(payload, sizeof(payload), &offset, 0u);
    if (status != HCTP_STATUS_OK) return status;

    while (index < count)
    {
        size_t entry_offset = offset;
        status = write_catalog_entry(payload, sizeof(payload), &offset, &entries[index]);
        if (status != HCTP_STATUS_OK || offset > HCT_CATALOG_CHUNK_MAX_PAYLOAD_BYTES)
        {
            /* This entry doesn't fit in the current chunk -- roll back and close the
             * chunk here (unless the chunk is still empty, which would mean a single
             * entry can never fit; treat that as a hard protocol error). */
            offset = entry_offset;
            if (chunk_entry_count == 0u)
            {
                return HCTP_STATUS_OVERSIZED_PAYLOAD;
            }
            break;
        }
        chunk_entry_count += 1u;
        index += 1u;
    }

    payload[entry_count_offset + 0u] = (uint8_t)(chunk_entry_count & 0xFFu);
    payload[entry_count_offset + 1u] = (uint8_t)((chunk_entry_count >> 8) & 0xFFu);

    *next_index = index;
    *is_final = (index >= count);

    return wrap_frame(HCTP_MSG_CAPABILITIES,
                      session_id,
                      sequence_id,
                      *is_final ? HCTP_FLAG_NONE : HCTP_FLAG_MORE,
                      payload,
                      offset,
                      frame_bytes,
                      frame_capacity,
                      frame_length);
}

hctp_status_t hct_build_catalog_frame(uint32_t session_id,
                                      uint32_t sequence_id,
                                      uint8_t *frame_bytes,
                                      size_t frame_capacity,
                                      size_t *frame_length)
{
    /* Back-compat single-shot wrapper for callers (e.g. benchmark_server_host_emit_main.c)
     * that only want the first chunk -- real session handling always uses
     * hct_build_catalog_frame_chunk() directly and loops until is_final. */
    size_t next_index = 0u;
    bool is_final = false;
    return hct_build_catalog_frame_chunk(session_id, sequence_id, 0u, frame_bytes, frame_capacity, frame_length, &next_index, &is_final);
}
