#include "hctp_protocol.h"

#include <string.h>

static uint16_t hctp_load_u16_le(const uint8_t *src)
{
    return (uint16_t)src[0] | ((uint16_t)src[1] << 8);
}

static uint32_t hctp_load_u32_le(const uint8_t *src)
{
    return (uint32_t)src[0]
         | ((uint32_t)src[1] << 8)
         | ((uint32_t)src[2] << 16)
         | ((uint32_t)src[3] << 24);
}

static void hctp_store_u16_le(uint8_t *dst, uint16_t value)
{
    dst[0] = (uint8_t)(value & 0xFFu);
    dst[1] = (uint8_t)((value >> 8) & 0xFFu);
}

static void hctp_store_u32_le(uint8_t *dst, uint32_t value)
{
    dst[0] = (uint8_t)(value & 0xFFu);
    dst[1] = (uint8_t)((value >> 8) & 0xFFu);
    dst[2] = (uint8_t)((value >> 16) & 0xFFu);
    dst[3] = (uint8_t)((value >> 24) & 0xFFu);
}

uint32_t hctp_crc32(const uint8_t *data, size_t length)
{
    uint32_t crc = 0xFFFFFFFFu;
    size_t index;
    unsigned bit;

    if ((data == NULL) && (length != 0u))
    {
        return 0u;
    }

    for (index = 0u; index < length; ++index)
    {
        crc ^= (uint32_t)data[index];
        for (bit = 0u; bit < 8u; ++bit)
        {
            const uint32_t mask = (uint32_t)(-(int32_t)(crc & 1u));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }

    return crc ^ 0xFFFFFFFFu;
}

void hctp_encode_header(uint8_t out_header[HCTP_HEADER_SIZE], const hctp_frame_header_t *header_without_crc)
{
    uint32_t header_crc32;

    if ((out_header == NULL) || (header_without_crc == NULL))
    {
        return;
    }

    hctp_store_u32_le(&out_header[0], header_without_crc->magic);
    hctp_store_u16_le(&out_header[4], header_without_crc->protocol_version);
    hctp_store_u16_le(&out_header[6], header_without_crc->message_type);
    hctp_store_u32_le(&out_header[8], header_without_crc->flags);
    hctp_store_u32_le(&out_header[12], header_without_crc->session_id);
    hctp_store_u32_le(&out_header[16], header_without_crc->sequence_id);
    hctp_store_u32_le(&out_header[20], header_without_crc->payload_length);
    hctp_store_u32_le(&out_header[24], header_without_crc->payload_crc32);
    header_crc32 = hctp_crc32(out_header, HCTP_HEADER_SIZE - sizeof(uint32_t));
    hctp_store_u32_le(&out_header[28], header_crc32);
}

hctp_status_t hctp_decode_header(const uint8_t *header_bytes, size_t header_length, uint32_t max_payload, hctp_frame_header_t *out_header)
{
    hctp_frame_header_t header;
    uint32_t actual_crc32;

    if ((header_bytes == NULL) || (out_header == NULL))
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    if (header_length != HCTP_HEADER_SIZE)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }

    header.magic = hctp_load_u32_le(&header_bytes[0]);
    header.protocol_version = hctp_load_u16_le(&header_bytes[4]);
    header.message_type = hctp_load_u16_le(&header_bytes[6]);
    header.flags = hctp_load_u32_le(&header_bytes[8]);
    header.session_id = hctp_load_u32_le(&header_bytes[12]);
    header.sequence_id = hctp_load_u32_le(&header_bytes[16]);
    header.payload_length = hctp_load_u32_le(&header_bytes[20]);
    header.payload_crc32 = hctp_load_u32_le(&header_bytes[24]);
    header.header_crc32 = hctp_load_u32_le(&header_bytes[28]);

    if (header.magic != HCTP_MAGIC_U32)
    {
        return HCTP_STATUS_INVALID_MAGIC;
    }
    if (header.protocol_version != HCTP_SUPPORTED_VERSION)
    {
        return HCTP_STATUS_UNSUPPORTED_VERSION;
    }
    if (header.payload_length > max_payload)
    {
        return HCTP_STATUS_OVERSIZED_PAYLOAD;
    }

    actual_crc32 = hctp_crc32(header_bytes, HCTP_HEADER_SIZE - sizeof(uint32_t));
    if (actual_crc32 != header.header_crc32)
    {
        return HCTP_STATUS_HEADER_CRC_MISMATCH;
    }

    *out_header = header;
    return HCTP_STATUS_OK;
}

hctp_status_t hctp_decode_frame(const uint8_t *frame_bytes, size_t frame_length, uint32_t max_payload, hctp_frame_view_t *out_frame)
{
    hctp_status_t status;
    hctp_frame_header_t header;
    uint32_t payload_crc32;

    if ((frame_bytes == NULL) || (out_frame == NULL))
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    if (frame_length < HCTP_HEADER_SIZE)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }

    status = hctp_decode_header(frame_bytes, HCTP_HEADER_SIZE, max_payload, &header);
    if (status != HCTP_STATUS_OK)
    {
        return status;
    }
    if (frame_length != (size_t)HCTP_HEADER_SIZE + (size_t)header.payload_length)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }

    payload_crc32 = hctp_crc32(frame_bytes + HCTP_HEADER_SIZE, (size_t)header.payload_length);
    if (payload_crc32 != header.payload_crc32)
    {
        return HCTP_STATUS_PAYLOAD_CRC_MISMATCH;
    }

    out_frame->header = header;
    out_frame->payload = frame_bytes + HCTP_HEADER_SIZE;
    return HCTP_STATUS_OK;
}
