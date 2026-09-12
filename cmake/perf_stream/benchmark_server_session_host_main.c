#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "benchmark_server_session.h"

static void write_u8(uint8_t *buffer, size_t *offset, uint8_t value)
{
    buffer[(*offset)++] = value;
}

static void write_u16(uint8_t *buffer, size_t *offset, uint16_t value)
{
    buffer[(*offset)++] = (uint8_t)(value & 0xFFu);
    buffer[(*offset)++] = (uint8_t)((value >> 8) & 0xFFu);
}

static void write_u32(uint8_t *buffer, size_t *offset, uint32_t value)
{
    buffer[(*offset)++] = (uint8_t)(value & 0xFFu);
    buffer[(*offset)++] = (uint8_t)((value >> 8) & 0xFFu);
    buffer[(*offset)++] = (uint8_t)((value >> 16) & 0xFFu);
    buffer[(*offset)++] = (uint8_t)((value >> 24) & 0xFFu);
}

static void write_i32(uint8_t *buffer, size_t *offset, int32_t value)
{
    write_u32(buffer, offset, (uint32_t)value);
}

static void write_text(uint8_t *buffer, size_t *offset, const char *value)
{
    const uint16_t length = (uint16_t)strlen(value);
    write_u16(buffer, offset, length);
    memcpy(&buffer[*offset], value, length);
    *offset += length;
}

static size_t encode_frame(uint16_t message_type, uint32_t session_id, uint32_t sequence_id, const uint8_t *payload, size_t payload_length, uint8_t *frame)
{
    hctp_frame_header_t header = {
        .magic = HCTP_MAGIC_U32,
        .protocol_version = HCTP_SUPPORTED_VERSION,
        .message_type = message_type,
        .flags = HCTP_FLAG_NONE,
        .session_id = session_id,
        .sequence_id = sequence_id,
        .payload_length = (uint32_t)payload_length,
        .payload_crc32 = hctp_crc32(payload, payload_length),
        .header_crc32 = 0u,
    };
    hctp_encode_header(frame, &header);
    memcpy(frame + HCTP_HEADER_SIZE, payload, payload_length);
    return HCTP_HEADER_SIZE + payload_length;
}

static int drain_single_message(hct_server_session_t *session, uint16_t expected_type, uint8_t *payload_out, size_t *payload_length)
{
    uint8_t frame_bytes[2048];
    hctp_frame_view_t frame;
    const size_t frame_length = hct_server_session_take_next_frame(session, frame_bytes, sizeof(frame_bytes));
    if (frame_length == 0u)
    {
        return 1;
    }
    if (hctp_decode_frame(frame_bytes, frame_length, HCTP_DEFAULT_MAX_PAYLOAD, &frame) != HCTP_STATUS_OK)
    {
        return 2;
    }
    if (frame.header.message_type != expected_type)
    {
        return 3;
    }
    memcpy(payload_out, frame.payload, frame.header.payload_length);
    *payload_length = frame.header.payload_length;
    return 0;
}

static int drain_catalog_frames(hct_server_session_t *session)
{
    /* TARGET_INFO_ACK triggers one or more paginated KERNEL_CATALOG frames (each
     * non-final chunk carries HCTP_FLAG_MORE); drain them all before expecting the
     * next protocol message. */
    uint8_t frame_bytes[2048];
    hctp_frame_view_t frame;
    for (;;)
    {
        const size_t frame_length = hct_server_session_take_next_frame(session, frame_bytes, sizeof(frame_bytes));
        if (frame_length == 0u) return 1;
        if (hctp_decode_frame(frame_bytes, frame_length, HCTP_DEFAULT_MAX_PAYLOAD, &frame) != HCTP_STATUS_OK) return 2;
        if (frame.header.message_type != HCTP_MSG_KERNEL_CATALOG) return 3;
        if ((frame.header.flags & HCTP_FLAG_MORE) == 0u) return 0;
    }
}

int main(void)
{
    static const int8_t kInput[] = {-12, -1, 0, 7, -99, 5, -8, 3, -4, 11, -2, 100};
    static const int8_t kExpected[] = {12, 1, 0, 7, 99, 5, 8, 3, 4, 11, 2, 100};
    hct_server_session_t session;
    uint8_t inbound_payload[512];
    uint8_t inbound_frame[1024];
    uint8_t outbound_payload[1024];
    size_t outbound_length = 0u;
    size_t offset = 0u;
    uint32_t next_host_sequence = 0u;
    uint32_t blob_offset = 0u;
    uint32_t requested_offset = 0u;
    uint16_t requested_length = 0u;
    hctp_frame_view_t frame;
    static uint8_t workspace[32768u];

    hct_server_session_init(&session, 0xC0DE1234u, 256u, workspace, (uint32_t)sizeof(workspace));
    if (drain_single_message(&session, HCTP_MSG_TARGET_INFO, outbound_payload, &outbound_length) != 0) return 10;

    offset = 0u;
    if (hct_server_session_accept_frame(&session, inbound_frame, encode_frame(HCTP_MSG_TARGET_INFO_ACK, session.session_id, next_host_sequence++, inbound_payload, 0u, inbound_frame)) != HCTP_STATUS_OK) return 11;
    if (drain_catalog_frames(&session) != 0) return 12;

    /* SESSION_PLAN: one case, 2 warmups, 3 samples x 4 iterations, and two PMU
     * passes -- a chained cpu pass (INST_RETIRED, STALL_FRONTEND) and an unchained mve
     * pass (MVE_INST_RETIRED). The host harness has no PMU, so the firmware must accept
     * the passes and report every event counter as unsupported. */
    offset = 0u;
    write_u16(inbound_payload, &offset, 1u);
    write_u8(inbound_payload, &offset, 1u);
    write_u16(inbound_payload, &offset, 2u);
    write_u16(inbound_payload, &offset, 3u);
    write_u32(inbound_payload, &offset, 4u);
    write_u32(inbound_payload, &offset, 512u);
    write_u32(inbound_payload, &offset, 128u);
    write_u8(inbound_payload, &offset, 2u);
    write_text(inbound_payload, &offset, "cpu_0");
    write_u8(inbound_payload, &offset, 1u);
    write_u8(inbound_payload, &offset, 2u);
    write_u16(inbound_payload, &offset, 0x0008u);
    write_u16(inbound_payload, &offset, 0x0023u);
    write_text(inbound_payload, &offset, "mve_0");
    write_u8(inbound_payload, &offset, 0u);
    write_u8(inbound_payload, &offset, 1u);
    write_u16(inbound_payload, &offset, 0x0200u);
    write_text(inbound_payload, &offset, "abs_default_s8_stream_demo");
    write_u32(inbound_payload, &offset, 1u);
    if (hct_server_session_accept_frame(&session, inbound_frame, encode_frame(HCTP_MSG_SESSION_PLAN, session.session_id, next_host_sequence++, inbound_payload, offset, inbound_frame)) != HCTP_STATUS_OK) return 13;
    if (drain_single_message(&session, HCTP_MSG_REQUEST_CASE, outbound_payload, &outbound_length) != 0) return 14;

    offset = 0u;
    write_text(inbound_payload, &offset, "abs_default_s8_stream_demo");
    write_u32(inbound_payload, &offset, 1u);
    write_u16(inbound_payload, &offset, 1u);
    write_u8(inbound_payload, &offset, 1u);
    write_i32(inbound_payload, &offset, 0);
    write_u32(inbound_payload, &offset, 0u);
    write_u32(inbound_payload, &offset, 0u);
    write_u8(inbound_payload, &offset, 1u);
    write_text(inbound_payload, &offset, "output_capacity_bytes");
    write_i32(inbound_payload, &offset, (int32_t)sizeof(kExpected));
    write_u16(inbound_payload, &offset, 1u);
    write_u32(inbound_payload, &offset, 1u);
    write_text(inbound_payload, &offset, "input_0");
    write_text(inbound_payload, &offset, "S8");
    write_u8(inbound_payload, &offset, 2u);
    write_u32(inbound_payload, &offset, 3u);
    write_u32(inbound_payload, &offset, 4u);
    write_u32(inbound_payload, &offset, 0u);
    write_u32(inbound_payload, &offset, 0u);
    write_u32(inbound_payload, &offset, 0u);
    write_u32(inbound_payload, &offset, 0u);
    write_u32(inbound_payload, &offset, (uint32_t)sizeof(kInput));
    write_u32(inbound_payload, &offset, 1u);
    write_u32(inbound_payload, &offset, hctp_crc32((const uint8_t *)kInput, sizeof(kInput)));
    write_u8(inbound_payload, &offset, 0u);
    write_u32(inbound_payload, &offset, 0u);
    if (hct_server_session_accept_frame(&session, inbound_frame, encode_frame(HCTP_MSG_CASE_META, session.session_id, next_host_sequence++, inbound_payload, offset, inbound_frame)) != HCTP_STATUS_OK) return 15;

    while (session.state == HCT_SERVER_STATE_WAIT_BLOB_CHUNK)
    {
        const size_t frame_length = hct_server_session_take_next_frame(&session, outbound_payload, sizeof(outbound_payload));
        if (hctp_decode_frame(outbound_payload, frame_length, HCTP_DEFAULT_MAX_PAYLOAD, &frame) != HCTP_STATUS_OK) return 16;
        if (frame.header.message_type != HCTP_MSG_REQUEST_BLOB) return 17;
        requested_offset = (uint32_t)frame.payload[4] | ((uint32_t)frame.payload[5] << 8) | ((uint32_t)frame.payload[6] << 16) | ((uint32_t)frame.payload[7] << 24);
        requested_length = (uint16_t)frame.payload[8] | ((uint16_t)frame.payload[9] << 8);
        offset = 0u;
        write_u32(inbound_payload, &offset, 1u);
        write_u32(inbound_payload, &offset, requested_offset);
        blob_offset = requested_offset;
        if ((size_t)requested_length > sizeof(kInput) - blob_offset) requested_length = (uint16_t)(sizeof(kInput) - blob_offset);
        write_u32(inbound_payload, &offset, requested_length);
        memcpy(&inbound_payload[offset], &kInput[blob_offset], requested_length);
        offset += requested_length;
        if (hct_server_session_accept_frame(&session, inbound_frame, encode_frame(HCTP_MSG_BLOB_CHUNK, session.session_id, next_host_sequence++, inbound_payload, offset, inbound_frame)) != HCTP_STATUS_OK) return 18;
    }

    if (drain_single_message(&session, HCTP_MSG_CASE_READY, outbound_payload, &outbound_length) != 0) return 19;
    if (hct_server_session_accept_frame(&session, inbound_frame, encode_frame(HCTP_MSG_RUN_CORRECTNESS, session.session_id, next_host_sequence++, inbound_payload, 0u, inbound_frame)) != HCTP_STATUS_OK) return 20;

    if (drain_single_message(&session, HCTP_MSG_CORRECTNESS_RESULT, outbound_payload, &outbound_length) != 0) return 21;
    if (drain_single_message(&session, HCTP_MSG_OUTPUT_BEGIN, outbound_payload, &outbound_length) != 0) return 22;
    {
        int chunk_count = 0;
        int8_t actual[sizeof(kExpected)] = {0};
        while (session.outbox_length > 0u || session.output_stream_active != 0u)
        {
            const size_t frame_length = hct_server_session_take_next_frame(&session, outbound_payload, sizeof(outbound_payload));
            if (hctp_decode_frame(outbound_payload, frame_length, HCTP_DEFAULT_MAX_PAYLOAD, &frame) != HCTP_STATUS_OK) return 23;
            if (frame.header.message_type == HCTP_MSG_OUTPUT_CHUNK)
            {
                uint32_t data_offset = (uint32_t)frame.payload[0] | ((uint32_t)frame.payload[1] << 8) | ((uint32_t)frame.payload[2] << 16) | ((uint32_t)frame.payload[3] << 24);
                uint32_t data_length = (uint32_t)frame.payload[4] | ((uint32_t)frame.payload[5] << 8) | ((uint32_t)frame.payload[6] << 16) | ((uint32_t)frame.payload[7] << 24);
                memcpy(&actual[data_offset], frame.payload + 8u, data_length);
                ++chunk_count;
            }
            else if (frame.header.message_type == HCTP_MSG_OUTPUT_END)
            {
                if (memcmp(actual, kExpected, sizeof(kExpected)) != 0) return 24;
                printf("chunks=%d bytes=%zu state=%d\n", chunk_count, sizeof(kExpected), (int)session.state);
                break;
            }
            else
            {
                return 25;
            }
        }
        if (session.state != HCT_SERVER_STATE_WAIT_CORRECTNESS_ACK) return 26;
    }

    /* CORRECTNESS_ACK(pass) -> RUN_PERFORMANCE -> SAMPLE_RESULT x (3 samples x 2 passes)
     * -> CASE_COMPLETE -> SESSION_COMPLETE. Every SAMPLE_RESULT must lead with the
     * ARM_PMU_CPU_CYCLES entry (event 0x0011, supported) and list the pass's event ids
     * after it with supported=0 on this PMU-less host build. */
    offset = 0u;
    write_u8(inbound_payload, &offset, 1u);
    if (hct_server_session_accept_frame(&session, inbound_frame, encode_frame(HCTP_MSG_CORRECTNESS_ACK, session.session_id, next_host_sequence++, inbound_payload, offset, inbound_frame)) != HCTP_STATUS_OK) return 27;
    if (hct_server_session_accept_frame(&session, inbound_frame, encode_frame(HCTP_MSG_RUN_PERFORMANCE, session.session_id, next_host_sequence++, inbound_payload, 0u, inbound_frame)) != HCTP_STATUS_OK) return 28;
    {
        int sample_count = 0;
        int cpu_pass_samples = 0;
        int mve_pass_samples = 0;
        for (;;)
        {
            const size_t frame_length = hct_server_session_take_next_frame(&session, outbound_payload, sizeof(outbound_payload));
            if (frame_length == 0u) return 29;
            if (hctp_decode_frame(outbound_payload, frame_length, HCTP_DEFAULT_MAX_PAYLOAD, &frame) != HCTP_STATUS_OK) return 30;
            if (frame.header.message_type == HCTP_MSG_SAMPLE_RESULT)
            {
                const uint8_t *p = frame.payload;
                size_t pos = 2u + 4u + 8u;
                uint16_t name_len = (uint16_t)p[pos] | ((uint16_t)p[pos + 1u] << 8);
                const char *pass_name = (const char *)&p[pos + 2u];
                uint8_t counter_count;
                uint16_t first_event;
                uint8_t first_supported;
                int expected_counters;
                pos += 2u + name_len;
                counter_count = p[pos++];
                if (p[pos] != 0u || p[pos + 1u] != 0u) return 31;   /* names are sent empty */
                first_event = (uint16_t)p[pos + 2u] | ((uint16_t)p[pos + 3u] << 8);
                first_supported = p[pos + 2u + 2u + 8u + 1u];
                if (first_event != 0x0011u || first_supported != 1u) return 32;
                expected_counters = (strncmp(pass_name, "cpu_0", name_len) == 0) ? 3 : 2;
                if (counter_count != expected_counters) return 33;
                pos += 2u + 2u + 8u + 1u + 1u;
                {
                    int index;
                    for (index = 1; index < counter_count; ++index)
                    {
                        const uint8_t supported = p[pos + 2u + 2u + 8u + 1u];
                        if (supported != 0u) return 34;   /* no PMU on the host build */
                        pos += 2u + 2u + 8u + 1u + 1u;
                    }
                }
                if (pos != frame.header.payload_length) return 35;
                if (strncmp(pass_name, "cpu_0", name_len) == 0) ++cpu_pass_samples; else ++mve_pass_samples;
                ++sample_count;
            }
            else if (frame.header.message_type == HCTP_MSG_CASE_COMPLETE)
            {
                continue;
            }
            else if (frame.header.message_type == HCTP_MSG_SESSION_COMPLETE)
            {
                if (sample_count != 6 || cpu_pass_samples != 3 || mve_pass_samples != 3) return 36;
                printf("samples=%d passes=2 state=%d\n", sample_count, (int)session.state);
                return 0;
            }
            else
            {
                return 37;
            }
        }
    }
}
