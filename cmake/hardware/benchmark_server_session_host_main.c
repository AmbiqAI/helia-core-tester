#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arm_nn_types.h"
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

static uint32_t read_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static const int8_t kInput[] = {-12, -1, 0, 7, -99, 5, -8, 3, -4, 11, -2, 100};
static uint8_t workspace[32768u];
static uint8_t inbound_frame[1024];
static uint32_t next_host_sequence = 0u;
extern uint32_t hct_host_fail_call;

static hctp_status_t send_frame(hct_server_session_t *session, uint16_t message_type, const uint8_t *payload, size_t payload_length)
{
    const size_t frame_length = encode_frame(message_type, session->session_id, next_host_sequence++, payload, payload_length, inbound_frame);
    return hct_server_session_accept_frame(session, inbound_frame, frame_length);
}

/* CASE_META for one abs_s8 case. */
static size_t encode_abs_meta(uint8_t *payload, const char *case_id)
{
    size_t offset = 0u;
    write_text(payload, &offset, case_id);
    write_u32(payload, &offset, 1u);
    write_u16(payload, &offset, 1u);
    write_u8(payload, &offset, 1u);
    write_i32(payload, &offset, 0);
    write_u32(payload, &offset, 0u);
    write_u32(payload, &offset, 0u);
    write_u8(payload, &offset, 1u);
    write_text(payload, &offset, "output_capacity_bytes");
    write_i32(payload, &offset, (int32_t)sizeof(kInput));
    write_u16(payload, &offset, 1u);
    write_u32(payload, &offset, 1u);
    write_text(payload, &offset, "input_0");
    write_text(payload, &offset, "S8");
    write_u8(payload, &offset, 2u);
    write_u32(payload, &offset, 3u);
    write_u32(payload, &offset, 4u);
    write_u32(payload, &offset, 0u);
    write_u32(payload, &offset, 0u);
    write_u32(payload, &offset, 0u);
    write_u32(payload, &offset, 0u);
    write_u32(payload, &offset, (uint32_t)sizeof(kInput));
    write_u32(payload, &offset, 1u);
    write_u32(payload, &offset, hctp_crc32((const uint8_t *)kInput, sizeof(kInput)));
    write_u8(payload, &offset, 0u);
    write_u32(payload, &offset, 0u);
    return offset;
}

/* Answer blob requests until CASE_READY. */
static int stream_input(hct_server_session_t *session)
{
    uint8_t outbound[1024];
    uint8_t payload[512];
    size_t length = 0u;
    hctp_frame_view_t frame;
    while (session->state == HCT_SERVER_STATE_WAIT_BLOB_CHUNK)
    {
        const size_t frame_length = hct_server_session_take_next_frame(session, outbound, sizeof(outbound));
        uint32_t requested_offset;
        uint32_t requested_length;
        size_t offset = 0u;
        if (hctp_decode_frame(outbound, frame_length, HCTP_DEFAULT_MAX_PAYLOAD, &frame) != HCTP_STATUS_OK) return 16;
        if (frame.header.message_type != HCTP_MSG_REQUEST_BLOB) return 17;
        requested_offset = read_u32(&frame.payload[4]);
        requested_length = (uint32_t)frame.payload[8] | ((uint32_t)frame.payload[9] << 8);
        if (requested_length > sizeof(kInput) - requested_offset) requested_length = (uint32_t)sizeof(kInput) - requested_offset;
        write_u32(payload, &offset, 1u);
        write_u32(payload, &offset, requested_offset);
        write_u32(payload, &offset, requested_length);
        memcpy(&payload[offset], &kInput[requested_offset], requested_length);
        offset += requested_length;
        if (send_frame(session, HCTP_MSG_BLOB_CHUNK, payload, offset) != HCTP_STATUS_OK) return 18;
    }
    return drain_single_message(session, HCTP_MSG_CASE_READY, outbound, &length) != 0 ? 19 : 0;
}

/* Count samples, then check CASE_COMPLETE. */
static int expect_case_complete(hct_server_session_t *session, uint8_t correctness_ran, uint8_t performance_ran, int samples)
{
    uint8_t outbound[1024];
    hctp_frame_view_t frame;
    for (;;)
    {
        const size_t frame_length = hct_server_session_take_next_frame(session, outbound, sizeof(outbound));
        uint32_t id_length;
        if (hctp_decode_frame(outbound, frame_length, HCTP_DEFAULT_MAX_PAYLOAD, &frame) != HCTP_STATUS_OK) return 60;
        if (frame.header.message_type == HCTP_MSG_SAMPLE_RESULT)
        {
            --samples;
            continue;
        }
        if (frame.header.message_type != HCTP_MSG_CASE_COMPLETE || samples != 0) return 61;
        /* id, ran flags, workspace, [status]. */
        id_length = (uint32_t)frame.payload[0] | ((uint32_t)frame.payload[1] << 8);
        if (frame.payload[2u + id_length] != correctness_ran || frame.payload[3u + id_length] != performance_ran) return 62;
        if (performance_ran != 0u) return frame.header.payload_length == 2u + id_length + 6u ? 0 : 63;
        if (frame.header.payload_length != 2u + id_length + 10u) return 64;
        return (int32_t)read_u32(&frame.payload[2u + id_length + 6u]) == ARM_CMSIS_NN_ARG_ERROR ? 0 : 65;
    }
}

/* The next case runs; the session completes. */
static int run_next_case(hct_server_session_t *session)
{
    uint8_t payload[512];
    size_t length = 0u;
    int status;
    if (drain_single_message(session, HCTP_MSG_REQUEST_CASE, payload, &length) != 0 || payload[0] != 1u) return 70;
    if (send_frame(session, HCTP_MSG_CASE_META, payload, encode_abs_meta(payload, "abs_next_s8")) != HCTP_STATUS_OK) return 71;
    status = stream_input(session);
    if (status != 0) return status;
    if (send_frame(session, HCTP_MSG_RUN_CORRECTNESS, payload, 0u) != HCTP_STATUS_OK) return 72;
    while (session->state != HCT_SERVER_STATE_WAIT_CORRECTNESS_ACK)
    {
        if (hct_server_session_take_next_frame(session, payload, sizeof(payload)) == 0u) return 73;
    }
    while (hct_server_session_take_next_frame(session, payload, sizeof(payload)) != 0u) {}
    payload[0] = 1u;
    if (send_frame(session, HCTP_MSG_CORRECTNESS_ACK, payload, 1u) != HCTP_STATUS_OK) return 74;
    if (send_frame(session, HCTP_MSG_RUN_PERFORMANCE, payload, 0u) != HCTP_STATUS_OK) return 75;
    status = expect_case_complete(session, 1u, 1u, 6);
    if (status != 0) return status;
    if (drain_single_message(session, HCTP_MSG_SESSION_COMPLETE, payload, &length) != 0 || payload[0] != 2u) return 76;
    return session->state == HCT_SERVER_STATE_COMPLETE ? 0 : 77;
}

/* A refused case ends alone, with its status. */
static int probe_rejection(const hct_server_session_t *session, uint16_t trigger, uint32_t fail_call, uint8_t correctness_ran, int samples, const char *label)
{
    static hct_server_session_t probe;
    static uint8_t probe_workspace[sizeof(workspace)];
    const uint8_t payload[1] = {0u};
    int status;
    memcpy(&probe, session, sizeof(probe));
    memcpy(probe_workspace, workspace, sizeof(workspace));
    probe.workspace = probe_workspace;
    /* Two-case plan: a next case follows. */
    probe.planned_case_count = 2u;
    strcpy(probe.planned_case_ids[1], "abs_next_s8");
    probe.planned_kernel_ids[1] = 1u;
    hct_host_fail_call = fail_call;
    if (send_frame(&probe, trigger, payload, 0u) != HCTP_STATUS_OK) return 50;
    status = expect_case_complete(&probe, correctness_ran, 0u, samples);
    if (status == 0) status = run_next_case(&probe);
    if (status == 0) printf("rejected %s samples_dropped=%d\n", label, samples);
    return status;
}

#ifdef HCT_HOST_PMU_STUB
/* Match tests/fixtures/pmu_stub/pmu_stub.c readings. */
#define STUB_CCNTR 1234u
#define STUB_OVS 0x80000009u

/* Entry: empty name, event, value, flags. */
static int check_stub_counter(const uint8_t *entry, int chained, uint32_t index)
{
    const uint16_t event_id = (uint16_t)entry[2] | ((uint16_t)entry[3] << 8);
    const uint32_t slot = chained ? (2u * index + 1u) : index;
    if (read_u32(&entry[4]) != event_id + (chained ? 0x10000u : 0u)) return 1;
    if (entry[12] != ((STUB_OVS >> slot) & 1u)) return 2;
    return entry[13] == 1u ? 0 : 3;
}
#endif

int main(void)
{
    static const int8_t kExpected[] = {12, 1, 0, 7, 99, 5, 8, 3, 4, 11, 2, 100};
    hct_server_session_t session;
    uint8_t inbound_payload[512];
    uint8_t outbound_payload[1024];
    size_t outbound_length = 0u;
    size_t offset = 0u;
    size_t second_id_offset = 0u;
    int status;
    hctp_frame_view_t frame;

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
    second_id_offset = offset;
    write_u16(inbound_payload, &offset, 0x0023u);
    write_text(inbound_payload, &offset, "mve_0");
    write_u8(inbound_payload, &offset, 0u);
    write_u8(inbound_payload, &offset, 1u);
    write_u16(inbound_payload, &offset, 0x0200u);
    write_text(inbound_payload, &offset, "abs_default_s8_stream_demo");
    write_u32(inbound_payload, &offset, 1u);
#ifdef HCT_HOST_PMU_STUB
    /* An id outside the module map fails. */
    {
        static hct_server_session_t probe;
        memcpy(&probe, &session, sizeof(probe));
        inbound_payload[second_id_offset] = 0x02u;
        if (hct_server_session_accept_frame(&probe, inbound_frame, encode_frame(HCTP_MSG_SESSION_PLAN, probe.session_id, next_host_sequence, inbound_payload, offset, inbound_frame)) != HCTP_STATUS_INVALID_ARGUMENT) return 43;
        inbound_payload[second_id_offset] = 0x23u;
    }
#else
    (void)second_id_offset;
#endif
    if (hct_server_session_accept_frame(&session, inbound_frame, encode_frame(HCTP_MSG_SESSION_PLAN, session.session_id, next_host_sequence++, inbound_payload, offset, inbound_frame)) != HCTP_STATUS_OK) return 13;
    if (drain_single_message(&session, HCTP_MSG_REQUEST_CASE, outbound_payload, &outbound_length) != 0) return 14;

    if (send_frame(&session, HCTP_MSG_CASE_META, inbound_payload, encode_abs_meta(inbound_payload, "abs_default_s8_stream_demo")) != HCTP_STATUS_OK) return 15;

    /* Regression: a BLOB_CHUNK whose declared length is near UINT32_MAX must be refused
     * as truncated, never handed to memcpy (has_capacity() used to compute offset+needed,
     * which wraps on the 32-bit target). Probe on a copy so the real session continues. */
    {
        static hct_server_session_t probe;
        memcpy(&probe, &session, sizeof(probe));
        offset = 0u;
        write_u32(inbound_payload, &offset, 1u);
        write_u32(inbound_payload, &offset, 0u);
        write_u32(inbound_payload, &offset, 0xFFFFFFF0u);
        if (hct_server_session_accept_frame(&probe, inbound_frame, encode_frame(HCTP_MSG_BLOB_CHUNK, probe.session_id, next_host_sequence, inbound_payload, offset, inbound_frame)) != HCTP_STATUS_TRUNCATED_FRAME) return 40;
        if (probe.blobs[probe.current_blob_index].bytes_received != 0u) return 41;
        if (probe.state != HCT_SERVER_STATE_WAIT_BLOB_CHUNK) return 42;
    }

    status = stream_input(&session);
    if (status != 0) return status;

    /* Kernel refusals: correctness, warmup, mid-sampling. */
    session.output_capacity_bytes = 4u;
    status = probe_rejection(&session, HCTP_MSG_RUN_CORRECTNESS, 0u, 0u, 0, "correctness");
    session.output_capacity_bytes = (uint32_t)sizeof(kExpected);
    if (status != 0) return status;
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
    /* Call 1: warmup. Call 19: pass 1 sampling. */
    status = probe_rejection(&session, HCTP_MSG_RUN_PERFORMANCE, 1u, 1u, 0, "warmup");
    if (status == 0) status = probe_rejection(&session, HCTP_MSG_RUN_PERFORMANCE, 19u, 1u, 3, "sampling");
    if (status != 0) return status;
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
#ifdef HCT_HOST_PMU_STUB
                /* One window per iteration: 4 kernel calls. */
                if (read_u32(&p[pos + 4u]) != 4u * STUB_CCNTR || p[pos + 12u] != 1u) return 38;
#endif
                expected_counters = (strncmp(pass_name, "cpu_0", name_len) == 0) ? 3 : 2;
                if (counter_count != expected_counters) return 33;
                pos += 2u + 2u + 8u + 1u + 1u;
                {
                    int index;
                    for (index = 1; index < counter_count; ++index)
                    {
#ifdef HCT_HOST_PMU_STUB
                        const int chained = strncmp(pass_name, "cpu_0", name_len) == 0;
                        if (check_stub_counter(&p[pos], chained, (uint32_t)(index - 1)) != 0) return 39;
#else
                        const uint8_t supported = p[pos + 2u + 2u + 8u + 1u];
                        if (supported != 0u) return 34;   /* no PMU on the host build */
#endif
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
