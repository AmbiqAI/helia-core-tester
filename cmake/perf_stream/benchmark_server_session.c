#include "benchmark_server_session.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "benchmark_server_adapter.h"
#include "benchmark_server_messages.h"
#include "arm_nnfunctions.h"

#ifdef HELIA_HARDWARE_BUILD
#include "am_mcu_apollo.h"
#include "pmu_armv8.h"
#endif

#define HCT_BLOB_ROLE_UNKNOWN 0u
#define HCT_BLOB_ROLE_INPUT_0 1u
#define HCT_BLOB_ROLE_WEIGHTS 2u
#define HCT_BLOB_ROLE_BIAS 3u
#define HCT_BLOB_ROLE_MULTIPLIER 4u
#define HCT_BLOB_ROLE_SHIFT 5u
#define HCT_BLOB_ROLE_INPUT_1 6u

#define HCT_DTYPE_UNKNOWN 0u
#define HCT_DTYPE_S8 1u
#define HCT_DTYPE_S32 2u

#define HCT_PADDING_VALID 0
#define HCT_PADDING_SAME 1

/* Kernel IDs sent by the host in CASE_META -- must match assets/kernel_registry.yaml
 * (the Python-side single source of truth) and helia_core_tester/perf_stream/kernel_registry.py. */
#define HCT_KERNEL_ID_ABS_S8 1u
#define HCT_KERNEL_ID_CONVOLVE_S8 2u
#define HCT_KERNEL_ID_ADD_S8 3u
#define HCT_KERNEL_ID_SUB_S8 4u
#define HCT_KERNEL_ID_MUL_S8 5u
#define HCT_KERNEL_ID_MAXIMUM_S8 6u
#define HCT_KERNEL_ID_MINIMUM_S8 7u
#define HCT_KERNEL_ID_DEPTHWISE_CONV_S8 8u

static bool has_capacity(size_t payload_length, size_t offset, size_t needed)
{
    return offset + needed <= payload_length;
}

static uint8_t read_u8(const uint8_t *buffer, size_t *offset)
{
    return buffer[(*offset)++];
}

static uint16_t read_u16(const uint8_t *buffer, size_t *offset)
{
    const uint16_t value = (uint16_t)buffer[*offset] | ((uint16_t)buffer[*offset + 1u] << 8);
    *offset += 2u;
    return value;
}

static uint32_t read_u32(const uint8_t *buffer, size_t *offset)
{
    const uint32_t value = (uint32_t)buffer[*offset]
                         | ((uint32_t)buffer[*offset + 1u] << 8)
                         | ((uint32_t)buffer[*offset + 2u] << 16)
                         | ((uint32_t)buffer[*offset + 3u] << 24);
    *offset += 4u;
    return value;
}

static int32_t read_i32(const uint8_t *buffer, size_t *offset)
{
    return (int32_t)read_u32(buffer, offset);
}

static hctp_status_t read_text(const uint8_t *buffer,
                               size_t payload_length,
                               size_t *offset,
                               char *dest,
                               size_t dest_capacity)
{
    size_t index;
    const uint16_t length = read_u16(buffer, offset);
    if (*offset + length > payload_length || (size_t)length + 1u > dest_capacity)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    for (index = 0u; index < length; ++index)
    {
        dest[index] = (char)buffer[*offset + index];
    }
    dest[length] = '\0';
    *offset += length;
    return HCTP_STATUS_OK;
}

static hctp_status_t write_u8(uint8_t *buffer, size_t capacity, size_t *offset, uint8_t value)
{
    if (*offset + 1u > capacity)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    buffer[(*offset)++] = value;
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

static hctp_status_t write_u64(uint8_t *buffer, size_t capacity, size_t *offset, uint64_t value)
{
    if (*offset + 8u > capacity)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    write_u32(buffer, capacity, offset, (uint32_t)(value & 0xFFFFFFFFu));
    write_u32(buffer, capacity, offset, (uint32_t)(value >> 32));
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
    if (*offset + length > capacity)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    memcpy(&buffer[*offset], value, length);
    *offset += length;
    return HCTP_STATUS_OK;
}

static uint32_t align_up(uint32_t value, uint32_t alignment)
{
    if (alignment <= 1u)
    {
        return value;
    }
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static uint8_t role_from_name(const char *name)
{
    if (strcmp(name, "input_0") == 0) return HCT_BLOB_ROLE_INPUT_0;
    if (strcmp(name, "weights") == 0) return HCT_BLOB_ROLE_WEIGHTS;
    if (strcmp(name, "bias") == 0) return HCT_BLOB_ROLE_BIAS;
    if (strcmp(name, "multiplier") == 0) return HCT_BLOB_ROLE_MULTIPLIER;
    if (strcmp(name, "shift") == 0) return HCT_BLOB_ROLE_SHIFT;
    if (strcmp(name, "input_1") == 0) return HCT_BLOB_ROLE_INPUT_1;
    return HCT_BLOB_ROLE_UNKNOWN;
}

static uint8_t dtype_from_name(const char *name)
{
    if (strcmp(name, "S8") == 0) return HCT_DTYPE_S8;
    if (strcmp(name, "S32") == 0) return HCT_DTYPE_S32;
    return HCT_DTYPE_UNKNOWN;
}

static hct_server_blob_t *find_blob_by_role(hct_server_session_t *session, uint8_t role)
{
    uint16_t index;
    for (index = 0u; index < session->blob_count; ++index)
    {
        if (session->blobs[index].role == role)
        {
            return &session->blobs[index];
        }
    }
    return NULL;
}

static uint8_t *blob_ptr(hct_server_session_t *session, const hct_server_blob_t *blob)
{
    return &session->case_arena[blob->arena_offset];
}

static hctp_status_t append_frame(hct_server_session_t *session, const uint8_t *frame_bytes, size_t frame_length)
{
    if (session->outbox_length + frame_length > sizeof(session->outbox))
    {
        session->state = HCT_SERVER_STATE_ERROR;
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    memcpy(&session->outbox[session->outbox_length], frame_bytes, frame_length);
    session->outbox_length += frame_length;
    return HCTP_STATUS_OK;
}

static hctp_status_t queue_frame(hct_server_session_t *session,
                                 uint16_t message_type,
                                 const uint8_t *payload,
                                 size_t payload_length)
{
    uint8_t frame[HCTP_HEADER_SIZE + 512u];
    hctp_frame_header_t header;
    const size_t total_length = HCTP_HEADER_SIZE + payload_length;

    if (sizeof(frame) < total_length)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    /* Check outbox capacity *before* consuming a sequence number. A dropped
     * frame must never advance next_outgoing_sequence: doing so leaves a
     * permanent gap the host can never observe or recover from (it just sees
     * an unexplained SequenceMismatchError on some later frame). */
    if (session->outbox_length + total_length > sizeof(session->outbox))
    {
        session->state = HCT_SERVER_STATE_ERROR;
        return HCTP_STATUS_TRUNCATED_FRAME;
    }

    header.magic = HCTP_MAGIC_U32;
    header.protocol_version = HCTP_SUPPORTED_VERSION;
    header.message_type = message_type;
    header.flags = HCTP_FLAG_NONE;
    header.session_id = session->session_id;
    header.sequence_id = session->next_outgoing_sequence++;
    header.payload_length = (uint32_t)payload_length;
    header.payload_crc32 = hctp_crc32(payload, payload_length);
    header.header_crc32 = 0u;
    hctp_encode_header(frame, &header);
    if (payload_length > 0u)
    {
        memcpy(frame + HCTP_HEADER_SIZE, payload, payload_length);
    }
    return append_frame(session, frame, total_length);
}

/* Best-effort ERROR reply so a rejected message (bad LOAD_PLAN/CASE_META/
 * BLOB_CHUNK, etc.) is visible to the host instead of leaving it waiting
 * forever for a reply that will never come (see hct_server_session_accept_frame()).
 * Payload matches the text-message ERROR convention already used by the host's
 * fake-target test double (a single length-prefixed text string) so both real
 * firmware and the fake target produce host-compatible ERROR frames. Deliberately
 * ignores its own queue_frame() failure (e.g. outbox full) -- there is nothing
 * more useful to do at that point than let the host's read timeout surface the
 * problem, and we must not let error reporting itself throw/hang the firmware. */
static void queue_error_frame(hct_server_session_t *session, uint16_t offending_message_type, hctp_status_t status)
{
    uint8_t payload[64];
    size_t offset = 0u;
    char message[48];
    (void)snprintf(message, sizeof(message), "message_type=%u status=%d", (unsigned)offending_message_type, (int)status);
    (void)write_text(payload, sizeof(payload), &offset, message);
    (void)queue_frame(session, HCTP_MSG_ERROR, payload, offset);
}


static hctp_status_t queue_request_case(hct_server_session_t *session)
{
    uint8_t payload[2];
    size_t offset = 0u;
    write_u16(payload, sizeof(payload), &offset, session->current_case_index);
    return queue_frame(session, HCTP_MSG_REQUEST_CASE, payload, offset);
}

static hctp_status_t queue_request_blob(hct_server_session_t *session)
{
    uint8_t payload[10];
    size_t offset = 0u;
    hct_server_blob_t *blob = &session->blobs[session->current_blob_index];
    const uint32_t remaining = blob->byte_length - blob->bytes_received;
    const uint16_t request_length = (uint16_t)((remaining > HCT_SERVER_BLOB_CHUNK_BYTES) ? HCT_SERVER_BLOB_CHUNK_BYTES : remaining);
    write_u32(payload, sizeof(payload), &offset, blob->blob_id);
    write_u32(payload, sizeof(payload), &offset, blob->bytes_received);
    write_u16(payload, sizeof(payload), &offset, request_length);
    return queue_frame(session, HCTP_MSG_REQUEST_BLOB, payload, offset);
}

static hctp_status_t queue_case_ready(hct_server_session_t *session)
{
    uint8_t payload[8];
    size_t offset = 0u;
    hct_server_blob_t *blob = &session->blobs[session->blob_count - 1u];
    write_u32(payload, sizeof(payload), &offset, blob->blob_id);
    write_u32(payload, sizeof(payload), &offset, blob->bytes_received);
    return queue_frame(session, HCTP_MSG_CASE_READY, payload, offset);
}

static hctp_status_t queue_correctness_output(hct_server_session_t *session)
{
    uint8_t payload[256];
    size_t offset = 0u;
    size_t cursor = 0u;
    uint32_t checksum = 0u;

    write_u32(payload, sizeof(payload), &offset, 0u);
    if (queue_frame(session, HCTP_MSG_CORRECTNESS_RESULT, payload, offset) != HCTP_STATUS_OK) return HCTP_STATUS_TRUNCATED_FRAME;

    offset = 0u;
    write_u32(payload, sizeof(payload), &offset, 0u);
    write_u32(payload, sizeof(payload), &offset, session->output_length);
    if (queue_frame(session, HCTP_MSG_OUTPUT_BEGIN, payload, offset) != HCTP_STATUS_OK) return HCTP_STATUS_TRUNCATED_FRAME;

    while (cursor < session->output_length)
    {
        const uint32_t chunk_length = (uint32_t)(((session->output_length - cursor) > 32u) ? 32u : (session->output_length - cursor));
        offset = 0u;
        write_u32(payload, sizeof(payload), &offset, (uint32_t)cursor);
        write_u32(payload, sizeof(payload), &offset, chunk_length);
        memcpy(&payload[offset], &session->output_buffer[cursor], chunk_length);
        offset += chunk_length;
        if (queue_frame(session, HCTP_MSG_OUTPUT_CHUNK, payload, offset) != HCTP_STATUS_OK) return HCTP_STATUS_TRUNCATED_FRAME;
        cursor += chunk_length;
    }

    for (cursor = 0u; cursor < session->output_length; ++cursor)
    {
        checksum += session->output_buffer[cursor];
    }
    offset = 0u;
    write_u32(payload, sizeof(payload), &offset, session->output_length);
    write_u32(payload, sizeof(payload), &offset, checksum);
    return queue_frame(session, HCTP_MSG_OUTPUT_END, payload, offset);
}

#ifdef HELIA_HARDWARE_BUILD
static void enable_dwt(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0u;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static uint32_t dwt_cycles(void)
{
    return DWT->CYCCNT;
}

static void pmu_disable_all(void)
{
    ARM_PMU_CNTR_Disable(0xFFFFFFFFu);
}

static void pmu_prepare_group(const char *group)
{
    ARM_PMU_Disable();
    ARM_PMU_CYCCNT_Reset();
    ARM_PMU_EVCNTR_ALL_Reset();
    if (strcmp(group, "cpu") == 0)
    {
        ARM_PMU_Set_EVTYPER(0, ARM_PMU_INST_RETIRED);
    }
    else if (strcmp(group, "memory") == 0)
    {
        ARM_PMU_Set_EVTYPER(0, ARM_PMU_MEM_ACCESS);
    }
    else if (strcmp(group, "mve") == 0)
    {
        ARM_PMU_Set_EVTYPER(0, ARM_PMU_MVE_INST_RETIRED);
    }
    ARM_PMU_Enable();
}

static void pmu_start_group(void)
{
    pmu_disable_all();
    ARM_PMU_CYCCNT_Reset();
    ARM_PMU_EVCNTR_ALL_Reset();
    ARM_PMU_CNTR_Enable(1u << 0);
}

static void pmu_stop_group(void)
{
    pmu_disable_all();
}
#else
static void enable_dwt(void) {}
static uint32_t dwt_cycles(void) { return 0u; }
static void pmu_prepare_group(const char *group) { (void)group; }
static void pmu_start_group(void) {}
static void pmu_stop_group(void) {}
#endif

static hctp_status_t queue_sample_result(hct_server_session_t *session,
                                         uint16_t sample_index,
                                         uint32_t iterations,
                                         uint64_t cycles,
                                         const char *group)
{
    uint8_t payload[256];
    size_t offset = 0u;
    char pass_name[HCT_SERVER_MAX_GROUP_NAME + 4u];
    uint8_t counter_count = 0u;

    snprintf(pass_name, sizeof(pass_name), "%s_0", group);
    write_u16(payload, sizeof(payload), &offset, sample_index);
    write_u32(payload, sizeof(payload), &offset, iterations);
    write_u64(payload, sizeof(payload), &offset, cycles);
    write_text(payload, sizeof(payload), &offset, pass_name);

    if (strcmp(group, "cpu") == 0)
    {
        counter_count = 2u;
        write_u8(payload, sizeof(payload), &offset, counter_count);
        write_text(payload, sizeof(payload), &offset, "ARM_PMU_CPU_CYCLES");
        write_u16(payload, sizeof(payload), &offset, 0x11u);
        write_u64(payload, sizeof(payload), &offset, cycles);
        write_u8(payload, sizeof(payload), &offset, 0u);
        write_u8(payload, sizeof(payload), &offset, 1u);
        write_text(payload, sizeof(payload), &offset, "ARM_PMU_INST_RETIRED");
        write_u16(payload, sizeof(payload), &offset, 0x08u);
#ifdef HELIA_HARDWARE_BUILD
        write_u64(payload, sizeof(payload), &offset, (uint64_t)ARM_PMU_Get_EVCNTR(0));
#else
        write_u64(payload, sizeof(payload), &offset, 0u);
#endif
        write_u8(payload, sizeof(payload), &offset, 0u);
        write_u8(payload, sizeof(payload), &offset, 1u);
    }
    else if (strcmp(group, "memory") == 0)
    {
        counter_count = 1u;
        write_u8(payload, sizeof(payload), &offset, counter_count);
        write_text(payload, sizeof(payload), &offset, "ARM_PMU_MEM_ACCESS");
        write_u16(payload, sizeof(payload), &offset, 0x13u);
#ifdef HELIA_HARDWARE_BUILD
        write_u64(payload, sizeof(payload), &offset, (uint64_t)ARM_PMU_Get_EVCNTR(0));
        write_u8(payload, sizeof(payload), &offset, 0u);
        write_u8(payload, sizeof(payload), &offset, 1u);
#else
        write_u64(payload, sizeof(payload), &offset, 0u);
        write_u8(payload, sizeof(payload), &offset, 0u);
        write_u8(payload, sizeof(payload), &offset, 0u);
#endif
    }
    else if (strcmp(group, "mve") == 0)
    {
        counter_count = 1u;
        write_u8(payload, sizeof(payload), &offset, counter_count);
        write_text(payload, sizeof(payload), &offset, "ARM_PMU_MVE_INST_RETIRED");
        write_u16(payload, sizeof(payload), &offset, 0x0200u);
#ifdef HELIA_HARDWARE_BUILD
        write_u64(payload, sizeof(payload), &offset, (uint64_t)ARM_PMU_Get_EVCNTR(0));
        write_u8(payload, sizeof(payload), &offset, 0u);
        write_u8(payload, sizeof(payload), &offset, 1u);
#else
        write_u64(payload, sizeof(payload), &offset, 0u);
        write_u8(payload, sizeof(payload), &offset, 0u);
        write_u8(payload, sizeof(payload), &offset, 0u);
#endif
    }
    else
    {
        write_u8(payload, sizeof(payload), &offset, 0u);
    }

    return queue_frame(session, HCTP_MSG_SAMPLE_RESULT, payload, offset);
}

static hctp_status_t queue_case_complete(hct_server_session_t *session)
{
    uint8_t payload[128];
    size_t offset = 0u;
    write_text(payload, sizeof(payload), &offset, session->current_case_id);
    write_u8(payload, sizeof(payload), &offset, 1u);
    write_u8(payload, sizeof(payload), &offset, 1u);
    write_u32(payload, sizeof(payload), &offset, session->case_arena_used_bytes);
    return queue_frame(session, HCTP_MSG_CASE_COMPLETE, payload, offset);
}

static hctp_status_t queue_session_complete(hct_server_session_t *session)
{
    uint8_t payload[2];
    size_t offset = 0u;
    write_u16(payload, sizeof(payload), &offset, session->planned_case_count);
    return queue_frame(session, HCTP_MSG_SESSION_COMPLETE, payload, offset);
}

static void reset_case_buffers(hct_server_session_t *session)
{
    memset(session->blobs, 0, sizeof(session->blobs));
    session->blob_count = 0u;
    session->current_blob_index = 0u;
    session->scratch_bytes = 0u;
    session->scratch_offset = 0u;
    session->case_arena_used_bytes = 0u;
    session->output_length = 0u;
    memset(session->case_arena, 0, sizeof(session->case_arena));
}

static hctp_status_t parse_scalar(hct_server_session_t *session, const char *name, int32_t value)
{
    if (strcmp(name, "stride_h") == 0) session->stride_h = value;
    else if (strcmp(name, "stride_w") == 0) session->stride_w = value;
    else if (strcmp(name, "padding") == 0) session->padding = value;
    else if (strcmp(name, "pad_h") == 0) session->pad_h = value;
    else if (strcmp(name, "pad_w") == 0) session->pad_w = value;
    else if (strcmp(name, "output_h") == 0) session->output_h = value;
    else if (strcmp(name, "output_w") == 0) session->output_w = value;
    else if (strcmp(name, "output_c") == 0) session->output_c = value;
    else if (strcmp(name, "dilation_h") == 0) session->dilation_h = value;
    else if (strcmp(name, "dilation_w") == 0) session->dilation_w = value;
    else if (strcmp(name, "input_offset") == 0) session->input_offset = value;
    else if (strcmp(name, "output_offset") == 0) session->output_offset = value;
    else if (strcmp(name, "activation_min") == 0) session->activation_min = value;
    else if (strcmp(name, "activation_max") == 0) session->activation_max = value;
    else if (strcmp(name, "input1_offset") == 0) session->input1_offset = value;
    else if (strcmp(name, "input1_mult") == 0) session->input1_mult = value;
    else if (strcmp(name, "input1_shift") == 0) session->input1_shift = value;
    else if (strcmp(name, "input2_offset") == 0) session->input2_offset = value;
    else if (strcmp(name, "input2_mult") == 0) session->input2_mult = value;
    else if (strcmp(name, "input2_shift") == 0) session->input2_shift = value;
    else if (strcmp(name, "left_shift") == 0) session->left_shift = value;
    else if (strcmp(name, "out_mult") == 0) session->out_mult = value;
    else if (strcmp(name, "out_shift") == 0) session->out_shift = value;
    else if (strcmp(name, "ch_mult") == 0) session->ch_mult = value;
    else return HCTP_STATUS_OK;
    return HCTP_STATUS_OK;
}

static hctp_status_t allocate_blob(hct_server_session_t *session, hct_server_blob_t *blob)
{
    const uint32_t aligned = align_up(session->case_arena_used_bytes, blob->alignment);
    /* Use 64-bit arithmetic for the bounds check: aligned/byte_length are both
     * wire-controlled uint32_t values, so aligned + blob->byte_length can wrap
     * around in 32-bit arithmetic and bypass these checks entirely. */
    const uint64_t end = (uint64_t)aligned + (uint64_t)blob->byte_length;
    if (end > (uint64_t)session->runtime_arena_capacity || end > (uint64_t)sizeof(session->case_arena))
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    blob->arena_offset = aligned;
    blob->bytes_received = 0u;
    session->case_arena_used_bytes = (uint32_t)end;
    return HCTP_STATUS_OK;
}

static arm_cmsis_nn_status run_abs_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_abs_s8_request_t request;
    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    session->output_length = input->byte_length;
    request.input = (const int8_t *)blob_ptr(session, input);
    request.input_offset = 0;
    request.output = (int8_t *)session->output_buffer;
    request.output_offset = 0;
    request.output_multiplier = 0x40000000;
    request.output_shift = 1;
    request.activation_min = -128;
    request.activation_max = 127;
    request.block_size = (int32_t)input->byte_length;
    request.needs_rescale = 0u;
    return hct_dispatch_abs_s8(&request);
}

/* >>> BEGIN GENERATED PERF-STREAM ADAPTERS -- see helia_core_tester/perf_stream/adapter_specs.py and scripts/generate_perf_stream_adapters.py. DO NOT EDIT THIS BLOCK BY HAND: rerun the generator after editing adapter_specs.py. >>> */
#ifndef HCT_HOST_ABS_ONLY

static hctp_status_t compute_convolve_output_dims(const hct_server_session_t *session,
                                                  const hct_server_blob_t *input,
                                                  const hct_server_blob_t *weights,
                                                  cmsis_nn_dims *output_dims,
                                                  uint32_t *output_bytes)
{
    /* Use the ground-truth output dims sent explicitly by the host (see parse_scalar())
     * rather than re-deriving them from stride/padding here: a SAME/VALID formula-based
     * recomputation can silently diverge from the real generator's actual output size for
     * real generated test cases (e.g. asymmetric padding, rounding conventions), producing
     * a garbage output_length that corrupts the correctness-output stream. */
    (void)input;
    (void)weights;
    output_dims->n = 1;
    output_dims->h = session->output_h;
    output_dims->w = session->output_w;
    output_dims->c = session->output_c;

    if (output_dims->h <= 0 || output_dims->w <= 0 || output_dims->c <= 0)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }

    *output_bytes = (uint32_t)(output_dims->n * output_dims->h * output_dims->w * output_dims->c);
    if (*output_bytes > sizeof(session->output_buffer))
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    return HCTP_STATUS_OK;
}

static arm_cmsis_nn_status run_convolve_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *weights = find_blob_by_role(session, HCT_BLOB_ROLE_WEIGHTS);
    hct_server_blob_t *bias = find_blob_by_role(session, HCT_BLOB_ROLE_BIAS);
    hct_server_blob_t *multiplier = find_blob_by_role(session, HCT_BLOB_ROLE_MULTIPLIER);
    hct_server_blob_t *shift = find_blob_by_role(session, HCT_BLOB_ROLE_SHIFT);
    cmsis_nn_context ctx;
    cmsis_nn_context weight_sum_ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    hct_convolve_s8_request_t request;
    int32_t required_scratch;
    uint32_t weight_sum_offset;
    uint32_t weight_sum_bytes;

    if (input == NULL || weights == NULL || bias == NULL || multiplier == NULL || shift == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    filter_dims.h = (int32_t)weights->dimensions[0];
    filter_dims.w = (int32_t)weights->dimensions[1];
    filter_dims.c = (int32_t)weights->dimensions[2];
    filter_dims.n = (int32_t)weights->dimensions[3];
    bias_dims.n = 0;
    bias_dims.h = 0;
    bias_dims.w = 0;
    bias_dims.c = (int32_t)bias->dimensions[0];

    if (compute_convolve_output_dims(session, input, weights, &output_dims, &session->output_length) != HCTP_STATUS_OK)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    required_scratch = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    if (required_scratch < 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if ((uint32_t)required_scratch > session->scratch_bytes)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    weight_sum_bytes = (uint32_t)output_dims.c * (uint32_t)sizeof(int32_t);
    weight_sum_offset = align_up(session->scratch_offset + session->scratch_bytes, 16u);
    if (weight_sum_offset + weight_sum_bytes > session->runtime_arena_capacity ||
        weight_sum_offset + weight_sum_bytes > sizeof(session->case_arena))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    ctx.buf = (session->scratch_bytes > 0u) ? &session->case_arena[session->scratch_offset] : NULL;
    ctx.size = session->scratch_bytes;
    weight_sum_ctx.buf = &session->case_arena[weight_sum_offset];
    weight_sum_ctx.size = (int32_t)weight_sum_bytes;
    conv_params.input_offset = session->input_offset;
    conv_params.output_offset = session->output_offset;
    conv_params.stride.w = (session->stride_w == 0) ? 1 : session->stride_w;
    conv_params.stride.h = (session->stride_h == 0) ? 1 : session->stride_h;
    /* Use the ground-truth "before" padding sent explicitly by the host (see
     * parse_scalar()) instead of re-deriving it here via a SAME-formula + symmetric
     * `/2` split, which assumes an even padding split that doesn't always hold and can
     * silently diverge from the real generator's exact padding convention. */
    conv_params.padding.w = session->pad_w;
    conv_params.padding.h = session->pad_h;
    conv_params.dilation.w = (session->dilation_w == 0) ? 1 : session->dilation_w;
    conv_params.dilation.h = (session->dilation_h == 0) ? 1 : session->dilation_h;
    conv_params.activation.min = session->activation_min;
    conv_params.activation.max = session->activation_max;
    quant_params.multiplier = (int32_t *)blob_ptr(session, multiplier);
    quant_params.shift = (int32_t *)blob_ptr(session, shift);
    if (arm_convolve_weight_sum((int32_t *)weight_sum_ctx.buf,
                                (const int8_t *)blob_ptr(session, weights),
                                &input_dims,
                                &filter_dims,
                                &output_dims,
                                session->input_offset,
                                (const int32_t *)blob_ptr(session, bias)) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (weight_sum_offset + weight_sum_bytes > session->case_arena_used_bytes)
    {
        session->case_arena_used_bytes = weight_sum_offset + weight_sum_bytes;
    }

    request.ctx = &ctx;
    request.weight_sum_ctx = &weight_sum_ctx;
    request.conv_params = &conv_params;
    request.quant_params = &quant_params;
    request.input_dims = &input_dims;
    request.input_data = (const int8_t *)blob_ptr(session, input);
    request.filter_dims = &filter_dims;
    request.filter_data = (const int8_t *)blob_ptr(session, weights);
    request.bias_dims = &bias_dims;
    request.bias_data = (const int32_t *)blob_ptr(session, bias);
    request.upscale_dims = NULL;
    request.output_dims = &output_dims;
    request.output_data = (int8_t *)session->output_buffer;
    return hct_dispatch_convolve_s8(&request);
}

/* arm_depthwise_conv_s8 is the low-level (non-wrapper) depthwise conv kernel: unlike
 * arm_convolve_s8 above, its `ctx` and `bias_dims` args are both unused internally (see
 * Source/ConvolutionFunctions/arm_depthwise_conv_s8.c's `(void)ctx;`/`(void)bias_dims;`),
 * so no scratch buffer or weight-sum precomputation is required here. filter_dims stay in
 * the generator's native (N=1, H, W, C_OUT) order -- no HWCN reordering like Convolve's
 * filter_dims (depthwise's cmsis_nn_dw_conv_params filter convention is already NHWC). */
static arm_cmsis_nn_status run_depthwise_conv_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *weights = find_blob_by_role(session, HCT_BLOB_ROLE_WEIGHTS);
    hct_server_blob_t *bias = find_blob_by_role(session, HCT_BLOB_ROLE_BIAS);
    hct_server_blob_t *multiplier = find_blob_by_role(session, HCT_BLOB_ROLE_MULTIPLIER);
    hct_server_blob_t *shift = find_blob_by_role(session, HCT_BLOB_ROLE_SHIFT);
    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    if (input == NULL || weights == NULL || bias == NULL || multiplier == NULL || shift == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    filter_dims.n = (int32_t)weights->dimensions[0];
    filter_dims.h = (int32_t)weights->dimensions[1];
    filter_dims.w = (int32_t)weights->dimensions[2];
    filter_dims.c = (int32_t)weights->dimensions[3];
    bias_dims.n = 0;
    bias_dims.h = 0;
    bias_dims.w = 0;
    bias_dims.c = (int32_t)bias->dimensions[0];

    output_dims.n = 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    session->output_length = (uint32_t)(output_dims.n * output_dims.h * output_dims.w * output_dims.c);
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    dw_conv_params.input_offset = session->input_offset;
    dw_conv_params.output_offset = session->output_offset;
    dw_conv_params.ch_mult = session->ch_mult;
    dw_conv_params.stride.w = (session->stride_w == 0) ? 1 : session->stride_w;
    dw_conv_params.stride.h = (session->stride_h == 0) ? 1 : session->stride_h;
    dw_conv_params.padding.w = session->pad_w;
    dw_conv_params.padding.h = session->pad_h;
    dw_conv_params.dilation.w = (session->dilation_w == 0) ? 1 : session->dilation_w;
    dw_conv_params.dilation.h = (session->dilation_h == 0) ? 1 : session->dilation_h;
    dw_conv_params.activation.min = session->activation_min;
    dw_conv_params.activation.max = session->activation_max;
    quant_params.multiplier = (int32_t *)blob_ptr(session, multiplier);
    quant_params.shift = (int32_t *)blob_ptr(session, shift);

    return arm_depthwise_conv_s8(&ctx,
                                  &dw_conv_params,
                                  &quant_params,
                                  &input_dims,
                                  (const int8_t *)blob_ptr(session, input),
                                  &filter_dims,
                                  (const int8_t *)blob_ptr(session, weights),
                                  &bias_dims,
                                  (const int32_t *)blob_ptr(session, bias),
                                  &output_dims,
                                  (int8_t *)session->output_buffer);
}

/* arm_add_s8/arm_sub_s8 share an identical signature/argument order; both are dispatched
 * from this one wrapper, branching only on session->expected_kernel_id (see
 * assets/kernel_registry.yaml for the kernel_id <-> operator mapping). Ground-truth output
 * dims (session->output_h/w/c) are sent explicitly by the host, same rationale as
 * compute_convolve_output_dims(): broadcasting output shape shouldn't be re-derived here. */
static arm_cmsis_nn_status run_elementwise_binary_once(hct_server_session_t *session)
{
    hct_server_blob_t *input1 = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *input2 = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_1);
    cmsis_nn_dims input1_dims;
    cmsis_nn_dims input2_dims;
    cmsis_nn_dims output_dims;

    if (input1 == NULL || input2 == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input1_dims.n = (int32_t)input1->dimensions[0];
    input1_dims.h = (int32_t)input1->dimensions[1];
    input1_dims.w = (int32_t)input1->dimensions[2];
    input1_dims.c = (int32_t)input1->dimensions[3];
    input2_dims.n = (int32_t)input2->dimensions[0];
    input2_dims.h = (int32_t)input2->dimensions[1];
    input2_dims.w = (int32_t)input2->dimensions[2];
    input2_dims.c = (int32_t)input2->dimensions[3];

    output_dims.n = 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    session->output_length = (uint32_t)(output_dims.n * output_dims.h * output_dims.w * output_dims.c);
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int8_t *input1_data = (const int8_t *)blob_ptr(session, input1);
    const int8_t *input2_data = (const int8_t *)blob_ptr(session, input2);
    int8_t *output_data = (int8_t *)session->output_buffer;

    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_ADD_S8:
            return arm_add_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                              session->input1_offset, session->input1_mult, session->input1_shift,
                              session->input2_offset, session->input2_mult, session->input2_shift,
                              session->left_shift,
                              output_data, &output_dims,
                              session->output_offset, session->out_mult, session->out_shift,
                              session->activation_min, session->activation_max);
        case HCT_KERNEL_ID_SUB_S8:
            return arm_sub_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                              session->input1_offset, session->input1_mult, session->input1_shift,
                              session->input2_offset, session->input2_mult, session->input2_shift,
                              session->left_shift,
                              output_data, &output_dims,
                              session->output_offset, session->out_mult, session->out_shift,
                              session->activation_min, session->activation_max);
        case HCT_KERNEL_ID_MUL_S8:
            /* arm_mul_s8 has no per-input mult/shift or left_shift -- it reuses only the
             * input1_offset/input2_offset scalar fields (shared with Add/Sub above). */
            return arm_mul_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                              session->input1_offset, session->input2_offset,
                              output_data, &output_dims,
                              session->output_offset, session->out_mult, session->out_shift,
                              session->activation_min, session->activation_max);
        case HCT_KERNEL_ID_MAXIMUM_S8:
        case HCT_KERNEL_ID_MINIMUM_S8:
        {
            /* Maximum/Minimum have no quant scalars at all -- just a scratch context, always
             * {NULL, 0} per the generated tests (no buffer-sizing helper exists for these ops). */
            cmsis_nn_context ctx = {NULL, 0};
            if (session->expected_kernel_id == HCT_KERNEL_ID_MAXIMUM_S8)
            {
                return arm_maximum_s8(&ctx, input1_data, &input1_dims, input2_data, &input2_dims,
                                      output_data, &output_dims);
            }
            return arm_minimum_s8(&ctx, input1_data, &input1_dims, input2_data, &input2_dims,
                                  output_data, &output_dims);
        }
        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}
#endif
/* <<< END GENERATED PERF-STREAM ADAPTERS <<< */

static arm_cmsis_nn_status run_kernel_once(hct_server_session_t *session)
{
    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_ABS_S8:
            return run_abs_once(session);
        case HCT_KERNEL_ID_CONVOLVE_S8:
#ifndef HCT_HOST_ABS_ONLY
            return run_convolve_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_DEPTHWISE_CONV_S8:
#ifndef HCT_HOST_ABS_ONLY
            return run_depthwise_conv_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_ADD_S8:
        case HCT_KERNEL_ID_SUB_S8:
        case HCT_KERNEL_ID_MUL_S8:
        case HCT_KERNEL_ID_MAXIMUM_S8:
        case HCT_KERNEL_ID_MINIMUM_S8:
#ifndef HCT_HOST_ABS_ONLY
            return run_elementwise_binary_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}

static uint32_t resolve_iterations(hct_server_session_t *session)
{
    uint32_t iterations = session->planned_iterations;
    uint32_t cycles = 0u;
    if (iterations != 0u)
    {
        return iterations;
    }
    iterations = 1u;
    while (iterations < session->max_iterations)
    {
        uint32_t index;
        enable_dwt();
        const uint32_t start = dwt_cycles();
        for (index = 0u; index < iterations; ++index)
        {
            if (run_kernel_once(session) != ARM_CMSIS_NN_SUCCESS)
            {
                return 1u;
            }
        }
        cycles = dwt_cycles() - start;
        if (cycles >= session->min_cycles)
        {
            break;
        }
        if (iterations > (session->max_iterations / 2u))
        {
            iterations = session->max_iterations;
            break;
        }
        iterations *= 2u;
    }
    return iterations;
}

static hctp_status_t finish_case(hct_server_session_t *session)
{
    if (queue_case_complete(session) != HCTP_STATUS_OK)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    reset_case_buffers(session);
    session->current_case_index += 1u;
    if (session->current_case_index < session->planned_case_count)
    {
        strcpy(session->current_case_id, session->planned_case_ids[session->current_case_index]);
        session->expected_kernel_id = session->planned_kernel_ids[session->current_case_index];
        session->state = HCT_SERVER_STATE_WAIT_CASE_META;
        return queue_request_case(session);
    }
    session->state = HCT_SERVER_STATE_COMPLETE;
    return queue_session_complete(session);
}

static hctp_status_t handle_load_plan(hct_server_session_t *session, const uint8_t *payload, size_t payload_length)
{
    size_t offset = 0u;
    uint16_t case_index;
    char scratch[HCT_SERVER_MAX_CASE_ID];

    if (payload_length < 20u)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }

    session->planned_case_count = read_u16(payload, &offset);
    (void)read_u8(payload, &offset);
    session->planned_warmups = read_u16(payload, &offset);
    session->planned_samples = read_u16(payload, &offset);
    session->planned_iterations = read_u32(payload, &offset);
    session->min_cycles = read_u32(payload, &offset);
    session->max_iterations = read_u32(payload, &offset);
    session->requested_group_count = read_u8(payload, &offset);

    if (session->planned_case_count == 0u || session->planned_case_count > HCT_SERVER_MAX_CASES)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    if (session->requested_group_count > HCT_SERVER_MAX_GROUPS)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }

    for (case_index = 0u; case_index < session->requested_group_count; ++case_index)
    {
        if (read_text(payload, payload_length, &offset, session->requested_groups[case_index], sizeof(session->requested_groups[case_index])) != HCTP_STATUS_OK)
        {
            return HCTP_STATUS_TRUNCATED_FRAME;
        }
    }
    if (session->requested_group_count == 0u)
    {
        strcpy(session->requested_groups[0], "cpu");
        session->requested_group_count = 1u;
    }

    for (case_index = 0u; case_index < session->planned_case_count; ++case_index)
    {
        if (read_text(payload, payload_length, &offset, session->planned_case_ids[case_index], sizeof(session->planned_case_ids[case_index])) != HCTP_STATUS_OK)
        {
            return HCTP_STATUS_TRUNCATED_FRAME;
        }
        session->planned_kernel_ids[case_index] = read_u32(payload, &offset);
    }

    strcpy(session->current_case_id, session->planned_case_ids[0]);
    session->expected_kernel_id = session->planned_kernel_ids[0];
    (void)scratch;
    session->state = HCT_SERVER_STATE_WAIT_CASE_META;
    return queue_request_case(session);
}

static hctp_status_t handle_case_meta(hct_server_session_t *session, const uint8_t *payload, size_t payload_length)
{
    size_t offset = 0u;
    uint8_t scalar_count;
    uint16_t blob_index;
    char scratch[64];
    hctp_status_t status;

    reset_case_buffers(session);
    session->stride_h = 1;
    session->stride_w = 1;
    session->padding = HCT_PADDING_VALID;
    session->dilation_h = 1;
    session->dilation_w = 1;
    session->activation_min = -128;
    session->activation_max = 127;
    session->input_offset = 0;
    session->output_offset = 0;

    status = read_text(payload, payload_length, &offset, scratch, sizeof(scratch));
    if (status != HCTP_STATUS_OK) return status;
    if (strcmp(scratch, session->current_case_id) != 0) return HCTP_STATUS_INVALID_ARGUMENT;
    if (!has_capacity(payload_length, offset, 4u)) return HCTP_STATUS_TRUNCATED_FRAME;
    if (read_u32(payload, &offset) != session->expected_kernel_id) return HCTP_STATUS_INVALID_ARGUMENT;

    if (!has_capacity(payload_length, offset, 16u)) return HCTP_STATUS_TRUNCATED_FRAME;
    (void)read_u16(payload, &offset);
    session->comparison_mode = read_u8(payload, &offset);
    session->tolerance = read_i32(payload, &offset);
    session->atol_q16 = read_u32(payload, &offset);
    session->rtol_q16 = read_u32(payload, &offset);
    scalar_count = read_u8(payload, &offset);
    while (scalar_count-- > 0u)
    {
        int32_t value;
        status = read_text(payload, payload_length, &offset, scratch, sizeof(scratch));
        if (status != HCTP_STATUS_OK) return status;
        if (!has_capacity(payload_length, offset, 4u)) return HCTP_STATUS_TRUNCATED_FRAME;
        value = read_i32(payload, &offset);
        status = parse_scalar(session, scratch, value);
        if (status != HCTP_STATUS_OK) return status;
    }

    if (!has_capacity(payload_length, offset, 2u)) return HCTP_STATUS_TRUNCATED_FRAME;
    session->blob_count = read_u16(payload, &offset);
    if (session->blob_count == 0u || session->blob_count > HCT_SERVER_MAX_BLOBS)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }

    for (blob_index = 0u; blob_index < session->blob_count; ++blob_index)
    {
        hct_server_blob_t *blob = &session->blobs[blob_index];
        uint8_t dim_index;
        if (!has_capacity(payload_length, offset, 4u)) return HCTP_STATUS_TRUNCATED_FRAME;
        blob->blob_id = read_u32(payload, &offset);
        status = read_text(payload, payload_length, &offset, scratch, sizeof(scratch));
        if (status != HCTP_STATUS_OK) return status;
        blob->role = role_from_name(scratch);
        status = read_text(payload, payload_length, &offset, scratch, sizeof(scratch));
        if (status != HCTP_STATUS_OK) return status;
        blob->dtype = dtype_from_name(scratch);
        if (!has_capacity(payload_length, offset, 1u + (6u * 4u) + 4u + 4u + 4u + 1u)) return HCTP_STATUS_TRUNCATED_FRAME;
        blob->rank = read_u8(payload, &offset);
        for (dim_index = 0u; dim_index < 6u; ++dim_index)
        {
            blob->dimensions[dim_index] = read_u32(payload, &offset);
        }
        blob->byte_length = read_u32(payload, &offset);
        blob->alignment = read_u32(payload, &offset);
        blob->crc32 = read_u32(payload, &offset);
        blob->mutable_data = read_u8(payload, &offset);
        if (blob->role == HCT_BLOB_ROLE_UNKNOWN || blob->dtype == HCT_DTYPE_UNKNOWN || blob->alignment == 0u)
        {
            return HCTP_STATUS_INVALID_ARGUMENT;
        }
        status = allocate_blob(session, blob);
        if (status != HCTP_STATUS_OK) return status;
    }

    if (!has_capacity(payload_length, offset, 4u)) return HCTP_STATUS_TRUNCATED_FRAME;
    session->scratch_bytes = read_u32(payload, &offset);
    if (session->scratch_bytes > 0u)
    {
        session->scratch_offset = align_up(session->case_arena_used_bytes, 16u);
        if (session->scratch_offset + session->scratch_bytes > session->runtime_arena_capacity ||
            session->scratch_offset + session->scratch_bytes > sizeof(session->case_arena))
        {
            return HCTP_STATUS_INVALID_ARGUMENT;
        }
        session->case_arena_used_bytes = session->scratch_offset + session->scratch_bytes;
    }

    session->current_blob_index = 0u;
    session->state = HCT_SERVER_STATE_WAIT_BLOB_CHUNK;
    return queue_request_blob(session);
}

static hctp_status_t handle_blob_chunk(hct_server_session_t *session, const uint8_t *payload, size_t payload_length)
{
    size_t offset = 0u;
    uint32_t blob_id;
    uint32_t chunk_offset;
    uint32_t chunk_length;
    hct_server_blob_t *blob;

    if (!has_capacity(payload_length, offset, 12u))
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    blob_id = read_u32(payload, &offset);
    chunk_offset = read_u32(payload, &offset);
    chunk_length = read_u32(payload, &offset);
    blob = &session->blobs[session->current_blob_index];

    if (blob_id != blob->blob_id) return HCTP_STATUS_INVALID_ARGUMENT;
    if (chunk_offset != blob->bytes_received) return HCTP_STATUS_INVALID_ARGUMENT;
    if (offset + chunk_length > payload_length) return HCTP_STATUS_TRUNCATED_FRAME;
    if (chunk_offset + chunk_length > blob->byte_length) return HCTP_STATUS_INVALID_ARGUMENT;
    if ((uint64_t)chunk_offset + (uint64_t)chunk_length > (uint64_t)sizeof(session->case_arena)) return HCTP_STATUS_INVALID_ARGUMENT;
    if ((blob->alignment > 1u) && ((chunk_offset % blob->alignment) != 0u)) return HCTP_STATUS_INVALID_ARGUMENT;

    memcpy(blob_ptr(session, blob) + chunk_offset, payload + offset, chunk_length);
    blob->bytes_received += chunk_length;

    if (blob->bytes_received < blob->byte_length)
    {
        return queue_request_blob(session);
    }
    if (hctp_crc32(blob_ptr(session, blob), blob->byte_length) != blob->crc32)
    {
        return HCTP_STATUS_PAYLOAD_CRC_MISMATCH;
    }

    session->current_blob_index += 1u;
    if (session->current_blob_index < session->blob_count)
    {
        return queue_request_blob(session);
    }

    session->state = HCT_SERVER_STATE_WAIT_RUN_CORRECTNESS;
    return queue_case_ready(session);
}

static hctp_status_t handle_run_correctness(hct_server_session_t *session)
{
    if (run_kernel_once(session) != ARM_CMSIS_NN_SUCCESS)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    session->state = HCT_SERVER_STATE_WAIT_CORRECTNESS_ACK;
    return queue_correctness_output(session);
}

static hctp_status_t handle_run_performance(hct_server_session_t *session)
{
    uint32_t group_index;
    uint32_t sample_index;
    uint32_t warmup;
    const uint32_t iterations = resolve_iterations(session);

    for (group_index = 0u; group_index < session->requested_group_count; ++group_index)
    {
        const char *group = session->requested_groups[group_index];
        enable_dwt();
        pmu_prepare_group(group);
        for (warmup = 0u; warmup < session->planned_warmups; ++warmup)
        {
            if (run_kernel_once(session) != ARM_CMSIS_NN_SUCCESS)
            {
                return HCTP_STATUS_INVALID_ARGUMENT;
            }
        }
        for (sample_index = 0u; sample_index < session->planned_samples; ++sample_index)
        {
            uint32_t iter;
            uint32_t start;
            uint32_t end;
            pmu_start_group();
            start = dwt_cycles();
            for (iter = 0u; iter < iterations; ++iter)
            {
                if (run_kernel_once(session) != ARM_CMSIS_NN_SUCCESS)
                {
                    pmu_stop_group();
                    return HCTP_STATUS_INVALID_ARGUMENT;
                }
            }
            end = dwt_cycles();
            pmu_stop_group();
            if (queue_sample_result(session, (uint16_t)sample_index, iterations, (uint64_t)(end - start), group) != HCTP_STATUS_OK)
            {
                return HCTP_STATUS_TRUNCATED_FRAME;
            }
        }
    }

    return finish_case(session);
}

void hct_server_session_init(hct_server_session_t *session,
                             uint32_t session_id,
                             uint32_t max_frame_payload,
                             uint32_t runtime_arena_capacity)
{
    size_t frame_length = 0u;
    memset(session, 0, sizeof(*session));
    session->session_id = session_id;
    session->max_frame_payload = max_frame_payload;
    session->runtime_arena_capacity = (runtime_arena_capacity > HCT_SERVER_MAX_ARENA_BYTES)
        ? HCT_SERVER_MAX_ARENA_BYTES
        : runtime_arena_capacity;
    session->state = HCT_SERVER_STATE_WAIT_HELLO_ACK;
    hct_build_hello_frame(session_id,
                          session->next_outgoing_sequence++,
                          max_frame_payload,
                          session->runtime_arena_capacity,
                          session->outbox,
                          sizeof(session->outbox),
                          &frame_length);
    session->outbox_length = frame_length;
}

hctp_status_t hct_server_session_accept_frame(hct_server_session_t *session,
                                              const uint8_t *frame_bytes,
                                              size_t frame_length)
{
    hctp_frame_view_t frame;
    uint8_t frame_buffer[1024u];
    size_t catalog_frame_length = 0u;
    hctp_status_t status;
    const hctp_status_t decode_status = hctp_decode_frame(frame_bytes, frame_length, HCTP_DEFAULT_MAX_PAYLOAD, &frame);
    if (decode_status != HCTP_STATUS_OK)
    {
        session->state = HCT_SERVER_STATE_ERROR;
        /* Frame didn't even decode (bad magic/version/CRC/length) -- message_type
         * isn't trustworthy, so report it against a sentinel of 0 rather than
         * whatever garbage frame.header.message_type might hold. */
        queue_error_frame(session, 0u, decode_status);
        return decode_status;
    }

    switch (frame.header.message_type)
    {
        case HCTP_MSG_HELLO_ACK:
            if (session->state != HCT_SERVER_STATE_WAIT_HELLO_ACK)
            {
                queue_error_frame(session, frame.header.message_type, HCTP_STATUS_INVALID_ARGUMENT);
                return HCTP_STATUS_INVALID_ARGUMENT;
            }
            session->state = HCT_SERVER_STATE_WAIT_PLAN;
            if (hct_build_catalog_frame(session->session_id,
                                        session->next_outgoing_sequence++,
                                        frame_buffer,
                                        sizeof(frame_buffer),
                                        &catalog_frame_length) != HCTP_STATUS_OK)
            {
                session->state = HCT_SERVER_STATE_ERROR;
                queue_error_frame(session, frame.header.message_type, HCTP_STATUS_TRUNCATED_FRAME);
                return HCTP_STATUS_TRUNCATED_FRAME;
            }
            return append_frame(session, frame_buffer, catalog_frame_length);
        case HCTP_MSG_LOAD_PLAN:
            if (session->state != HCT_SERVER_STATE_WAIT_PLAN)
            {
                queue_error_frame(session, frame.header.message_type, HCTP_STATUS_INVALID_ARGUMENT);
                return HCTP_STATUS_INVALID_ARGUMENT;
            }
            status = handle_load_plan(session, frame.payload, frame.header.payload_length);
            if (status != HCTP_STATUS_OK) queue_error_frame(session, frame.header.message_type, status);
            return status;
        case HCTP_MSG_CASE_META:
            if (session->state != HCT_SERVER_STATE_WAIT_CASE_META)
            {
                queue_error_frame(session, frame.header.message_type, HCTP_STATUS_INVALID_ARGUMENT);
                return HCTP_STATUS_INVALID_ARGUMENT;
            }
            status = handle_case_meta(session, frame.payload, frame.header.payload_length);
            if (status != HCTP_STATUS_OK) queue_error_frame(session, frame.header.message_type, status);
            return status;
        case HCTP_MSG_BLOB_CHUNK:
            if (session->state != HCT_SERVER_STATE_WAIT_BLOB_CHUNK)
            {
                queue_error_frame(session, frame.header.message_type, HCTP_STATUS_INVALID_ARGUMENT);
                return HCTP_STATUS_INVALID_ARGUMENT;
            }
            status = handle_blob_chunk(session, frame.payload, frame.header.payload_length);
            if (status != HCTP_STATUS_OK) queue_error_frame(session, frame.header.message_type, status);
            return status;
        case HCTP_MSG_RUN_CORRECTNESS:
            if (session->state != HCT_SERVER_STATE_WAIT_RUN_CORRECTNESS)
            {
                queue_error_frame(session, frame.header.message_type, HCTP_STATUS_INVALID_ARGUMENT);
                return HCTP_STATUS_INVALID_ARGUMENT;
            }
            status = handle_run_correctness(session);
            if (status != HCTP_STATUS_OK) queue_error_frame(session, frame.header.message_type, status);
            return status;
        case HCTP_MSG_CORRECTNESS_ACK:
            if (session->state != HCT_SERVER_STATE_WAIT_CORRECTNESS_ACK)
            {
                queue_error_frame(session, frame.header.message_type, HCTP_STATUS_INVALID_ARGUMENT);
                return HCTP_STATUS_INVALID_ARGUMENT;
            }
            session->state = HCT_SERVER_STATE_WAIT_RUN_PERFORMANCE;
            return HCTP_STATUS_OK;
        case HCTP_MSG_RUN_PERFORMANCE:
            if (session->state != HCT_SERVER_STATE_WAIT_RUN_PERFORMANCE)
            {
                queue_error_frame(session, frame.header.message_type, HCTP_STATUS_INVALID_ARGUMENT);
                return HCTP_STATUS_INVALID_ARGUMENT;
            }
            status = handle_run_performance(session);
            if (status != HCTP_STATUS_OK) queue_error_frame(session, frame.header.message_type, status);
            return status;
        default:
            session->state = HCT_SERVER_STATE_ERROR;
            queue_error_frame(session, frame.header.message_type, HCTP_STATUS_INVALID_ARGUMENT);
            return HCTP_STATUS_INVALID_ARGUMENT;
    }
}

size_t hct_server_session_take_outbound(hct_server_session_t *session,
                                        uint8_t *buffer,
                                        size_t capacity)
{
    const size_t count = (session->outbox_length < capacity) ? session->outbox_length : capacity;
    memcpy(buffer, session->outbox, count);
    memmove(session->outbox, session->outbox + count, session->outbox_length - count);
    session->outbox_length -= count;
    return count;
}

size_t hct_server_session_take_next_frame(hct_server_session_t *session,
                                          uint8_t *buffer,
                                          size_t capacity)
{
    hctp_frame_header_t header;
    size_t frame_length;

    if (session->outbox_length < HCTP_HEADER_SIZE)
    {
        return 0u;
    }
    if (hctp_decode_header(session->outbox, HCTP_HEADER_SIZE, HCTP_DEFAULT_MAX_PAYLOAD, &header) != HCTP_STATUS_OK)
    {
        return 0u;
    }
    frame_length = HCTP_HEADER_SIZE + (size_t)header.payload_length;
    if (frame_length > session->outbox_length || frame_length > capacity)
    {
        return 0u;
    }
    memcpy(buffer, session->outbox, frame_length);
    memmove(session->outbox, session->outbox + frame_length, session->outbox_length - frame_length);
    session->outbox_length -= frame_length;
    return frame_length;
}
