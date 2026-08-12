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
#define HCT_DTYPE_S16 3u
#define HCT_DTYPE_S64 4u
#define HCT_DTYPE_F32 5u

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
#define HCT_KERNEL_ID_ADD_S16 9u
#define HCT_KERNEL_ID_SUB_S16 10u
#define HCT_KERNEL_ID_MUL_S16 11u
#define HCT_KERNEL_ID_MAXIMUM_S16 12u
#define HCT_KERNEL_ID_MINIMUM_S16 13u
#define HCT_KERNEL_ID_CONVOLVE_S16 14u
#define HCT_KERNEL_ID_DEPTHWISE_CONV_S16 15u
#define HCT_KERNEL_ID_AVGPOOL_S8 16u
#define HCT_KERNEL_ID_MAXPOOL_S8 17u
#define HCT_KERNEL_ID_AVGPOOL_S16 18u
#define HCT_KERNEL_ID_MAXPOOL_S16 19u
#define HCT_KERNEL_ID_RELU_S8 20u
#define HCT_KERNEL_ID_RELU_S16 21u
#define HCT_KERNEL_ID_RELU6_S8 22u
#define HCT_KERNEL_ID_RELU6_S16 23u
#define HCT_KERNEL_ID_CLAMP_S8 24u
#define HCT_KERNEL_ID_CLAMP_S16 25u
#define HCT_KERNEL_ID_LEAKY_RELU_S8 26u
#define HCT_KERNEL_ID_LEAKY_RELU_S16 27u
#define HCT_KERNEL_ID_LOGISTIC_S16 28u
#define HCT_KERNEL_ID_TANH_S16 29u
#define HCT_KERNEL_ID_HARD_SWISH_COMPAT_S8 30u
#define HCT_KERNEL_ID_HARD_SWISH_PRECISE_S8 31u
#define HCT_KERNEL_ID_HARD_SWISH_PRECISE_S16 32u
#define HCT_KERNEL_ID_PRELU_S8 33u
#define HCT_KERNEL_ID_PRELU_S16 34u
#define HCT_KERNEL_ID_PRELU_SCALAR_S8 35u
#define HCT_KERNEL_ID_PRELU_SCALAR_S16 36u
#define HCT_KERNEL_ID_QUANTIZE_S8 37u
#define HCT_KERNEL_ID_QUANTIZE_S16 38u
#define HCT_KERNEL_ID_DEQUANTIZE_S8 39u
#define HCT_KERNEL_ID_DEQUANTIZE_S16 40u
#define HCT_KERNEL_ID_SOFTMAX_S8 41u
#define HCT_KERNEL_ID_SOFTMAX_S16 42u
#define HCT_KERNEL_ID_SOFTMAX_S8_S16 43u
#define HCT_KERNEL_ID_FULLY_CONNECTED_S8 44u
#define HCT_KERNEL_ID_FULLY_CONNECTED_S16 45u
#define HCT_KERNEL_ID_BATCH_MATMUL_S8 46u
#define HCT_KERNEL_ID_BATCH_MATMUL_S16 47u
#define HCT_KERNEL_ID_ABS_S16 48u
#define HCT_KERNEL_ID_ARGMAX_S8 49u
#define HCT_KERNEL_ID_ARGMAX_S16 50u
#define HCT_KERNEL_ID_ARGMIN_S8 51u
#define HCT_KERNEL_ID_ARGMIN_S16 52u
#define HCT_KERNEL_ID_MEAN_S8 53u
#define HCT_KERNEL_ID_MEAN_S16 54u
#define HCT_KERNEL_ID_REDUCE_MAX_S8 55u
#define HCT_KERNEL_ID_REDUCE_MAX_S16 56u
#define HCT_KERNEL_ID_REDUCE_MIN_S8 57u
#define HCT_KERNEL_ID_REDUCE_MIN_S16 58u
#define HCT_KERNEL_ID_RSQRT_S16_PER_OP 59u
#define HCT_KERNEL_ID_RSQRT_S16_UNIVERSAL 60u
#define HCT_KERNEL_ID_SQRT_S8 61u
#define HCT_KERNEL_ID_SQRT_S16 62u
#define HCT_KERNEL_ID_SQUARED_DIFFERENCE_S8 63u
#define HCT_KERNEL_ID_SQUARED_DIFFERENCE_S16 64u
#define HCT_KERNEL_ID_REQUANTIZE_S8 65u
#define HCT_KERNEL_ID_REQUANTIZE_S16 66u
#define HCT_KERNEL_ID_EQUAL_S8 67u
#define HCT_KERNEL_ID_EQUAL_S16 68u
#define HCT_KERNEL_ID_NOT_EQUAL_S8 69u
#define HCT_KERNEL_ID_NOT_EQUAL_S16 70u
#define HCT_KERNEL_ID_GREATER_S8 71u
#define HCT_KERNEL_ID_GREATER_S16 72u
#define HCT_KERNEL_ID_GREATER_EQUAL_S8 73u
#define HCT_KERNEL_ID_GREATER_EQUAL_S16 74u
#define HCT_KERNEL_ID_LESS_S8 75u
#define HCT_KERNEL_ID_LESS_S16 76u
#define HCT_KERNEL_ID_LESS_EQUAL_S8 77u
#define HCT_KERNEL_ID_LESS_EQUAL_S16 78u

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
    if (strcmp(name, "S16") == 0) return HCT_DTYPE_S16;
    if (strcmp(name, "S64") == 0) return HCT_DTYPE_S64;
    if (strcmp(name, "FP32") == 0) return HCT_DTYPE_F32;
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
    session->output_n = 0;
    session->axis_n = 0;
    session->axis_h = 0;
    session->axis_w = 0;
    session->axis_c = 0;
    session->axis = 0;
    session->needs_rescale = 0;
    memset(session->case_arena, 0, sizeof(session->case_arena));
}

static hctp_status_t parse_scalar(hct_server_session_t *session, const char *name, int32_t value)
{
    if (strcmp(name, "stride_h") == 0) session->stride_h = value;
    else if (strcmp(name, "stride_w") == 0) session->stride_w = value;
    else if (strcmp(name, "padding") == 0) session->padding = value;
    else if (strcmp(name, "pad_h") == 0) session->pad_h = value;
    else if (strcmp(name, "pad_w") == 0) session->pad_w = value;
    else if (strcmp(name, "output_n") == 0) session->output_n = value;
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
    else if (strcmp(name, "pool_h") == 0) session->pool_h = value;
    else if (strcmp(name, "pool_w") == 0) session->pool_w = value;
    else if (strcmp(name, "out_mult_alpha") == 0) session->out_mult_alpha = value;
    else if (strcmp(name, "out_shift_alpha") == 0) session->out_shift_alpha = value;
    else if (strcmp(name, "out_mult_fp") == 0) session->out_mult_fp = value;
    else if (strcmp(name, "out_mult_exp") == 0) session->out_mult_exp = value;
    else if (strcmp(name, "relu_mult_fp") == 0) session->relu_mult_fp = value;
    else if (strcmp(name, "relu_mult_exp") == 0) session->relu_mult_exp = value;
    else if (strcmp(name, "relu_q3") == 0) session->relu_q3 = value;
    else if (strcmp(name, "relu_q6") == 0) session->relu_q6 = value;
    else if (strcmp(name, "prescale") == 0) session->prescale = value;
    else if (strcmp(name, "input_mult") == 0) session->input_mult = value;
    else if (strcmp(name, "input_left_shift") == 0) session->input_left_shift = value;
    else if (strcmp(name, "alpha_offset") == 0) session->alpha_offset = value;
    else if (strcmp(name, "block_size") == 0) session->block_size = value;
    else if (strcmp(name, "scale_bits") == 0) session->scale_bits = value;
    else if (strcmp(name, "activation_kind") == 0) session->activation_kind = value;
    else if (strcmp(name, "num_rows") == 0) session->num_rows = value;
    else if (strcmp(name, "row_size") == 0) session->row_size = value;
    else if (strcmp(name, "diff_min") == 0) session->diff_min = value;
    else if (strcmp(name, "filter_offset") == 0) session->filter_offset = value;
    else if (strcmp(name, "axis_n") == 0) session->axis_n = value;
    else if (strcmp(name, "axis_h") == 0) session->axis_h = value;
    else if (strcmp(name, "axis_w") == 0) session->axis_w = value;
    else if (strcmp(name, "axis_c") == 0) session->axis_c = value;
    else if (strcmp(name, "axis") == 0) session->axis = value;
    else if (strcmp(name, "needs_rescale") == 0) session->needs_rescale = value;
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
    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    session->output_length = input->byte_length;
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (session->expected_kernel_id == HCT_KERNEL_ID_ABS_S16)
    {
#ifdef HCT_HOST_ABS_ONLY
        return ARM_CMSIS_NN_ARG_ERROR;
#else
        return arm_abs_s16((const int16_t *)blob_ptr(session, input),
                           session->input_offset,
                           (int16_t *)session->output_buffer,
                           session->output_offset,
                           session->out_mult,
                           session->out_shift,
                           session->needs_rescale != 0,
                           session->activation_min,
                           session->activation_max,
                           (int32_t)(input->byte_length / sizeof(int16_t)));
#endif
    }
    {
        hct_abs_s8_request_t request;
        request.input = (const int8_t *)blob_ptr(session, input);
        request.input_offset = session->input_offset;
        request.output = (int8_t *)session->output_buffer;
        request.output_offset = session->output_offset;
        request.output_multiplier = session->out_mult;
        request.output_shift = session->out_shift;
        request.activation_min = session->activation_min;
        request.activation_max = session->activation_max;
        request.block_size = (int32_t)input->byte_length;
        request.needs_rescale = (uint8_t)(session->needs_rescale != 0);
        return hct_dispatch_abs_s8(&request);
    }
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
    cmsis_nn_conv_params conv_params;
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

    if (session->expected_kernel_id == HCT_KERNEL_ID_CONVOLVE_S16)
    {
        /* arm_convolve_wrapper_s16 needs no weight-sum precompute (unlike S8's
         * arm_convolve_s8, whose bias must be pre-adjusted via arm_convolve_weight_sum())
         * and its bias is a cmsis_nn_bias_data struct wrapping a plain int64_t payload
         * (see arm_nnfunctions.h), not S8's raw int32_t* bias pointer. */
        cmsis_nn_bias_data bias_data = {blob_ptr(session, bias), false};
        int32_t required_scratch = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
        if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        ctx.buf = (session->scratch_bytes > 0u) ? &session->case_arena[session->scratch_offset] : NULL;
        ctx.size = session->scratch_bytes;

        /* session->output_length is transmitted to the host as a raw byte count (see the
         * RTT send loop below) -- compute_convolve_output_dims() above computed it in
         * elements, so rescale to bytes for 2-byte-per-element S16 output. */
        session->output_length = (uint32_t)(session->output_length * sizeof(int16_t));
        if (session->output_length > sizeof(session->output_buffer))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        return arm_convolve_wrapper_s16(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        (const int16_t *)blob_ptr(session, input),
                                        &filter_dims,
                                        (const int8_t *)blob_ptr(session, weights),
                                        &bias_dims,
                                        &bias_data,
                                        &output_dims,
                                        (int16_t *)session->output_buffer);
    }

    {
        /* S8 path: arm_convolve_s8 requires a precomputed per-output-channel weight sum
         * (via arm_convolve_weight_sum()) placed in its own scratch region, distinct from
         * the general im2col-style `ctx` scratch above. */
        cmsis_nn_context weight_sum_ctx;
        hct_convolve_s8_request_t request;
        int32_t required_scratch;
        uint32_t weight_sum_offset;
        uint32_t weight_sum_bytes;

        required_scratch = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
        if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
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
}

/* arm_depthwise_conv_s8 is the low-level (non-wrapper) depthwise conv kernel: unlike
 * arm_convolve_s8 above, its `ctx` and `bias_dims` args are both unused internally (see
 * Source/ConvolutionFunctions/arm_depthwise_conv_s8.c's `(void)ctx;`/`(void)bias_dims;`),
 * so no scratch buffer or weight-sum precomputation is required here. filter_dims stay in
 * the generator's native (N=1, H, W, C_OUT) order -- no HWCN reordering like Convolve's
 * filter_dims (depthwise's cmsis_nn_dw_conv_params filter convention is already NHWC).
 * The S16 variant (arm_depthwise_conv_wrapper_s16) needs a real scratch buffer and takes a
 * plain int64_t* bias pointer directly (unlike Convolve S16's cmsis_nn_bias_data-wrapped
 * bias) -- see arm_nnfunctions.h. */
static arm_cmsis_nn_status run_depthwise_conv_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *weights = find_blob_by_role(session, HCT_BLOB_ROLE_WEIGHTS);
    hct_server_blob_t *bias = find_blob_by_role(session, HCT_BLOB_ROLE_BIAS);
    hct_server_blob_t *multiplier = find_blob_by_role(session, HCT_BLOB_ROLE_MULTIPLIER);
    hct_server_blob_t *shift = find_blob_by_role(session, HCT_BLOB_ROLE_SHIFT);
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

    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    session->output_length = (uint32_t)(output_dims.n * output_dims.h * output_dims.w * output_dims.c);

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

    if (session->expected_kernel_id == HCT_KERNEL_ID_DEPTHWISE_CONV_S16)
    {
        cmsis_nn_context ctx;
        int32_t required_scratch = arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
        if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        ctx.buf = (session->scratch_bytes > 0u) ? &session->case_arena[session->scratch_offset] : NULL;
        ctx.size = session->scratch_bytes;

        session->output_length = (uint32_t)(session->output_length * sizeof(int16_t));
        if (session->output_length > sizeof(session->output_buffer))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        return arm_depthwise_conv_wrapper_s16(&ctx,
                                              &dw_conv_params,
                                              &quant_params,
                                              &input_dims,
                                              (const int16_t *)blob_ptr(session, input),
                                              &filter_dims,
                                              (const int8_t *)blob_ptr(session, weights),
                                              &bias_dims,
                                              (const int64_t *)blob_ptr(session, bias),
                                              &output_dims,
                                              (int16_t *)session->output_buffer);
    }

    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    {
        cmsis_nn_context ctx = {NULL, 0};
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
}

/* arm_avgpool_s8/arm_max_pool_s8 (and their S16 counterparts) all share the same
 * cmsis_nn_pool_params-based signature -- unlike Convolve/DepthwiseConv, pooling has no
 * input_offset/output_offset (see cmsis_nn_pool_params's definition in arm_nn_types.h:
 * only stride, padding, activation) and no weights/bias blobs at all, since a pool window
 * has no learned parameters -- just its size (session->pool_h/w, sent explicitly by the
 * host since there is no weights blob to read filter dims off of, unlike Convolve's).
 * MaxPool never needs a scratch buffer for either dtype. AvgPool needs one sized via
 * arm_avgpool_{s8,s16}_get_buffer_size(output_w, input_c) -- zero for many small cases,
 * so scratch_bytes may legitimately be 0 (case_arena pointer is never dereferenced when
 * ctx.size is 0). */
static arm_cmsis_nn_status run_pooling_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;
    cmsis_nn_context ctx;

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    filter_dims.n = 1;
    filter_dims.h = session->pool_h;
    filter_dims.w = session->pool_w;
    filter_dims.c = 1;

    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    pool_params.stride.w = (session->stride_w == 0) ? 1 : session->stride_w;
    pool_params.stride.h = (session->stride_h == 0) ? 1 : session->stride_h;
    pool_params.padding.w = session->pad_w;
    pool_params.padding.h = session->pad_h;
    pool_params.activation.min = session->activation_min;
    pool_params.activation.max = session->activation_max;

    if (session->expected_kernel_id == HCT_KERNEL_ID_MAXPOOL_S8 || session->expected_kernel_id == HCT_KERNEL_ID_MAXPOOL_S16)
    {
        ctx.buf = NULL;
        ctx.size = 0;
    }
    else
    {
        int32_t required_scratch = (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_S16)
            ? arm_avgpool_s16_get_buffer_size((int)output_dims.w, (int)input_dims.c)
            : arm_avgpool_s8_get_buffer_size((int)output_dims.w, (int)input_dims.c);
        if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        ctx.buf = (session->scratch_bytes > 0u) ? &session->case_arena[session->scratch_offset] : NULL;
        ctx.size = session->scratch_bytes;
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_S16 || session->expected_kernel_id == HCT_KERNEL_ID_MAXPOOL_S16)
    {
        session->output_length = (uint32_t)(output_dims.n * output_dims.h * output_dims.w * output_dims.c * (int32_t)sizeof(int16_t));
        if (session->output_length > sizeof(session->output_buffer))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_S16)
        {
            return arm_avgpool_s16(&ctx, &pool_params, &input_dims, (const int16_t *)blob_ptr(session, input),
                                   &filter_dims, &output_dims, (int16_t *)session->output_buffer);
        }
        return arm_max_pool_s16(&ctx, &pool_params, &input_dims, (const int16_t *)blob_ptr(session, input),
                                &filter_dims, &output_dims, (int16_t *)session->output_buffer);
    }

    session->output_length = (uint32_t)(output_dims.n * output_dims.h * output_dims.w * output_dims.c);
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_S8)
    {
        return arm_avgpool_s8(&ctx, &pool_params, &input_dims, (const int8_t *)blob_ptr(session, input),
                              &filter_dims, &output_dims, (int8_t *)session->output_buffer);
    }
    return arm_max_pool_s8(&ctx, &pool_params, &input_dims, (const int8_t *)blob_ptr(session, input),
                           &filter_dims, &output_dims, (int8_t *)session->output_buffer);
}

/* ActivationFunctions unary ops (Relu/Relu6/Clamp/LeakyRelu/Logistic/Tanh/HardSwish*) all
 * take a single input tensor and produce a same-shape output with no weights/bias blob and
 * no scratch buffer -- unlike Convolve/Pooling, their generated headers have no named
 * scalar-params struct either (see generated_test_bridge.py's _build_activation_case,
 * which reads the scalars positionally out of the generated .c file's call site, mirroring
 * how BasicMathFunctions elementwise ops are bridged). session->output_h/w/c holds the
 * (single) output shape; element count is used directly as each kernel's `size` argument. */
static arm_cmsis_nn_status run_activation_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    int32_t size;
    bool is_s16;

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (session->output_h <= 0 || session->output_w <= 0 || session->output_c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    size = session->output_h * session->output_w * session->output_c;

    is_s16 = (session->expected_kernel_id == HCT_KERNEL_ID_RELU_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_RELU6_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_CLAMP_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_LEAKY_RELU_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_LOGISTIC_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_TANH_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_HARD_SWISH_PRECISE_S16);
    session->output_length = (uint32_t)(size * (is_s16 ? (int32_t)sizeof(int16_t) : (int32_t)sizeof(int8_t)));
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (is_s16)
    {
        const int16_t *in16 = (const int16_t *)blob_ptr(session, input);
        int16_t *out16 = (int16_t *)session->output_buffer;
        switch (session->expected_kernel_id)
        {
            case HCT_KERNEL_ID_RELU_S16:
                return arm_relu_s16(in16, session->input_offset, session->output_offset,
                                    session->out_mult, session->out_shift, out16, size);
            case HCT_KERNEL_ID_RELU6_S16:
                return arm_relu_generic_s16(in16, session->input_offset, session->output_offset,
                                             session->out_mult, session->out_shift,
                                             session->activation_min, session->activation_max, out16, size);
            case HCT_KERNEL_ID_CLAMP_S16:
                return arm_clamp_s16(in16, (int16_t)session->activation_min, (int16_t)session->activation_max,
                                      out16, size);
            case HCT_KERNEL_ID_LEAKY_RELU_S16:
                return arm_leaky_relu_s16(in16, session->input_offset, session->output_offset,
                                          session->out_mult_alpha, session->out_shift_alpha,
                                          session->out_mult, session->out_shift, out16, size);
            case HCT_KERNEL_ID_LOGISTIC_S16:
                return arm_logistic_s16(in16, out16, size, session->input_mult, session->input_left_shift);
            case HCT_KERNEL_ID_TANH_S16:
                return arm_tanh_s16(in16, out16, size, session->input_mult, session->input_left_shift);
            case HCT_KERNEL_ID_HARD_SWISH_PRECISE_S16:
                return arm_hard_swish_precise_s16(in16, session->input_offset, session->output_offset,
                                                  session->out_mult, session->out_shift,
                                                  session->relu_q3, session->relu_q6, session->prescale,
                                                  out16, size);
            default:
                return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    else
    {
        const int8_t *in8 = (const int8_t *)blob_ptr(session, input);
        int8_t *out8 = (int8_t *)session->output_buffer;
        switch (session->expected_kernel_id)
        {
            case HCT_KERNEL_ID_RELU_S8:
                return arm_relu_s8(in8, session->input_offset, session->output_offset,
                                   session->out_mult, session->out_shift, out8, size);
            case HCT_KERNEL_ID_RELU6_S8:
                return arm_relu_generic_s8(in8, session->input_offset, session->output_offset,
                                           session->out_mult, session->out_shift,
                                           session->activation_min, session->activation_max, out8, size);
            case HCT_KERNEL_ID_CLAMP_S8:
                return arm_clamp_s8(in8, (int8_t)session->activation_min, (int8_t)session->activation_max,
                                    out8, size);
            case HCT_KERNEL_ID_LEAKY_RELU_S8:
                return arm_leaky_relu_s8(in8, session->input_offset, session->output_offset,
                                         session->out_mult_alpha, session->out_shift_alpha,
                                         session->out_mult, session->out_shift, out8, size);
            case HCT_KERNEL_ID_HARD_SWISH_COMPAT_S8:
                return arm_hard_swish_compat_s8(in8, session->input_offset, session->output_offset,
                                                session->out_mult_fp, session->out_mult_exp,
                                                session->relu_mult_fp, session->relu_mult_exp, out8, size);
            case HCT_KERNEL_ID_HARD_SWISH_PRECISE_S8:
                return arm_hard_swish_precise_s8(in8, session->input_offset, session->output_offset,
                                                 session->out_mult, session->out_shift,
                                                 session->relu_q3, session->relu_q6, session->prescale,
                                                 out8, size);
            default:
                return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
}

/* PReLU (arm_prelu_s8/s16 -- alpha broadcastable per cmsis_nn_dims semantics, same style as
 * elementwise binary Add/Sub) and PReLUScalar (arm_prelu_scalar_s8/s16 -- a direct
 * flat-vector API used when one side is a true per-pixel scalar; see arm_nnfunctions.h).
 * Both share input_offset/output_offset (own tensor's zero points, reused from the unary
 * activations above), out_mult/out_shift (the "identity" branch), and out_mult_alpha/
 * out_shift_alpha (reused from LeakyRelu's alpha branch -- identical semantics). alpha_offset
 * is the one new quantized scalar. PReLUScalar's `scalar_is_input` argument is always `true`
 * in every real generated test (see prelu_scalar.c.j2's hardcoded call), so it's hardcoded
 * here rather than added as a session field. */
static arm_cmsis_nn_status run_prelu_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *alpha = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_1);

    if (input == NULL || alpha == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_PRELU_SCALAR_S8 ||
        session->expected_kernel_id == HCT_KERNEL_ID_PRELU_SCALAR_S16)
    {
        bool is_s16 = (session->expected_kernel_id == HCT_KERNEL_ID_PRELU_SCALAR_S16);
        int32_t block_size = session->block_size;
        if (block_size <= 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        session->output_length = (uint32_t)(block_size * (is_s16 ? (int32_t)sizeof(int16_t) : (int32_t)sizeof(int8_t)));
        if (session->output_length > sizeof(session->output_buffer))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (is_s16)
        {
            return arm_prelu_scalar_s16((const int16_t *)blob_ptr(session, input),
                                       (const int16_t *)blob_ptr(session, alpha),
                                       true,
                                       session->input_offset, session->alpha_offset, session->output_offset,
                                       session->out_mult, session->out_shift,
                                       session->out_mult_alpha, session->out_shift_alpha,
                                       (int16_t *)session->output_buffer, block_size);
        }
        return arm_prelu_scalar_s8((const int8_t *)blob_ptr(session, input),
                                   (const int8_t *)blob_ptr(session, alpha),
                                   true,
                                   session->input_offset, session->alpha_offset, session->output_offset,
                                   session->out_mult, session->out_shift,
                                   session->out_mult_alpha, session->out_shift_alpha,
                                   (int8_t *)session->output_buffer, block_size);
    }

    cmsis_nn_dims input_dims;
    cmsis_nn_dims alpha_dims;
    cmsis_nn_dims output_dims;
    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    alpha_dims.n = (int32_t)alpha->dimensions[0];
    alpha_dims.h = (int32_t)alpha->dimensions[1];
    alpha_dims.w = (int32_t)alpha->dimensions[2];
    alpha_dims.c = (int32_t)alpha->dimensions[3];
    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    bool out_is_s16 = (session->expected_kernel_id == HCT_KERNEL_ID_PRELU_S16);
    session->output_length = (uint32_t)(output_dims.n * output_dims.h * output_dims.w * output_dims.c *
                                        (out_is_s16 ? (int32_t)sizeof(int16_t) : (int32_t)sizeof(int8_t)));
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (out_is_s16)
    {
        return arm_prelu_s16(&input_dims, (const int16_t *)blob_ptr(session, input),
                             &alpha_dims, (const int16_t *)blob_ptr(session, alpha),
                             session->input_offset, session->alpha_offset, session->output_offset,
                             session->out_mult, session->out_shift,
                             session->out_mult_alpha, session->out_shift_alpha,
                             &output_dims, (int16_t *)session->output_buffer);
    }
    return arm_prelu_s8(&input_dims, (const int8_t *)blob_ptr(session, input),
                        &alpha_dims, (const int8_t *)blob_ptr(session, alpha),
                        session->input_offset, session->alpha_offset, session->output_offset,
                        session->out_mult, session->out_shift,
                        session->out_mult_alpha, session->out_shift_alpha,
                        &output_dims, (int8_t *)session->output_buffer);
}

/* Reinterprets session->scale_bits (transmitted as a raw int32) back into the float scale
 * value it was bit-cast from on the host -- see the scale_bits field comment in
 * benchmark_server_session.h. */
static float quant_scale_from_bits(int32_t bits)
{
    float scale;
    memcpy(&scale, &bits, sizeof(scale));
    return scale;
}

/* arm_quantize_f32_{s8,s16} (float input -> quantized output). The generated test's ReLU/
 * ReLU6 activation (if any) is applied to the float input BEFORE this kernel call in the
 * real TFLite-parity template, entirely in float space -- so the host-side bridge folds it
 * into the input blob it sends (see _build_quantize_case() in generated_test_bridge.py),
 * and this wrapper only ever needs to invoke the kernel itself. */
static arm_cmsis_nn_status run_quantize_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    int32_t size;
    float scale;

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (session->output_h <= 0 || session->output_w <= 0 || session->output_c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    size = session->output_h * session->output_w * session->output_c;
    scale = quant_scale_from_bits(session->scale_bits);

    if (session->expected_kernel_id == HCT_KERNEL_ID_QUANTIZE_S16)
    {
        session->output_length = (uint32_t)(size * (int32_t)sizeof(int16_t));
        if (session->output_length > sizeof(session->output_buffer))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_quantize_f32_s16((const float *)blob_ptr(session, input),
                                    (int16_t *)session->output_buffer,
                                    size, session->output_offset, scale);
    }
    session->output_length = (uint32_t)size;
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    return arm_quantize_f32_s8((const float *)blob_ptr(session, input),
                              (int8_t *)session->output_buffer,
                              size, session->output_offset, scale);
}

/* arm_dequantize_{s8,s16}_f32 (quantized input -> float output). Unlike Quantize, the
 * generated test's ReLU/ReLU6 activation (if any) is applied AFTER this kernel call, to
 * the dequantized float output -- so, unlike Quantize, it must be replicated here to match
 * the golden output (see activation_kind field comment in benchmark_server_session.h). Reuses
 * quant_scale_from_bits() defined above by the Quantize adapter. */
static arm_cmsis_nn_status run_dequantize_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    int32_t size;
    float scale;
    float *out;
    arm_cmsis_nn_status status;

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (session->output_h <= 0 || session->output_w <= 0 || session->output_c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    size = session->output_h * session->output_w * session->output_c;
    session->output_length = (uint32_t)(size * (int32_t)sizeof(float));
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    scale = quant_scale_from_bits(session->scale_bits);
    out = (float *)session->output_buffer;

    if (session->expected_kernel_id == HCT_KERNEL_ID_DEQUANTIZE_S16)
    {
        status = arm_dequantize_s16_f32((const int16_t *)blob_ptr(session, input),
                                        out, size, session->input_offset, scale);
    }
    else
    {
        status = arm_dequantize_s8_f32((const int8_t *)blob_ptr(session, input),
                                       out, size, session->input_offset, scale);
    }
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return status;
    }

    if (session->activation_kind == 1) /* RELU */
    {
        int32_t i;
        for (i = 0; i < size; ++i)
        {
            if (out[i] < 0.0f) out[i] = 0.0f;
        }
    }
    else if (session->activation_kind == 2) /* RELU6 */
    {
        int32_t i;
        for (i = 0; i < size; ++i)
        {
            if (out[i] < 0.0f) out[i] = 0.0f;
            if (out[i] > 6.0f) out[i] = 6.0f;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status run_requantize_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    session->output_length = input->byte_length;
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_REQUANTIZE_S16)
    {
        return arm_requantize_s16_s16((const int16_t *)blob_ptr(session, input),
                                      (int16_t *)session->output_buffer,
                                      (int32_t)(input->byte_length / sizeof(int16_t)),
                                      session->out_mult,
                                      session->out_shift,
                                      session->input_offset,
                                      session->output_offset);
    }
    return arm_requantize_s8_s8((const int8_t *)blob_ptr(session, input),
                                (int8_t *)session->output_buffer,
                                (int32_t)input->byte_length,
                                session->out_mult,
                                session->out_shift,
                                session->input_offset,
                                session->output_offset);
}

static arm_cmsis_nn_status run_comparison_once(hct_server_session_t *session)
{
    hct_server_blob_t *input_1 = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *input_2 = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_1);
    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;
    int32_t output_size;

    if (input_1 == NULL || input_2 == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_1_dims.n = (int32_t)input_1->dimensions[0];
    input_1_dims.h = (int32_t)input_1->dimensions[1];
    input_1_dims.w = (int32_t)input_1->dimensions[2];
    input_1_dims.c = (int32_t)input_1->dimensions[3];

    input_2_dims.n = (int32_t)input_2->dimensions[0];
    input_2_dims.h = (int32_t)input_2->dimensions[1];
    input_2_dims.w = (int32_t)input_2->dimensions[2];
    input_2_dims.c = (int32_t)input_2->dimensions[3];

    output_dims.n = session->output_n;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    output_size = output_dims.n * output_dims.h * output_dims.w * output_dims.c;
    if (output_size <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    session->output_length = (uint32_t)output_size;
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_EQUAL_S8:
            return arm_equal_s8(NULL,
                                (const int8_t *)blob_ptr(session, input_1),
                                &input_1_dims,
                                (const int8_t *)blob_ptr(session, input_2),
                                &input_2_dims,
                                (bool *)session->output_buffer,
                                &output_dims,
                                session->input1_offset,
                                session->input1_mult,
                                session->input1_shift,
                                session->input2_offset,
                                session->input2_mult,
                                session->input2_shift,
                                session->left_shift);
        case HCT_KERNEL_ID_NOT_EQUAL_S8:
            return arm_not_equal_s8(NULL,
                                    (const int8_t *)blob_ptr(session, input_1),
                                    &input_1_dims,
                                    (const int8_t *)blob_ptr(session, input_2),
                                    &input_2_dims,
                                    (bool *)session->output_buffer,
                                    &output_dims,
                                    session->input1_offset,
                                    session->input1_mult,
                                    session->input1_shift,
                                    session->input2_offset,
                                    session->input2_mult,
                                    session->input2_shift,
                                    session->left_shift);
        case HCT_KERNEL_ID_GREATER_S8:
            return arm_greater_s8(NULL,
                                  (const int8_t *)blob_ptr(session, input_1),
                                  &input_1_dims,
                                  (const int8_t *)blob_ptr(session, input_2),
                                  &input_2_dims,
                                  (bool *)session->output_buffer,
                                  &output_dims,
                                  session->input1_offset,
                                  session->input1_mult,
                                  session->input1_shift,
                                  session->input2_offset,
                                  session->input2_mult,
                                  session->input2_shift,
                                  session->left_shift);
        case HCT_KERNEL_ID_GREATER_EQUAL_S8:
            return arm_greater_equal_s8(NULL,
                                        (const int8_t *)blob_ptr(session, input_1),
                                        &input_1_dims,
                                        (const int8_t *)blob_ptr(session, input_2),
                                        &input_2_dims,
                                        (bool *)session->output_buffer,
                                        &output_dims,
                                        session->input1_offset,
                                        session->input1_mult,
                                        session->input1_shift,
                                        session->input2_offset,
                                        session->input2_mult,
                                        session->input2_shift,
                                        session->left_shift);
        case HCT_KERNEL_ID_LESS_S8:
            return arm_less_s8(NULL,
                               (const int8_t *)blob_ptr(session, input_1),
                               &input_1_dims,
                               (const int8_t *)blob_ptr(session, input_2),
                               &input_2_dims,
                               (bool *)session->output_buffer,
                               &output_dims,
                               session->input1_offset,
                               session->input1_mult,
                               session->input1_shift,
                               session->input2_offset,
                               session->input2_mult,
                               session->input2_shift,
                               session->left_shift);
        case HCT_KERNEL_ID_LESS_EQUAL_S8:
            return arm_less_equal_s8(NULL,
                                     (const int8_t *)blob_ptr(session, input_1),
                                     &input_1_dims,
                                     (const int8_t *)blob_ptr(session, input_2),
                                     &input_2_dims,
                                     (bool *)session->output_buffer,
                                     &output_dims,
                                     session->input1_offset,
                                     session->input1_mult,
                                     session->input1_shift,
                                     session->input2_offset,
                                     session->input2_mult,
                                     session->input2_shift,
                                     session->left_shift);
        case HCT_KERNEL_ID_EQUAL_S16:
            return arm_equal_s16(NULL,
                                 (const int16_t *)blob_ptr(session, input_1),
                                 &input_1_dims,
                                 (const int16_t *)blob_ptr(session, input_2),
                                 &input_2_dims,
                                 (bool *)session->output_buffer,
                                 &output_dims,
                                 session->input1_offset,
                                 session->input1_mult,
                                 session->input1_shift,
                                 session->input2_offset,
                                 session->input2_mult,
                                 session->input2_shift,
                                 session->left_shift);
        case HCT_KERNEL_ID_NOT_EQUAL_S16:
            return arm_not_equal_s16(NULL,
                                     (const int16_t *)blob_ptr(session, input_1),
                                     &input_1_dims,
                                     (const int16_t *)blob_ptr(session, input_2),
                                     &input_2_dims,
                                     (bool *)session->output_buffer,
                                     &output_dims,
                                     session->input1_offset,
                                     session->input1_mult,
                                     session->input1_shift,
                                     session->input2_offset,
                                     session->input2_mult,
                                     session->input2_shift,
                                     session->left_shift);
        case HCT_KERNEL_ID_GREATER_S16:
            return arm_greater_s16(NULL,
                                   (const int16_t *)blob_ptr(session, input_1),
                                   &input_1_dims,
                                   (const int16_t *)blob_ptr(session, input_2),
                                   &input_2_dims,
                                   (bool *)session->output_buffer,
                                   &output_dims,
                                   session->input1_offset,
                                   session->input1_mult,
                                   session->input1_shift,
                                   session->input2_offset,
                                   session->input2_mult,
                                   session->input2_shift,
                                   session->left_shift);
        case HCT_KERNEL_ID_GREATER_EQUAL_S16:
            return arm_greater_equal_s16(NULL,
                                         (const int16_t *)blob_ptr(session, input_1),
                                         &input_1_dims,
                                         (const int16_t *)blob_ptr(session, input_2),
                                         &input_2_dims,
                                         (bool *)session->output_buffer,
                                         &output_dims,
                                         session->input1_offset,
                                         session->input1_mult,
                                         session->input1_shift,
                                         session->input2_offset,
                                         session->input2_mult,
                                         session->input2_shift,
                                         session->left_shift);
        case HCT_KERNEL_ID_LESS_S16:
            return arm_less_s16(NULL,
                                (const int16_t *)blob_ptr(session, input_1),
                                &input_1_dims,
                                (const int16_t *)blob_ptr(session, input_2),
                                &input_2_dims,
                                (bool *)session->output_buffer,
                                &output_dims,
                                session->input1_offset,
                                session->input1_mult,
                                session->input1_shift,
                                session->input2_offset,
                                session->input2_mult,
                                session->input2_shift,
                                session->left_shift);
        case HCT_KERNEL_ID_LESS_EQUAL_S16:
            return arm_less_equal_s16(NULL,
                                      (const int16_t *)blob_ptr(session, input_1),
                                      &input_1_dims,
                                      (const int16_t *)blob_ptr(session, input_2),
                                      &input_2_dims,
                                      (bool *)session->output_buffer,
                                      &output_dims,
                                      session->input1_offset,
                                      session->input1_mult,
                                      session->input1_shift,
                                      session->input2_offset,
                                      session->input2_mult,
                                      session->input2_shift,
                                      session->left_shift);
        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}

/* Fixed CMSIS-NN reference lookup tables required by arm_softmax_s16() -- identical bit
 * patterns are used by every generated S16 softmax test case (see
 * Tests/helia-core-tester/assets/templates/SoftmaxFunctions/softmax/softmax.h.j2), so they
 * are embedded once as firmware constants rather than transmitted per case. */
static const int16_t hct_softmax_exp_lut[513] = {

    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     3,     3,     3,     3,     3,
    3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     4,     4,     4,     4,
    4,     4,     4,     4,     4,     4,     4,     4,     4,     5,     5,     5,     5,     5,     5,     5,
    5,     5,     5,     6,     6,     6,     6,     6,     6,     6,     6,     7,     7,     7,     7,     7,
    7,     7,     7,     8,     8,     8,     8,     8,     8,     9,     9,     9,     9,     9,     9,     10,
    10,    10,    10,    10,    11,    11,    11,    11,    11,    12,    12,    12,    12,    13,    13,    13,
    13,    14,    14,    14,    14,    15,    15,    15,    16,    16,    16,    17,    17,    17,    18,    18,
    18,    19,    19,    19,    20,    20,    21,    21,    21,    22,    22,    23,    23,    24,    24,    25,
    25,    26,    26,    27,    27,    28,    28,    29,    29,    30,    30,    31,    32,    32,    33,    34,
    34,    35,    36,    36,    37,    37,    38,    39,    40,    40,    42,    42,    43,    44,    45,    45,
    46,    47,    48,    49,    50,    51,    52,    53,    54,    55,    56,    57,    59,    60,    60,    62,
    63,    65,    65,    67,    68,    69,    71,    73,    74,    75,    77,    78,    80,    81,    83,    85,
    86,    88,    90,    92,    93,    95,    97,    99,    101,   103,   105,   107,   109,   112,   114,   116,
    118,   121,   123,   126,   128,   131,   133,   135,   139,   141,   144,   147,   149,   152,   155,   158,
    162,   165,   168,   171,   174,   178,   181,   185,   189,   192,   196,   200,   204,   208,   212,   217,
    221,   225,   230,   234,   239,   243,   248,   253,   258,   263,   268,   273,   279,   284,   290,   296,
    302,   308,   314,   320,   327,   333,   340,   346,   353,   360,   366,   374,   381,   389,   397,   404,
    413,   421,   429,   437,   446,   455,   464,   473,   482,   492,   501,   511,   522,   532,   543,   553,
    564,   575,   586,   598,   610,   622,   634,   646,   659,   672,   685,   699,   713,   727,   741,   756,
    771,   786,   801,   817,   833,   850,   866,   884,   901,   919,   937,   955,   974,   993,   1013,  1033,
    1053,  1074,  1095,  1117,  1139,  1161,  1184,  1207,  1232,  1256,  1281,  1306,  1332,  1358,  1385,  1412,
    1440,  1468,  1497,  1527,  1557,  1587,  1619,  1651,  1683,  1716,  1750,  1785,  1820,  1856,  1892,  1930,
    1968,  2006,  2046,  2087,  2128,  2170,  2212,  2256,  2300,  2346,  2392,  2439,  2488,  2537,  2587,  2638,
    2690,  2743,  2796,  2852,  2908,  2966,  3024,  3084,  3145,  3207,  3270,  3334,  3400,  3467,  3535,  3605,
    3677,  3749,  3822,  3898,  3975,  4053,  4133,  4214,  4297,  4383,  4469,  4557,  4647,  4739,  4833,  4927,
    5024,  5124,  5225,  5328,  5433,  5541,  5649,  5761,  5875,  5991,  6109,  6230,  6352,  6477,  6605,  6736,
    6868,  7004,  7141,  7282,  7427,  7572,  7722,  7874,  8030,  8188,  8350,  8514,  8683,  8854,  9028,  9206,
    9387,  9572,  9762,  9954,  10151, 10351, 10555, 10763, 10976, 11191, 11412, 11637, 11867, 12102, 12341, 12583,
    12831, 13085, 13342, 13606, 13874, 14148, 14427, 14711, 15002, 15297, 15599, 15907, 16221, 16541, 16867, 17199,
    17539, 17884, 18237, 18597, 18964, 19338, 19719, 20108, 20505, 20909, 21322, 21742, 22171, 22608, 23054, 23509,
    23973, 24445, 24928, 25419, 25921, 26432, 26953, 27485, 28027, 28580, 29143, 29718, 30304, 30902, 31512, 32133,
    32767
};
static const int16_t hct_softmax_one_by_one_lut[513] = {

    32767, 32704, 32640, 32578, 32514, 32451, 32388, 32326, 32264, 32202, 32141, 32079, 32018, 31957, 31896, 31835,
    31775, 31715, 31655, 31596, 31537, 31476, 31418, 31359, 31301, 31242, 31184, 31127, 31069, 31011, 30954, 30897,
    30840, 30784, 30727, 30671, 30615, 30560, 30504, 30449, 30394, 30339, 30283, 30229, 30175, 30121, 30067, 30013,
    29960, 29906, 29853, 29800, 29746, 29694, 29642, 29589, 29537, 29486, 29434, 29382, 29331, 29280, 29229, 29177,
    29127, 29076, 29026, 28976, 28926, 28877, 28827, 28777, 28728, 28679, 28630, 28581, 28532, 28484, 28436, 28388,
    28340, 28292, 28244, 28197, 28150, 28103, 28056, 28008, 27962, 27915, 27869, 27823, 27777, 27731, 27685, 27640,
    27594, 27549, 27504, 27459, 27413, 27369, 27324, 27280, 27236, 27192, 27148, 27104, 27060, 27016, 26973, 26930,
    26887, 26844, 26801, 26758, 26715, 26673, 26630, 26588, 26546, 26504, 26463, 26421, 26380, 26338, 26297, 26255,
    26214, 26174, 26132, 26092, 26051, 26011, 25971, 25931, 25891, 25851, 25811, 25772, 25732, 25693, 25653, 25614,
    25575, 25536, 25497, 25458, 25420, 25381, 25343, 25305, 25267, 25229, 25191, 25153, 25116, 25078, 25041, 25003,
    24966, 24928, 24892, 24855, 24818, 24781, 24745, 24709, 24672, 24636, 24600, 24564, 24528, 24492, 24457, 24421,
    24385, 24350, 24315, 24280, 24245, 24210, 24175, 24140, 24105, 24070, 24036, 24002, 23967, 23933, 23899, 23865,
    23831, 23798, 23764, 23730, 23697, 23664, 23630, 23597, 23564, 23530, 23498, 23465, 23432, 23399, 23366, 23334,
    23302, 23269, 23237, 23205, 23173, 23141, 23109, 23077, 23046, 23014, 22982, 22951, 22920, 22888, 22857, 22826,
    22795, 22764, 22733, 22703, 22672, 22641, 22611, 22580, 22550, 22520, 22490, 22459, 22429, 22400, 22370, 22340,
    22310, 22281, 22251, 22221, 22192, 22163, 22134, 22104, 22075, 22046, 22017, 21988, 21959, 21931, 21902, 21874,
    21845, 21817, 21788, 21760, 21732, 21704, 21676, 21648, 21620, 21592, 21565, 21537, 21509, 21482, 21455, 21427,
    21400, 21372, 21345, 21318, 21291, 21264, 21237, 21210, 21183, 21157, 21130, 21103, 21077, 21050, 21024, 20998,
    20971, 20945, 20919, 20893, 20867, 20841, 20816, 20790, 20764, 20738, 20713, 20687, 20662, 20636, 20611, 20586,
    20560, 20535, 20510, 20485, 20460, 20435, 20410, 20385, 20360, 20336, 20311, 20287, 20262, 20238, 20213, 20189,
    20165, 20141, 20117, 20092, 20068, 20044, 20021, 19997, 19973, 19949, 19926, 19902, 19878, 19855, 19832, 19808,
    19784, 19762, 19738, 19715, 19692, 19668, 19645, 19622, 19600, 19577, 19553, 19531, 19508, 19485, 19463, 19440,
    19418, 19395, 19373, 19351, 19328, 19306, 19284, 19262, 19240, 19218, 19196, 19174, 19152, 19130, 19109, 19087,
    19065, 19044, 19022, 19000, 18979, 18958, 18936, 18915, 18893, 18872, 18851, 18830, 18809, 18787, 18766, 18745,
    18725, 18704, 18682, 18662, 18641, 18620, 18600, 18579, 18559, 18538, 18518, 18497, 18477, 18457, 18436, 18416,
    18396, 18376, 18356, 18336, 18316, 18296, 18276, 18256, 18236, 18216, 18197, 18177, 18157, 18138, 18118, 18099,
    18079, 18059, 18040, 18021, 18001, 17982, 17963, 17944, 17924, 17905, 17886, 17867, 17848, 17829, 17810, 17791,
    17772, 17754, 17735, 17716, 17697, 17679, 17660, 17641, 17623, 17604, 17586, 17568, 17549, 17531, 17513, 17494,
    17476, 17458, 17440, 17422, 17404, 17386, 17368, 17350, 17332, 17314, 17296, 17278, 17261, 17243, 17225, 17208,
    17190, 17172, 17155, 17137, 17120, 17102, 17085, 17067, 17050, 17033, 17015, 16999, 16981, 16964, 16947, 16930,
    16913, 16895, 16878, 16862, 16845, 16828, 16810, 16794, 16777, 16760, 16743, 16727, 16710, 16693, 16677, 16660,
    16644, 16627, 16611, 16594, 16578, 16562, 16545, 16529, 16513, 16497, 16480, 16464, 16448, 16432, 16416, 16400,
    16384
};
static const cmsis_nn_softmax_lut_s16 hct_softmax_lut_s16 = {
    .exp_lut = hct_softmax_exp_lut,
    .one_by_one_lut = hct_softmax_one_by_one_lut
};

/* arm_softmax_s8/arm_softmax_s16/arm_softmax_s8_s16 -- all three share the same
 * (num_rows, row_size, mult, shift[, diff_min]) requantization scheme; softmax always
 * operates over the last dimension of a flattened 2D (num_rows, row_size) view, sent
 * explicitly as scalar params by the host (see _build_softmax_case() in
 * generated_test_bridge.py). out_mult/out_shift are reused for mult/shift; diff_min is only
 * meaningful for the two int8-input kernels (arm_softmax_s8, arm_softmax_s8_s16) and is
 * left unused (0) for the pure-S16 kernel. */
static arm_cmsis_nn_status run_softmax_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    int32_t size;

    if (input == NULL || session->num_rows <= 0 || session->row_size <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    size = session->num_rows * session->row_size;

    if (session->expected_kernel_id == HCT_KERNEL_ID_SOFTMAX_S16)
    {
        session->output_length = (uint32_t)(size * (int32_t)sizeof(int16_t));
        if (session->output_length > sizeof(session->output_buffer))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_softmax_s16((const int16_t *)blob_ptr(session, input),
                               session->num_rows, session->row_size,
                               session->out_mult, session->out_shift,
                               &hct_softmax_lut_s16,
                               (int16_t *)session->output_buffer);
    }
    if (session->expected_kernel_id == HCT_KERNEL_ID_SOFTMAX_S8_S16)
    {
        session->output_length = (uint32_t)(size * (int32_t)sizeof(int16_t));
        if (session->output_length > sizeof(session->output_buffer))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        arm_softmax_s8_s16((const int8_t *)blob_ptr(session, input),
                           session->num_rows, session->row_size,
                           session->out_mult, session->out_shift, session->diff_min,
                           (int16_t *)session->output_buffer);
        return ARM_CMSIS_NN_SUCCESS;
    }
    session->output_length = (uint32_t)size;
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    arm_softmax_s8((const int8_t *)blob_ptr(session, input),
                   session->num_rows, session->row_size,
                   session->out_mult, session->out_shift, session->diff_min,
                   (int8_t *)session->output_buffer);
    return ARM_CMSIS_NN_SUCCESS;
}

/* FullyConnectedFunctions FullyConnected -- both S8 and S16 always dispatch through the
 * "per-channel" wrapper entry points (arm_fully_connected_wrapper_s8/_s16) with an
 * array-shaped multiplier/shift blob, broadcast to a uniform value across every output
 * channel for genuinely per-tensor-quantized descriptors, rather than replicating the two
 * distinct scalar-vs-array call shapes CMSIS-NN exposes. Broadcasting a constant across
 * all channels of the per-channel kernel is mathematically identical to the scalar
 * per-tensor kernel (per-tensor quantization is simply the degenerate case of per-channel
 * with every channel equal) -- confirmed against the real generated code, whose "default"
 * (non-explicitly-per-channel) descriptors already emit per-channel-shaped arrays too. This
 * one code path therefore handles every generated FullyConnected test case uniformly,
 * mirroring how Convolve's own array-based quant_params blob already transparently handles
 * both cases.
 *
 * S8's ctx->buf must hold a precomputed "kernel_sum" (one int32 per output channel: the
 * row-wise weight sum scaled by input_offset/filter_offset, with the real bias folded in)
 * -- computed here at runtime via the real CMSIS-NN arm_vector_sum_s8() kernel, after which
 * NULL is passed as the wrapper's own bias argument (it is already baked into kernel_sum),
 * exactly matching what the real generated code does (see fully_connected.c.j2's
 * `ctx.buf = ..._weight_sum;` / `NULL` bias-argument pattern, confirmed even for the
 * "per_tensor" S8 descriptors). S16's ctx->buf is instead just scratch memory the kernel
 * fills itself (a per-channel "reduced_multiplier" cache, per
 * arm_fully_connected_get_buffer_sizes_s16.c) -- no weight-sum precompute needed, and its
 * bias is passed directly as a real int64_t* array (or NULL) since FC only precomputes a
 * weight-sum for S8 output (see fully_connected.py's `_should_precompute_weight_sum()`).
 *
 * Both dtypes' ctx buffer is sized filter_dims.c * sizeof(int32_t) (i.e. output_units * 4
 * bytes) -- see arm_fully_connected_{s8,per_channel_s16}_get_buffer_size{,_mve}() -- sent
 * by the host via CASE_META's scratch_buffer.bytes, identical mechanism to Convolve/
 * DepthwiseConv/Pooling's scratch sizing.
 *
 * Weights blob dimensions are transmitted as (output_units, input_features) (the natural
 * numpy weight-matrix shape) -- filter_dims.c/.n are read directly from that, not
 * reordered like Convolve's HWCN filter_dims convention (FullyConnected's filter_dims is
 * already the CMSIS-NN native (n=input_features "col_dim", c=output_units "row_dim")
 * layout once mapped this way). The batch dimension (input_dims.n / output_dims.n) is
 * handled entirely inside the CMSIS-NN kernel's own per-batch loop, so batch > 1 is
 * supported without any extra host-side looping, unlike Convolve's single-invocation
 * batch-1-only bridge restriction. */
static arm_cmsis_nn_status run_fully_connected_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *weights = find_blob_by_role(session, HCT_BLOB_ROLE_WEIGHTS);
    hct_server_blob_t *bias = find_blob_by_role(session, HCT_BLOB_ROLE_BIAS); /* optional */
    hct_server_blob_t *multiplier = find_blob_by_role(session, HCT_BLOB_ROLE_MULTIPLIER);
    hct_server_blob_t *shift = find_blob_by_role(session, HCT_BLOB_ROLE_SHIFT);
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    uint32_t required_scratch;

    if (input == NULL || weights == NULL || multiplier == NULL || shift == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = 1;
    input_dims.w = 1;
    input_dims.c = (int32_t)input->dimensions[1];
    filter_dims.c = (int32_t)weights->dimensions[0];
    filter_dims.n = (int32_t)weights->dimensions[1];
    filter_dims.h = 1;
    filter_dims.w = 1;
    bias_dims.n = 0;
    bias_dims.h = 0;
    bias_dims.w = 0;
    bias_dims.c = filter_dims.c;
    output_dims.n = input_dims.n;
    output_dims.h = 1;
    output_dims.w = 1;
    output_dims.c = filter_dims.c;

    required_scratch = (uint32_t)filter_dims.c * (uint32_t)sizeof(int32_t);
    if (required_scratch > session->scratch_bytes)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    ctx.buf = (session->scratch_bytes > 0u) ? &session->case_arena[session->scratch_offset] : NULL;
    ctx.size = (int32_t)session->scratch_bytes;

    fc_params.input_offset = session->input_offset;
    fc_params.filter_offset = session->filter_offset;
    fc_params.output_offset = session->output_offset;
    fc_params.activation.min = session->activation_min;
    fc_params.activation.max = session->activation_max;
    quant_params.multiplier = (int32_t *)blob_ptr(session, multiplier);
    quant_params.shift = (int32_t *)blob_ptr(session, shift);
    quant_params.is_per_channel = 1;

    if (session->expected_kernel_id == HCT_KERNEL_ID_FULLY_CONNECTED_S16)
    {
        const int64_t *bias_i64 = (bias != NULL) ? (const int64_t *)blob_ptr(session, bias) : NULL;

        session->output_length = (uint32_t)(output_dims.n * output_dims.c * (int32_t)sizeof(int16_t));
        if (session->output_length > sizeof(session->output_buffer))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_fully_connected_wrapper_s16(&ctx,
                                               &fc_params,
                                               &quant_params,
                                               &input_dims,
                                               (const int16_t *)blob_ptr(session, input),
                                               &filter_dims,
                                               (const int8_t *)blob_ptr(session, weights),
                                               &bias_dims,
                                               bias_i64,
                                               &output_dims,
                                               (int16_t *)session->output_buffer);
    }

    {
        const int32_t *bias_i32 = (bias != NULL) ? (const int32_t *)blob_ptr(session, bias) : NULL;

        if (arm_vector_sum_s8((int32_t *)ctx.buf,
                              filter_dims.n,
                              filter_dims.c,
                              (const int8_t *)blob_ptr(session, weights),
                              session->input_offset,
                              session->filter_offset,
                              bias_i32) != ARM_CMSIS_NN_SUCCESS)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        session->output_length = (uint32_t)(output_dims.n * output_dims.c);
        if (session->output_length > sizeof(session->output_buffer))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_fully_connected_wrapper_s8(&ctx,
                                              &fc_params,
                                              &quant_params,
                                              &input_dims,
                                              (const int8_t *)blob_ptr(session, input),
                                              &filter_dims,
                                              (const int8_t *)blob_ptr(session, weights),
                                              &bias_dims,
                                              NULL,
                                              &output_dims,
                                              (int8_t *)session->output_buffer);
    }
}

/* FullyConnectedFunctions BatchMatMul -- arm_batch_matmul_s8/_s16 both take two same-
 * dtype operands (unlike FullyConnected, there is no separate always-S8 "weights" tensor:
 * for S16 both lhs and rhs are int16_t) and a single per-tensor cmsis_nn_per_tensor_quant_
 * params (a plain {multiplier, shift} struct, not the array-shaped per-channel blob
 * FullyConnected needs) -- reusing session->out_mult/out_shift is sufficient, no new
 * multiplier/shift blob roles required.
 *
 * Neither kernel reads bmm_params->adj_x/adj_y (see arm_batch_matmul_s8.c's own "Does not
 * perform transposes" comment) -- the real generated test's transposed-operand descriptors
 * already pre-arrange their raw lhs/rhs header array data and dims into the final
 * row-major layout the kernel expects, so this bridge only needs to stream that data/dims
 * through unchanged; no transpose flag is transmitted at all.
 *
 * The real generated test harness always uses a single-invocation shape with
 * input_lhs_dims/input_rhs_dims/output_dims.n == .h == 1 (batch/height looping handled
 * entirely inside the kernel's own loop) regardless of how many logical batches the
 * descriptor name implies -- exactly the same "batch is cosmetic at the single-invocation
 * level" pattern already established for FullyConnected. Blob dimensions are transmitted
 * as compact 2-tuples (rows, cols) per operand/output, matching FullyConnected's wire
 * convention; dims.n/.h are always reconstructed here as 1. Because
 * arm_nn_vec_mat_mult_t_s8/s16 (the per-row primitive this kernel calls) treats rhs as
 * already-transposed ([N, K] rather than [K, N]), input_rhs_dims.w is N (the shared
 * output/reduction-count dimension -- output_dims.c) while input_rhs_dims.c is K (the
 * inner dimension shared with input_lhs_dims.c), NOT the more intuitive "rows=M,
 * cols=N" reading -- output_dims.c must be read off input_rhs_dims.w, not .c.
 *
 * S8's ctx->buf is an N-sized (input_rhs_dims.w) int32 kernel-sum scratch buffer the
 * kernel itself fills at runtime via its own internal arm_vector_sum_s8() call (unlike
 * FullyConnected, which must precompute+bake in a real bias here) -- sized via the same
 * arm_fully_connected_s8_get_buffer_size() helper FullyConnected already uses, since
 * BatchMatMul's rhs plays the identical "filter" role for buffer-sizing purposes (see
 * arm_batch_matmul_s8.c's own "we use RHS dims as filter_dims for buffer size
 * calculation" comment, which maps filter_dims.c = N = input_rhs_dims.w). S16 needs no
 * scratch at all (ctx is unused in arm_batch_matmul_s16 -- no vector_sum precompute in
 * the S16 path). */
static arm_cmsis_nn_status run_batch_matmul_once(hct_server_session_t *session)
{
    hct_server_blob_t *input_lhs = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *input_rhs = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_1);
    cmsis_nn_context ctx;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_lhs_dims;
    cmsis_nn_dims input_rhs_dims;
    cmsis_nn_dims output_dims;

    if (input_lhs == NULL || input_rhs == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_lhs_dims.n = 1;
    input_lhs_dims.h = 1;
    input_lhs_dims.w = (int32_t)input_lhs->dimensions[0];
    input_lhs_dims.c = (int32_t)input_lhs->dimensions[1];
    input_rhs_dims.n = 1;
    input_rhs_dims.h = 1;
    input_rhs_dims.w = (int32_t)input_rhs->dimensions[0];
    input_rhs_dims.c = (int32_t)input_rhs->dimensions[1];
    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = 1;
    output_dims.w = input_lhs_dims.w;
    output_dims.c = input_rhs_dims.w;

    /* cmsis_nn_bmm_params's adj_x/adj_y are `const bool` fields, which makes the whole
     * struct non-assignable after declaration in C -- must be fully initialized here via
     * a designated initializer instead of member-by-member assignment. */
    const cmsis_nn_bmm_params bmm_params = {
        .adj_x = false,
        .adj_y = false,
        .fc_params = {
            .input_offset = session->input_offset,
            .filter_offset = session->filter_offset,
            .output_offset = session->output_offset,
            .activation = { .min = session->activation_min, .max = session->activation_max }
        }
    };
    quant_params.multiplier = session->out_mult;
    quant_params.shift = session->out_shift;

    if (session->expected_kernel_id == HCT_KERNEL_ID_BATCH_MATMUL_S16)
    {
        session->output_length = (uint32_t)(output_dims.w * output_dims.c * (int32_t)sizeof(int16_t));
        if (session->output_length > sizeof(session->output_buffer))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        ctx.buf = NULL;
        ctx.size = 0;
        return arm_batch_matmul_s16(&ctx,
                                    &bmm_params,
                                    &quant_params,
                                    &input_lhs_dims,
                                    (const int16_t *)blob_ptr(session, input_lhs),
                                    &input_rhs_dims,
                                    (const int16_t *)blob_ptr(session, input_rhs),
                                    &output_dims,
                                    (int16_t *)session->output_buffer);
    }

    {
        const uint32_t required_scratch = (uint32_t)input_rhs_dims.w * (uint32_t)sizeof(int32_t);
        if (required_scratch > session->scratch_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        ctx.buf = (session->scratch_bytes > 0u) ? &session->case_arena[session->scratch_offset] : NULL;
        ctx.size = (int32_t)session->scratch_bytes;

        session->output_length = (uint32_t)(output_dims.w * output_dims.c);
        if (session->output_length > sizeof(session->output_buffer))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_batch_matmul_s8(&ctx,
                                   &bmm_params,
                                   &quant_params,
                                   &input_lhs_dims,
                                   (const int8_t *)blob_ptr(session, input_lhs),
                                   &input_rhs_dims,
                                   (const int8_t *)blob_ptr(session, input_rhs),
                                   &output_dims,
                                   (int8_t *)session->output_buffer);
    }
}

static arm_cmsis_nn_status run_basic_math_reduction_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    cmsis_nn_dims input_dims;
    cmsis_nn_dims axis_dims;
    cmsis_nn_dims output_dims;
    uint32_t output_elements;

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];

    axis_dims.n = session->axis_n;
    axis_dims.h = session->axis_h;
    axis_dims.w = session->axis_w;
    axis_dims.c = session->axis_c;

    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.n <= 0 || output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    output_elements = (uint32_t)(output_dims.n * output_dims.h * output_dims.w * output_dims.c);

    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_ARGMAX_S8:
        case HCT_KERNEL_ID_ARGMIN_S8:
        case HCT_KERNEL_ID_ARGMAX_S16:
        case HCT_KERNEL_ID_ARGMIN_S16:
            session->output_length = output_elements * sizeof(int32_t);
            if (session->output_length > sizeof(session->output_buffer) || session->axis < 0 || session->axis > 3)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_ARGMAX_S8)
            {
                return arm_argmax_s8((const int8_t *)blob_ptr(session, input), &input_dims, session->axis, (int32_t *)session->output_buffer);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_ARGMIN_S8)
            {
                return arm_argmin_s8((const int8_t *)blob_ptr(session, input), &input_dims, session->axis, (int32_t *)session->output_buffer);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_ARGMAX_S16)
            {
                return arm_argmax_s16((const int16_t *)blob_ptr(session, input), &input_dims, session->axis, (int32_t *)session->output_buffer);
            }
            return arm_argmin_s16((const int16_t *)blob_ptr(session, input), &input_dims, session->axis, (int32_t *)session->output_buffer);

        case HCT_KERNEL_ID_MEAN_S8:
        case HCT_KERNEL_ID_REDUCE_MAX_S8:
        case HCT_KERNEL_ID_REDUCE_MIN_S8:
            session->output_length = output_elements;
            if (session->output_length > sizeof(session->output_buffer))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_MEAN_S8)
            {
                return arm_mean_s8((const int8_t *)blob_ptr(session, input), &input_dims, session->input_offset,
                                   &axis_dims, (int8_t *)session->output_buffer, &output_dims,
                                   session->output_offset, session->out_mult, session->out_shift);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_REDUCE_MAX_S8)
            {
                return arm_reduce_max_s8((const int8_t *)blob_ptr(session, input), &input_dims, &axis_dims,
                                         (int8_t *)session->output_buffer, &output_dims);
            }
            return arm_reduce_min_s8((const int8_t *)blob_ptr(session, input), &input_dims, &axis_dims,
                                     (int8_t *)session->output_buffer, &output_dims);

        case HCT_KERNEL_ID_MEAN_S16:
        case HCT_KERNEL_ID_REDUCE_MAX_S16:
        case HCT_KERNEL_ID_REDUCE_MIN_S16:
            session->output_length = output_elements * sizeof(int16_t);
            if (session->output_length > sizeof(session->output_buffer))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_MEAN_S16)
            {
                return arm_mean_s16((const int16_t *)blob_ptr(session, input), &input_dims, session->input_offset,
                                    &axis_dims, (int16_t *)session->output_buffer, &output_dims,
                                    session->output_offset, session->out_mult, session->out_shift);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_REDUCE_MAX_S16)
            {
                return arm_reduce_max_s16((const int16_t *)blob_ptr(session, input), &input_dims, &axis_dims,
                                          (int16_t *)session->output_buffer, &output_dims);
            }
            return arm_reduce_min_s16((const int16_t *)blob_ptr(session, input), &input_dims, &axis_dims,
                                      (int16_t *)session->output_buffer, &output_dims);

        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}

static arm_cmsis_nn_status run_basic_math_lut_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *lut = find_blob_by_role(session, HCT_BLOB_ROLE_WEIGHTS);
    cmsis_nn_dims input_dims;
    int32_t block_size;

    if (input == NULL || lut == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    block_size = input_dims.n * input_dims.h * input_dims.w * input_dims.c;
    session->output_length = input->byte_length;
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_SQRT_S8:
            return arm_sqrt_s8((const int8_t *)blob_ptr(session, input), &input_dims,
                               (int8_t *)session->output_buffer, (int8_t *)blob_ptr(session, lut));
        case HCT_KERNEL_ID_SQRT_S16:
            return arm_sqrt_s16((const int16_t *)blob_ptr(session, input), &input_dims,
                                (int16_t *)session->output_buffer, (const int16_t *)blob_ptr(session, lut));
        case HCT_KERNEL_ID_RSQRT_S16_PER_OP:
            return arm_rsqrt_s16_per_op((const int16_t *)blob_ptr(session, input), session->input_offset,
                                        (int16_t *)session->output_buffer, session->output_offset,
                                        session->activation_min, session->activation_max, block_size,
                                        (const int16_t *)blob_ptr(session, lut));
        case HCT_KERNEL_ID_RSQRT_S16_UNIVERSAL:
            return arm_rsqrt_s16_universal((const int16_t *)blob_ptr(session, input), session->input_offset,
                                           (int16_t *)session->output_buffer, session->output_offset,
                                           session->out_mult, session->out_shift, session->needs_rescale != 0,
                                           session->activation_min, session->activation_max, block_size,
                                           (const int32_t *)blob_ptr(session, lut));
        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}

/* arm_add_s8/arm_sub_s8 (and their S16 counterparts arm_add_s16/arm_sub_s16) share an
 * identical signature/argument order per dtype; all are dispatched from this one wrapper,
 * branching only on session->expected_kernel_id (see assets/kernel_registry.yaml for the
 * kernel_id <-> (operator, dtype) mapping). The S16 kernels have the exact same argument
 * shape as their S8 counterparts (just int16_t data pointers) -- see arm_nnfunctions.h --
 * so no separate scalar fields or output-dims handling are needed for S16. Ground-truth
 * output dims (session->output_h/w/c) are sent explicitly by the host, same rationale as
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

    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    session->output_length = (uint32_t)(output_dims.n * output_dims.h * output_dims.w * output_dims.c);

    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_ADD_S8:
        case HCT_KERNEL_ID_SUB_S8:
        case HCT_KERNEL_ID_MUL_S8:
        case HCT_KERNEL_ID_MAXIMUM_S8:
        case HCT_KERNEL_ID_MINIMUM_S8:
        case HCT_KERNEL_ID_SQUARED_DIFFERENCE_S8:
        {
            if (session->output_length > sizeof(session->output_buffer))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            const int8_t *input1_data = (const int8_t *)blob_ptr(session, input1);
            const int8_t *input2_data = (const int8_t *)blob_ptr(session, input2);
            int8_t *output_data = (int8_t *)session->output_buffer;
            if (session->expected_kernel_id == HCT_KERNEL_ID_ADD_S8)
            {
                return arm_add_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                                  session->input1_offset, session->input1_mult, session->input1_shift,
                                  session->input2_offset, session->input2_mult, session->input2_shift,
                                  session->left_shift,
                                  output_data, &output_dims,
                                  session->output_offset, session->out_mult, session->out_shift,
                                  session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SUB_S8)
            {
                return arm_sub_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                                  session->input1_offset, session->input1_mult, session->input1_shift,
                                  session->input2_offset, session->input2_mult, session->input2_shift,
                                  session->left_shift,
                                  output_data, &output_dims,
                                  session->output_offset, session->out_mult, session->out_shift,
                                  session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_MUL_S8)
            {
                /* arm_mul_s8 has no per-input mult/shift or left_shift -- it reuses only the
                 * input1_offset/input2_offset scalar fields (shared with Add/Sub above). */
                return arm_mul_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                                  session->input1_offset, session->input2_offset,
                                  output_data, &output_dims,
                                  session->output_offset, session->out_mult, session->out_shift,
                                  session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SQUARED_DIFFERENCE_S8)
            {
                return arm_squared_difference_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                                                 session->input1_offset, session->input1_mult, session->input1_shift,
                                                 session->input2_offset, session->input2_mult, session->input2_shift,
                                                 session->left_shift,
                                                 output_data, &output_dims,
                                                 session->output_offset, session->out_mult, session->out_shift,
                                                 session->activation_min, session->activation_max);
            }
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
        case HCT_KERNEL_ID_ADD_S16:
        case HCT_KERNEL_ID_SUB_S16:
        case HCT_KERNEL_ID_MUL_S16:
        case HCT_KERNEL_ID_MAXIMUM_S16:
        case HCT_KERNEL_ID_MINIMUM_S16:
        case HCT_KERNEL_ID_SQUARED_DIFFERENCE_S16:
        {
            if (session->output_length * sizeof(int16_t) > sizeof(session->output_buffer))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            /* session->output_length is transmitted to the host as a raw byte count (see
             * the RTT send loop in benchmark_server_session.c), so it must be rescaled from
             * elements to bytes for 2-byte-per-element S16 output -- otherwise only the
             * first half of the output buffer is sent back over the wire. */
            session->output_length = (uint32_t)(session->output_length * sizeof(int16_t));
            const int16_t *input1_data = (const int16_t *)blob_ptr(session, input1);
            const int16_t *input2_data = (const int16_t *)blob_ptr(session, input2);
            int16_t *output_data = (int16_t *)session->output_buffer;
            if (session->expected_kernel_id == HCT_KERNEL_ID_ADD_S16)
            {
                return arm_add_s16(input1_data, &input1_dims, input2_data, &input2_dims,
                                   session->input1_offset, session->input1_mult, session->input1_shift,
                                   session->input2_offset, session->input2_mult, session->input2_shift,
                                   session->left_shift,
                                   output_data, &output_dims,
                                   session->output_offset, session->out_mult, session->out_shift,
                                   session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SUB_S16)
            {
                return arm_sub_s16(input1_data, &input1_dims, input2_data, &input2_dims,
                                   session->input1_offset, session->input1_mult, session->input1_shift,
                                   session->input2_offset, session->input2_mult, session->input2_shift,
                                   session->left_shift,
                                   output_data, &output_dims,
                                   session->output_offset, session->out_mult, session->out_shift,
                                   session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_MUL_S16)
            {
                /* arm_mul_s16 mirrors arm_mul_s8's (shorter) signature: no per-input mult/shift. */
                return arm_mul_s16(input1_data, &input1_dims, input2_data, &input2_dims,
                                   session->input1_offset, session->input2_offset,
                                   output_data, &output_dims,
                                   session->output_offset, session->out_mult, session->out_shift,
                                   session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SQUARED_DIFFERENCE_S16)
            {
                return arm_squared_difference_s16(input1_data, &input1_dims, input2_data, &input2_dims,
                                                  session->input1_offset, session->input1_mult, session->input1_shift,
                                                  session->input2_offset, session->input2_mult, session->input2_shift,
                                                  session->left_shift,
                                                  output_data, &output_dims,
                                                  session->output_offset, session->out_mult, session->out_shift,
                                                  session->activation_min, session->activation_max);
            }
            /* Maximum/Minimum S16 mirror the S8 variants: scratch ctx only, no quant scalars. */
            cmsis_nn_context ctx16 = {NULL, 0};
            if (session->expected_kernel_id == HCT_KERNEL_ID_MAXIMUM_S16)
            {
                return arm_maximum_s16(&ctx16, input1_data, &input1_dims, input2_data, &input2_dims,
                                       output_data, &output_dims);
            }
            return arm_minimum_s16(&ctx16, input1_data, &input1_dims, input2_data, &input2_dims,
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
        case HCT_KERNEL_ID_ABS_S16:
            return run_abs_once(session);
        case HCT_KERNEL_ID_CONVOLVE_S8:
        case HCT_KERNEL_ID_CONVOLVE_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_convolve_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_DEPTHWISE_CONV_S8:
        case HCT_KERNEL_ID_DEPTHWISE_CONV_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_depthwise_conv_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_AVGPOOL_S8:
        case HCT_KERNEL_ID_AVGPOOL_S16:
        case HCT_KERNEL_ID_MAXPOOL_S8:
        case HCT_KERNEL_ID_MAXPOOL_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_pooling_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_RELU_S8:
        case HCT_KERNEL_ID_RELU_S16:
        case HCT_KERNEL_ID_RELU6_S8:
        case HCT_KERNEL_ID_RELU6_S16:
        case HCT_KERNEL_ID_CLAMP_S8:
        case HCT_KERNEL_ID_CLAMP_S16:
        case HCT_KERNEL_ID_LEAKY_RELU_S8:
        case HCT_KERNEL_ID_LEAKY_RELU_S16:
        case HCT_KERNEL_ID_LOGISTIC_S16:
        case HCT_KERNEL_ID_TANH_S16:
        case HCT_KERNEL_ID_HARD_SWISH_COMPAT_S8:
        case HCT_KERNEL_ID_HARD_SWISH_PRECISE_S8:
        case HCT_KERNEL_ID_HARD_SWISH_PRECISE_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_activation_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_PRELU_S8:
        case HCT_KERNEL_ID_PRELU_S16:
        case HCT_KERNEL_ID_PRELU_SCALAR_S8:
        case HCT_KERNEL_ID_PRELU_SCALAR_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_prelu_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_QUANTIZE_S8:
        case HCT_KERNEL_ID_QUANTIZE_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_quantize_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_DEQUANTIZE_S8:
        case HCT_KERNEL_ID_DEQUANTIZE_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_dequantize_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_REQUANTIZE_S8:
        case HCT_KERNEL_ID_REQUANTIZE_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_requantize_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_EQUAL_S8:
        case HCT_KERNEL_ID_EQUAL_S16:
        case HCT_KERNEL_ID_NOT_EQUAL_S8:
        case HCT_KERNEL_ID_NOT_EQUAL_S16:
        case HCT_KERNEL_ID_GREATER_S8:
        case HCT_KERNEL_ID_GREATER_S16:
        case HCT_KERNEL_ID_GREATER_EQUAL_S8:
        case HCT_KERNEL_ID_GREATER_EQUAL_S16:
        case HCT_KERNEL_ID_LESS_S8:
        case HCT_KERNEL_ID_LESS_S16:
        case HCT_KERNEL_ID_LESS_EQUAL_S8:
        case HCT_KERNEL_ID_LESS_EQUAL_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_comparison_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_SOFTMAX_S8:
        case HCT_KERNEL_ID_SOFTMAX_S16:
        case HCT_KERNEL_ID_SOFTMAX_S8_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_softmax_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_FULLY_CONNECTED_S8:
        case HCT_KERNEL_ID_FULLY_CONNECTED_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_fully_connected_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_BATCH_MATMUL_S8:
        case HCT_KERNEL_ID_BATCH_MATMUL_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_batch_matmul_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_ARGMAX_S8:
        case HCT_KERNEL_ID_ARGMAX_S16:
        case HCT_KERNEL_ID_ARGMIN_S8:
        case HCT_KERNEL_ID_ARGMIN_S16:
        case HCT_KERNEL_ID_MEAN_S8:
        case HCT_KERNEL_ID_MEAN_S16:
        case HCT_KERNEL_ID_REDUCE_MAX_S8:
        case HCT_KERNEL_ID_REDUCE_MAX_S16:
        case HCT_KERNEL_ID_REDUCE_MIN_S8:
        case HCT_KERNEL_ID_REDUCE_MIN_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_basic_math_reduction_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_RSQRT_S16_PER_OP:
        case HCT_KERNEL_ID_RSQRT_S16_UNIVERSAL:
        case HCT_KERNEL_ID_SQRT_S8:
        case HCT_KERNEL_ID_SQRT_S16:
#ifndef HCT_HOST_ABS_ONLY
            return run_basic_math_lut_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
        case HCT_KERNEL_ID_ADD_S8:
        case HCT_KERNEL_ID_SUB_S8:
        case HCT_KERNEL_ID_MUL_S8:
        case HCT_KERNEL_ID_MAXIMUM_S8:
        case HCT_KERNEL_ID_MINIMUM_S8:
        case HCT_KERNEL_ID_SQUARED_DIFFERENCE_S8:
        case HCT_KERNEL_ID_ADD_S16:
        case HCT_KERNEL_ID_SUB_S16:
        case HCT_KERNEL_ID_MUL_S16:
        case HCT_KERNEL_ID_MAXIMUM_S16:
        case HCT_KERNEL_ID_MINIMUM_S16:
        case HCT_KERNEL_ID_SQUARED_DIFFERENCE_S16:
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
    /* Must be large enough for the longest string this handler reads via
     * read_text(): the case_id (up to HCT_SERVER_MAX_CASE_ID, e.g. the 79-char
     * FullyConnected per-channel descriptor names), not just the short
     * scalar-name/role/dtype strings that also flow through this same buffer. */
    char scratch[HCT_SERVER_MAX_CASE_ID];
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
