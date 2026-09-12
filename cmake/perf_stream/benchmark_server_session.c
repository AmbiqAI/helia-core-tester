#include "benchmark_server_session.h"
#include "benchmark_server_adapters.h"
#include "benchmark_server_validation.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "benchmark_server_adapter.h"
#include "benchmark_server_catalog.h"
#include "benchmark_server_messages.h"
#include "arm_nnfunctions.h"

#ifdef HELIA_HARDWARE_BUILD
#include "am_mcu_apollo.h"
#endif

/* The Armv8.1-M PMU (8 x 16-bit event counters + 32-bit CCNTR on Cortex-M55) is only
 * present when the device header says so; a Cortex-M4 hardware build or the host
 * harness compile take the DWT-only path below. */
#if defined(__PMU_PRESENT) && (__PMU_PRESENT == 1)
#include "pmu_armv8.h"
#define HCT_PMU_AVAILABLE 1
#else
#define HCT_PMU_AVAILABLE 0
#endif

static bool has_capacity(size_t payload_length, size_t offset, size_t needed)
{
    return offset + needed <= payload_length;
}

/* Bounded cursor API (F006): every primitive read below verifies that enough bytes
 * remain in the payload *before* it dereferences the buffer or advances the offset.
 * Once a read runs off the end, the cursor is latched into an overrun state -- every
 * subsequent read becomes a harmless no-op (returns 0/false without touching the
 * buffer or offset again) so callers can keep composing reads without re-checking
 * after every single call, and simply test cursor.overrun (or the return value) once
 * at a convenient point to detect any truncation across the whole sequence. This
 * replaces the previous unchecked read_u8/u16/u32/i32/text helpers, which indexed the
 * buffer and advanced the offset unconditionally -- a short/truncated SESSION_PLAN,
 * CASE_META, or BLOB_CHUNK payload could walk the cursor arbitrarily far past the
 * validated payload_length. */
typedef struct
{
    const uint8_t *buffer;
    size_t length;
    size_t offset;
    bool overrun;
} hct_cursor_t;

static void cursor_init(hct_cursor_t *cursor, const uint8_t *buffer, size_t length)
{
    cursor->buffer = buffer;
    cursor->length = length;
    cursor->offset = 0u;
    cursor->overrun = false;
}

static bool cursor_require(hct_cursor_t *cursor, size_t needed)
{
    if (cursor->overrun || !has_capacity(cursor->length, cursor->offset, needed))
    {
        cursor->overrun = true;
        return false;
    }
    return true;
}

static uint8_t cursor_u8(hct_cursor_t *cursor)
{
    uint8_t value;
    if (!cursor_require(cursor, 1u))
    {
        return 0u;
    }
    value = cursor->buffer[cursor->offset];
    cursor->offset += 1u;
    return value;
}

static uint16_t cursor_u16(hct_cursor_t *cursor)
{
    uint16_t value;
    if (!cursor_require(cursor, 2u))
    {
        return 0u;
    }
    value = (uint16_t)cursor->buffer[cursor->offset] | ((uint16_t)cursor->buffer[cursor->offset + 1u] << 8);
    cursor->offset += 2u;
    return value;
}

static uint32_t cursor_u32(hct_cursor_t *cursor)
{
    uint32_t value;
    if (!cursor_require(cursor, 4u))
    {
        return 0u;
    }
    value = (uint32_t)cursor->buffer[cursor->offset]
          | ((uint32_t)cursor->buffer[cursor->offset + 1u] << 8)
          | ((uint32_t)cursor->buffer[cursor->offset + 2u] << 16)
          | ((uint32_t)cursor->buffer[cursor->offset + 3u] << 24);
    cursor->offset += 4u;
    return value;
}

static int32_t cursor_i32(hct_cursor_t *cursor)
{
    return (int32_t)cursor_u32(cursor);
}

static bool cursor_text(hct_cursor_t *cursor, char *dest, size_t dest_capacity)
{
    size_t index;
    uint16_t length;
    if (cursor->overrun)
    {
        return false;
    }
    length = cursor_u16(cursor);
    if (cursor->overrun)
    {
        return false;
    }
    if (!cursor_require(cursor, length) || (size_t)length + 1u > dest_capacity)
    {
        cursor->overrun = true;
        return false;
    }
    for (index = 0u; index < length; ++index)
    {
        dest[index] = (char)cursor->buffer[cursor->offset + index];
    }
    dest[length] = '\0';
    cursor->offset += length;
    return true;
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

static hctp_status_t write_i32(uint8_t *buffer, size_t capacity, size_t *offset, int32_t value)
{
    return write_u32(buffer, capacity, offset, (uint32_t)value);
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

static uint8_t role_from_name(const char *name)
{
    if (strcmp(name, "input_0") == 0) return HCT_BLOB_ROLE_INPUT_0;
    if (strcmp(name, "weights") == 0) return HCT_BLOB_ROLE_WEIGHTS;
    if (strcmp(name, "bias") == 0) return HCT_BLOB_ROLE_BIAS;
    if (strcmp(name, "multiplier") == 0) return HCT_BLOB_ROLE_MULTIPLIER;
    if (strcmp(name, "shift") == 0) return HCT_BLOB_ROLE_SHIFT;
    if (strcmp(name, "input_1") == 0) return HCT_BLOB_ROLE_INPUT_1;
    if (strcmp(name, "input_2") == 0) return HCT_BLOB_ROLE_INPUT_2;
    if (strcmp(name, "meta_0") == 0) return HCT_BLOB_ROLE_META_0;
    return HCT_BLOB_ROLE_UNKNOWN;
}

static uint8_t dtype_from_name(const char *name)
{
    if (strcmp(name, "S8") == 0) return HCT_DTYPE_S8;
    if (strcmp(name, "S32") == 0) return HCT_DTYPE_S32;
    if (strcmp(name, "S16") == 0) return HCT_DTYPE_S16;
    if (strcmp(name, "S64") == 0) return HCT_DTYPE_S64;
    if (strcmp(name, "S4") == 0) return HCT_DTYPE_S4;
    if (strcmp(name, "BOOL") == 0) return HCT_DTYPE_BOOL;
    if (strcmp(name, "FP32") == 0) return HCT_DTYPE_F32;
    if (strcmp(name, "FP16") == 0) return HCT_DTYPE_F16;
    return HCT_DTYPE_UNKNOWN;
}

static bool kernel_status_is_fatal(const hct_server_session_t *session, arm_cmsis_nn_status status)
{
    return (status != ARM_CMSIS_NN_SUCCESS) && !expects_exact_status(session);
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

/* Best-effort ERROR reply so a rejected message (bad SESSION_PLAN/CASE_META/
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

    write_i32(payload, sizeof(payload), &offset, session->last_kernel_status);
    if (queue_frame(session, HCTP_MSG_CORRECTNESS_RESULT, payload, offset) != HCTP_STATUS_OK) return HCTP_STATUS_TRUNCATED_FRAME;

    offset = 0u;
    write_u32(payload, sizeof(payload), &offset, 0u);
    write_u32(payload, sizeof(payload), &offset, session->output_length);
    if (queue_frame(session, HCTP_MSG_OUTPUT_BEGIN, payload, offset) != HCTP_STATUS_OK) return HCTP_STATUS_TRUNCATED_FRAME;

    session->output_stream_offset = 0u;
    session->output_stream_checksum = 0u;
    session->output_stream_active = 1u;
    session->state = HCT_SERVER_STATE_STREAM_OUTPUT;
    return HCTP_STATUS_OK;
}

static hctp_status_t pump_correctness_output(hct_server_session_t *session)
{
    uint8_t payload[256];
    size_t offset = 0u;

    if (session->output_stream_active == 0u)
    {
        return HCTP_STATUS_OK;
    }
    if (session->output_stream_offset < session->output_length)
    {
        uint32_t index;
        const uint32_t remaining = session->output_length - session->output_stream_offset;
        const uint32_t chunk_length = (remaining > 224u) ? 224u : remaining;
        write_u32(payload, sizeof(payload), &offset, session->output_stream_offset);
        write_u32(payload, sizeof(payload), &offset, chunk_length);
        memcpy(&payload[offset], &hct_output_ptr(session)[session->output_stream_offset], chunk_length);
        for (index = 0u; index < chunk_length; ++index)
        {
            session->output_stream_checksum += hct_output_ptr(session)[session->output_stream_offset + index];
        }
        offset += chunk_length;
        session->output_stream_offset += chunk_length;
        return queue_frame(session, HCTP_MSG_OUTPUT_CHUNK, payload, offset);
    }

    write_u32(payload, sizeof(payload), &offset, session->output_length);
    write_u32(payload, sizeof(payload), &offset, session->output_stream_checksum);
    session->output_stream_active = 0u;
    session->state = HCT_SERVER_STATE_WAIT_CORRECTNESS_ACK;
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
#else
static void enable_dwt(void) {}
static uint32_t dwt_cycles(void) { return 0u; }
#endif

/* One measured sample's PMU readings for a pass: the cycle counter plus one value per
 * requested event counter. `supported` is 0 on a DWT-only build, where only `ccntr`
 * (taken from DWT) is meaningful. */
typedef struct
{
    uint32_t ccntr;
    uint8_t ccntr_overflow;
    uint32_t values[HCT_SERVER_MAX_COUNTERS_PER_PASS];
    uint8_t overflow[HCT_SERVER_MAX_COUNTERS_PER_PASS];
    uint8_t supported[HCT_SERVER_MAX_COUNTERS_PER_PASS];
} hct_pmu_sample_t;

#define HCT_PMU_EVENT_CPU_CYCLES 0x0011u

#if HCT_PMU_AVAILABLE
#define HCT_PMU_CCNTR_BIT (1u << 31)
#define HCT_PMU_EVCNTR_MASK 0xFFFFu

/* Slot layout for a pass: counter i lives in slot i, or -- when chained -- in slots 2i
 * (the event) and 2i+1 (ARM_PMU_CHAIN, incrementing on the even slot's overflow) so the
 * pair reads as one 32-bit counter. */
static uint32_t pmu_pass_slot_mask(const hct_pmu_pass_t *pass)
{
    uint32_t mask = 0u;
    uint32_t index;
    for (index = 0u; index < pass->count; ++index)
    {
        if (pass->chained)
        {
            mask |= 3u << (2u * index);
        }
        else
        {
            mask |= 1u << index;
        }
    }
    return mask;
}

static void pmu_pass_program(const hct_pmu_pass_t *pass)
{
    uint32_t index;
    ARM_PMU_Disable();
    ARM_PMU_CNTR_Disable(0xFFFFFFFFu);
    for (index = 0u; index < pass->count; ++index)
    {
        if (pass->chained)
        {
            ARM_PMU_Set_EVTYPER(2u * index, pass->event_ids[index]);
            ARM_PMU_Set_EVTYPER(2u * index + 1u, ARM_PMU_CHAIN);
        }
        else
        {
            ARM_PMU_Set_EVTYPER(index, pass->event_ids[index]);
        }
    }
}

/* Reset every counter and the overflow status, then start the pass's slots and CCNTR
 * together. Called immediately before the DWT start read of each sample. */
static void pmu_sample_start(uint32_t slot_mask)
{
    ARM_PMU_Disable();
    ARM_PMU_CNTR_Disable(0xFFFFFFFFu);
    ARM_PMU_EVCNTR_ALL_Reset();
    ARM_PMU_CYCCNT_Reset();
    ARM_PMU_Set_CNTR_OVS(0xFFFFFFFFu);
    ARM_PMU_CNTR_Enable(slot_mask | HCT_PMU_CCNTR_BIT);
    ARM_PMU_Enable();
}

/* Stop the counters, read them and the overflow status register (bit n = slot n,
 * bit 31 = CCNTR), and clear exactly the overflow bits that were set. A chained
 * pair's overflow is the high (odd) slot's bit; the low slot overflowing is what
 * feeds the chain and is expected. */
static void pmu_sample_stop(const hct_pmu_pass_t *pass, uint32_t slot_mask, uint32_t dwt_elapsed, hct_pmu_sample_t *out)
{
    uint32_t ovs;
    uint32_t index;
    (void)dwt_elapsed;
    ARM_PMU_CNTR_Disable(slot_mask | HCT_PMU_CCNTR_BIT);
    out->ccntr = ARM_PMU_Get_CCNTR();
    ovs = ARM_PMU_Get_CNTR_OVS();
    out->ccntr_overflow = (ovs & HCT_PMU_CCNTR_BIT) ? 1u : 0u;
    for (index = 0u; index < pass->count; ++index)
    {
        if (pass->chained)
        {
            const uint32_t low = ARM_PMU_Get_EVCNTR(2u * index) & HCT_PMU_EVCNTR_MASK;
            const uint32_t high = ARM_PMU_Get_EVCNTR(2u * index + 1u) & HCT_PMU_EVCNTR_MASK;
            out->values[index] = (high << 16) | low;
            out->overflow[index] = (ovs & (1u << (2u * index + 1u))) ? 1u : 0u;
        }
        else
        {
            out->values[index] = ARM_PMU_Get_EVCNTR(index) & HCT_PMU_EVCNTR_MASK;
            out->overflow[index] = (ovs & (1u << index)) ? 1u : 0u;
        }
        out->supported[index] = 1u;
    }
    if (ovs != 0u)
    {
        ARM_PMU_Set_CNTR_OVS(ovs);
    }
    ARM_PMU_Disable();
}
#else
static uint32_t pmu_pass_slot_mask(const hct_pmu_pass_t *pass) { (void)pass; return 0u; }
static void pmu_pass_program(const hct_pmu_pass_t *pass) { (void)pass; }
static void pmu_sample_start(uint32_t slot_mask) { (void)slot_mask; }
static void pmu_sample_stop(const hct_pmu_pass_t *pass, uint32_t slot_mask, uint32_t dwt_elapsed, hct_pmu_sample_t *out)
{
    uint32_t index;
    (void)slot_mask;
    out->ccntr = dwt_elapsed;
    out->ccntr_overflow = 0u;
    for (index = 0u; index < pass->count; ++index)
    {
        out->values[index] = 0u;
        out->overflow[index] = 0u;
        out->supported[index] = 0u;
    }
}
#endif

/* SAMPLE_RESULT: u16 sample_index, u32 iterations, u64 cycles (DWT), text pass_name,
 * u8 counter_count, then per counter (text name, u16 event_id, u64 value, u8 overflow,
 * u8 supported). Names are sent empty -- the host resolves them from its catalog by
 * event id. The first entry is always ARM_PMU_CPU_CYCLES from CCNTR (bit 31 of the
 * overflow status), so DWT `cycles` stays an independent cross-check. */
static hctp_status_t queue_sample_result(hct_server_session_t *session,
                                         uint16_t sample_index,
                                         uint32_t iterations,
                                         uint64_t cycles,
                                         const hct_pmu_pass_t *pass,
                                         const hct_pmu_sample_t *sample)
{
    uint8_t payload[256];
    size_t offset = 0u;
    uint32_t index;

    write_u16(payload, sizeof(payload), &offset, sample_index);
    write_u32(payload, sizeof(payload), &offset, iterations);
    write_u64(payload, sizeof(payload), &offset, cycles);
    write_text(payload, sizeof(payload), &offset, pass->name);
    write_u8(payload, sizeof(payload), &offset, (uint8_t)(1u + pass->count));

    write_text(payload, sizeof(payload), &offset, "");
    write_u16(payload, sizeof(payload), &offset, HCT_PMU_EVENT_CPU_CYCLES);
    write_u64(payload, sizeof(payload), &offset, (uint64_t)sample->ccntr);
    write_u8(payload, sizeof(payload), &offset, sample->ccntr_overflow);
    write_u8(payload, sizeof(payload), &offset, 1u);

    for (index = 0u; index < pass->count; ++index)
    {
        write_text(payload, sizeof(payload), &offset, "");
        write_u16(payload, sizeof(payload), &offset, pass->event_ids[index]);
        write_u64(payload, sizeof(payload), &offset, (uint64_t)sample->values[index]);
        write_u8(payload, sizeof(payload), &offset, sample->overflow[index]);
        if (write_u8(payload, sizeof(payload), &offset, sample->supported[index]) != HCTP_STATUS_OK)
        {
            return HCTP_STATUS_TRUNCATED_FRAME;
        }
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
    write_u32(payload, sizeof(payload), &offset, session->workspace_used_bytes);
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
    const uint32_t previous_workspace_used = session->workspace_used_bytes;
    if (session->workspace != NULL && previous_workspace_used <= session->workspace_bytes)
    {
        memset(session->workspace, 0, previous_workspace_used);
    }
    memset(session->blobs, 0, sizeof(session->blobs));
    session->blob_count = 0u;
    session->current_blob_index = 0u;
    session->scratch_bytes = 0u;
    session->scratch_offset = 0u;
    session->workspace_used_bytes = 0u;
    session->output_capacity_bytes = 0u;
    session->output_workspace_offset = 0u;
    session->output_length = 0u;
    session->output_stream_offset = 0u;
    session->output_stream_checksum = 0u;
    session->output_stream_active = 0u;
    session->last_kernel_status = ARM_CMSIS_NN_SUCCESS;
    /* Zero every per-case scalar param field (stride_h..adj_y, contiguous in the struct --
     * see benchmark_server_session.h) in one shot. Individually resetting only a handful of
     * these fields left the rest (e.g. output_h, float_activation_min_bits/max_bits) stale
     * across cases within the same batch/session, so a later case that doesn't retransmit a
     * given scalar (because the bridge omits it when it equals its own default, e.g.
     * BatchMatMul's output_h) would silently reuse a previous case's leaked value. */
    memset(&session->stride_h, 0, offsetof(hct_server_session_t, blob_count) - offsetof(hct_server_session_t, stride_h));
}

static hctp_status_t parse_scalar(hct_server_session_t *session, const char *name, int32_t value)
{
    if (strcmp(name, "stride_h") == 0) session->stride_h = value;
    else if (strcmp(name, "stride_w") == 0) session->stride_w = value;
    else if (strcmp(name, "padding") == 0) session->padding = value;
    else if (strcmp(name, "pad_h") == 0) session->pad_h = value;
    else if (strcmp(name, "pad_w") == 0) session->pad_w = value;
    else if (strcmp(name, "pad_offset_h") == 0) session->pad_offset_h = value;
    else if (strcmp(name, "pad_offset_w") == 0) session->pad_offset_w = value;
    else if (strcmp(name, "output_n") == 0) session->output_n = value;
    else if (strcmp(name, "output_h") == 0) session->output_h = value;
    else if (strcmp(name, "output_w") == 0) session->output_w = value;
    else if (strcmp(name, "output_c") == 0) session->output_c = value;
    else if (strcmp(name, "dilation_h") == 0) session->dilation_h = value;
    else if (strcmp(name, "dilation_w") == 0) session->dilation_w = value;
    else if (strcmp(name, "input_offset") == 0) session->input_offset = value;
    else if (strcmp(name, "output_offset") == 0) session->output_offset = value;
    else if (strcmp(name, "output_capacity_bytes") == 0)
    {
        if (value < 0) return HCTP_STATUS_INVALID_ARGUMENT;
        session->output_capacity_bytes = (uint32_t)value;
    }
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
    else if (strcmp(name, "float_activation_min_bits") == 0) session->float_activation_min_bits = value;
    else if (strcmp(name, "float_activation_max_bits") == 0) session->float_activation_max_bits = value;
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
    else if (strcmp(name, "adj_x") == 0) session->adj_x = value;
    else if (strcmp(name, "adj_y") == 0) session->adj_y = value;
    else if (strcmp(name, "weight_format_is_packed") == 0) session->weight_format_is_packed = value;
    else if (strcmp(name, "filter_offset") == 0) session->filter_offset = value;
    else if (strcmp(name, "axis_n") == 0) session->axis_n = value;
    else if (strcmp(name, "axis_h") == 0) session->axis_h = value;
    else if (strcmp(name, "axis_w") == 0) session->axis_w = value;
    else if (strcmp(name, "axis_c") == 0) session->axis_c = value;
    else if (strcmp(name, "axis") == 0) session->axis = value;
    else if (strcmp(name, "needs_rescale") == 0) session->needs_rescale = value;
    else if (strcmp(name, "null_arg_mask") == 0) session->null_arg_mask = value;
    /* An unrecognized scalar name (typo'd, renamed, or a new field added to a
     * builder's manifest without a matching entry here) used to silently no-op,
     * leaving the corresponding session field at its zeroed default instead of
     * erroring -- see test_perf_stream_adapter_codegen.py for the (currently
     * partial) build-time cross-check this complements at run time. */
    else return HCTP_STATUS_INVALID_ARGUMENT;
    return HCTP_STATUS_OK;
}

static hctp_status_t allocate_blob(hct_server_session_t *session, hct_server_blob_t *blob)
{
    uint32_t aligned;
    uint32_t end;
    if (!hct_checked_aligned_range(session->workspace_used_bytes,
                               blob->alignment,
                               blob->byte_length,
                               session->workspace_bytes,
                               &aligned,
                               &end))
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    blob->arena_offset = aligned;
    blob->bytes_received = 0u;
    session->workspace_used_bytes = end;
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
    if (session->output_length > session->output_capacity_bytes)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (session->expected_kernel_id == HCT_KERNEL_ID_ABS_F32)
    {
#ifndef HCT_HOST_ABS_ONLY
        return arm_nn_abs_f32((const float *)blob_ptr(session, input),
                              (float *)hct_output_ptr(session),
                              session->block_size);
#else
        return ARM_CMSIS_NN_ARG_ERROR;
#endif
    }
    if (session->expected_kernel_id == HCT_KERNEL_ID_ABS_F16)
    {
#ifndef HCT_HOST_ABS_ONLY
        return arm_nn_abs_f16((const float16_t *)blob_ptr(session, input),
                              (float16_t *)hct_output_ptr(session),
                              session->block_size);
#else
        return ARM_CMSIS_NN_ARG_ERROR;
#endif
    }
    if (session->expected_kernel_id == HCT_KERNEL_ID_ABS_S16)
    {
#ifdef HCT_HOST_ABS_ONLY
        return ARM_CMSIS_NN_ARG_ERROR;
#else
        return arm_abs_s16((const int16_t *)blob_ptr(session, input),
                           session->input_offset,
                           (int16_t *)hct_output_ptr(session),
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
        request.output = (int8_t *)hct_output_ptr(session);
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

static arm_cmsis_nn_status run_kernel_once(hct_server_session_t *session)
{
    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_ABS_S8:
        case HCT_KERNEL_ID_ABS_S16:
        case HCT_KERNEL_ID_ABS_F32:
        case HCT_KERNEL_ID_ABS_F16:
            return run_abs_once(session);
        default:
#ifndef HCT_HOST_ABS_ONLY
            return hct_run_adapter_once(session);
#else
            return ARM_CMSIS_NN_ARG_ERROR;
#endif
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
            arm_cmsis_nn_status status = run_kernel_once(session);
            session->last_kernel_status = status;
            if (kernel_status_is_fatal(session, status))
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

/* SESSION_PLAN: u16 case_count, u8 transfer_mode, u16 warmups, u16 samples,
 * u32 iterations_per_sample, u32 min_cycles, u32 max_iterations, u8 pass_count,
 * per pass (text name, u8 chained, u8 counter_count, u16 event_id[counter_count]),
 * then per case (text case_id, u32 kernel_id). Event ids are not validated against
 * a list -- whatever the host asks for is programmed and reported -- but a pass that
 * cannot fit the PMU (too many counters, or more chained slots than the core has) is
 * rejected here rather than silently truncated at RUN_PERFORMANCE time. */
static hctp_status_t parse_pmu_passes(hct_server_session_t *session, hct_cursor_t *cursor)
{
    const uint8_t slots_available = hct_benchmark_server_pmu_counter_slots();
    uint32_t pass_index;

    session->pass_count = cursor_u8(cursor);
    if (cursor->overrun)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    if (session->pass_count > HCT_SERVER_MAX_PASSES)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    for (pass_index = 0u; pass_index < session->pass_count; ++pass_index)
    {
        hct_pmu_pass_t *pass = &session->passes[pass_index];
        uint32_t counter_index;
        uint32_t slots_needed;
        if (!cursor_text(cursor, pass->name, sizeof(pass->name)))
        {
            return HCTP_STATUS_TRUNCATED_FRAME;
        }
        pass->chained = cursor_u8(cursor) ? 1u : 0u;
        pass->count = cursor_u8(cursor);
        if (cursor->overrun)
        {
            return HCTP_STATUS_TRUNCATED_FRAME;
        }
        if (pass->count > HCT_SERVER_MAX_COUNTERS_PER_PASS)
        {
            return HCTP_STATUS_INVALID_ARGUMENT;
        }
        for (counter_index = 0u; counter_index < pass->count; ++counter_index)
        {
            pass->event_ids[counter_index] = cursor_u16(cursor);
        }
        if (cursor->overrun)
        {
            return HCTP_STATUS_TRUNCATED_FRAME;
        }
        slots_needed = pass->chained ? (2u * pass->count) : pass->count;
        if (slots_available > 0u && slots_needed > slots_available)
        {
            return HCTP_STATUS_INVALID_ARGUMENT;
        }
    }
    if (session->pass_count == 0u)
    {
        /* No PMU passes requested: still time the case once (DWT + CCNTR only). */
        strcpy(session->passes[0].name, "cpu_0");
        session->passes[0].chained = 1u;
        session->passes[0].count = 0u;
        session->pass_count = 1u;
    }
    return HCTP_STATUS_OK;
}

static hctp_status_t handle_session_plan(hct_server_session_t *session, const uint8_t *payload, size_t payload_length)
{
    /* F006: every field below is read through the bounded cursor API, which checks
     * remaining capacity before each access/advance -- a truncated SESSION_PLAN (short
     * header, short pass/case-id text, or a plan cut off mid kernel-id list) is
     * caught by the cursor.overrun checks instead of risking an out-of-bounds read. */
    hct_cursor_t cursor;
    uint16_t case_index;
    hctp_status_t status;

    cursor_init(&cursor, payload, payload_length);

    session->planned_case_count = cursor_u16(&cursor);
    (void)cursor_u8(&cursor);
    session->planned_warmups = cursor_u16(&cursor);
    session->planned_samples = cursor_u16(&cursor);
    session->planned_iterations = cursor_u32(&cursor);
    session->min_cycles = cursor_u32(&cursor);
    session->max_iterations = cursor_u32(&cursor);

    if (cursor.overrun)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    if (session->planned_case_count == 0u || session->planned_case_count > HCT_SERVER_MAX_CASES)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }

    status = parse_pmu_passes(session, &cursor);
    if (status != HCTP_STATUS_OK)
    {
        return status;
    }

    for (case_index = 0u; case_index < session->planned_case_count; ++case_index)
    {
        if (!cursor_text(&cursor, session->planned_case_ids[case_index], sizeof(session->planned_case_ids[case_index])))
        {
            return HCTP_STATUS_TRUNCATED_FRAME;
        }
        session->planned_kernel_ids[case_index] = cursor_u32(&cursor);
    }
    if (cursor.overrun)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }

    strcpy(session->current_case_id, session->planned_case_ids[0]);
    session->expected_kernel_id = session->planned_kernel_ids[0];
    session->state = HCT_SERVER_STATE_WAIT_CASE_META;
    return queue_request_case(session);
}

static hctp_status_t handle_case_meta(hct_server_session_t *session, const uint8_t *payload, size_t payload_length)
{
    /* F006: converted to the bounded cursor API -- see its definition above for the
     * overrun-latching rationale. Every field read here (case_id, kernel_id, comparison
     * config, scalar name/value pairs, and every blob's id/role/dtype/rank/dims/byte
     * length/alignment/crc/mutability) now goes through cursor_u8/u16/u32/i32/text
     * instead of the previous unchecked read_*() + manual has_capacity() calls. */
    hct_cursor_t cursor;
    uint8_t scalar_count;
    uint16_t blob_index;
    /* Must be large enough for the longest string this handler reads via
     * cursor_text(): the case_id (up to HCT_SERVER_MAX_CASE_ID, e.g. the 79-char
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

    cursor_init(&cursor, payload, payload_length);

    if (!cursor_text(&cursor, scratch, sizeof(scratch))) return HCTP_STATUS_TRUNCATED_FRAME;
    if (strcmp(scratch, session->current_case_id) != 0) return HCTP_STATUS_INVALID_ARGUMENT;
    if (cursor_u32(&cursor) != session->expected_kernel_id)
    {
        return cursor.overrun ? HCTP_STATUS_TRUNCATED_FRAME : HCTP_STATUS_INVALID_ARGUMENT;
    }

    (void)cursor_u16(&cursor);
    session->comparison_mode = cursor_u8(&cursor);
    session->tolerance = cursor_i32(&cursor);
    session->atol_q16 = cursor_u32(&cursor);
    session->rtol_q16 = cursor_u32(&cursor);
    scalar_count = cursor_u8(&cursor);
    if (cursor.overrun)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    while (scalar_count-- > 0u)
    {
        int32_t value;
        if (!cursor_text(&cursor, scratch, sizeof(scratch))) return HCTP_STATUS_TRUNCATED_FRAME;
        value = cursor_i32(&cursor);
        if (cursor.overrun) return HCTP_STATUS_TRUNCATED_FRAME;
        status = parse_scalar(session, scratch, value);
        if (status != HCTP_STATUS_OK) return status;
    }

    session->blob_count = cursor_u16(&cursor);
    if (cursor.overrun)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    if (session->blob_count == 0u || session->blob_count > HCT_SERVER_MAX_BLOBS)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }

    for (blob_index = 0u; blob_index < session->blob_count; ++blob_index)
    {
        hct_server_blob_t *blob = &session->blobs[blob_index];
        uint8_t dim_index;
        blob->blob_id = cursor_u32(&cursor);
        if (!cursor_text(&cursor, scratch, sizeof(scratch))) return HCTP_STATUS_TRUNCATED_FRAME;
        blob->role = role_from_name(scratch);
        if (!cursor_text(&cursor, scratch, sizeof(scratch))) return HCTP_STATUS_TRUNCATED_FRAME;
        blob->dtype = dtype_from_name(scratch);
        blob->rank = cursor_u8(&cursor);
        for (dim_index = 0u; dim_index < 6u; ++dim_index)
        {
            blob->dimensions[dim_index] = cursor_u32(&cursor);
        }
        blob->byte_length = cursor_u32(&cursor);
        blob->alignment = cursor_u32(&cursor);
        blob->crc32 = cursor_u32(&cursor);
        blob->mutable_data = cursor_u8(&cursor);
        if (cursor.overrun)
        {
            return HCTP_STATUS_TRUNCATED_FRAME;
        }
        if (blob->role == HCT_BLOB_ROLE_UNKNOWN || blob->dtype == HCT_DTYPE_UNKNOWN || blob->alignment == 0u)
        {
            return HCTP_STATUS_INVALID_ARGUMENT;
        }
        status = allocate_blob(session, blob);
        if (status != HCTP_STATUS_OK) return status;
    }

    session->scratch_bytes = cursor_u32(&cursor);
    if (cursor.overrun)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    {
        uint32_t end_offset;
        if (session->scratch_bytes > 0u &&
            !hct_checked_aligned_range(session->workspace_used_bytes,
                                   16u, session->scratch_bytes, session->workspace_bytes,
                                   &session->scratch_offset, &end_offset))
        {
            return HCTP_STATUS_INVALID_ARGUMENT;
        }
        if (session->scratch_bytes > 0u)
        {
            session->workspace_used_bytes = end_offset;
        }
        if (session->output_capacity_bytes > 0u &&
            !hct_checked_aligned_range(session->workspace_used_bytes,
                                   16u, session->output_capacity_bytes, session->workspace_bytes,
                                   &session->output_workspace_offset, &end_offset))
        {
            return HCTP_STATUS_INVALID_ARGUMENT;
        }
        if (session->output_capacity_bytes > 0u)
        {
            session->workspace_used_bytes = end_offset;
        }
    }

    session->current_blob_index = 0u;
    session->state = HCT_SERVER_STATE_WAIT_BLOB_CHUNK;
    return queue_request_blob(session);
}

static hctp_status_t handle_blob_chunk(hct_server_session_t *session, const uint8_t *payload, size_t payload_length)
{
    /* F006: converted to the bounded cursor API. */
    hct_cursor_t cursor;
    uint32_t blob_id;
    uint32_t chunk_offset;
    uint32_t chunk_length;
    hct_server_blob_t *blob;

    cursor_init(&cursor, payload, payload_length);
    blob_id = cursor_u32(&cursor);
    chunk_offset = cursor_u32(&cursor);
    chunk_length = cursor_u32(&cursor);
    if (cursor.overrun)
    {
        return HCTP_STATUS_TRUNCATED_FRAME;
    }
    blob = &session->blobs[session->current_blob_index];

    if (blob_id != blob->blob_id) return HCTP_STATUS_INVALID_ARGUMENT;
    if (chunk_offset != blob->bytes_received) return HCTP_STATUS_INVALID_ARGUMENT;
    if (!cursor_require(&cursor, chunk_length)) return HCTP_STATUS_TRUNCATED_FRAME;
    if (chunk_offset + chunk_length > blob->byte_length) return HCTP_STATUS_INVALID_ARGUMENT;
    if ((uint64_t)blob->arena_offset + chunk_offset + chunk_length > session->workspace_bytes) return HCTP_STATUS_INVALID_ARGUMENT;
    if ((blob->alignment > 1u) && ((chunk_offset % blob->alignment) != 0u)) return HCTP_STATUS_INVALID_ARGUMENT;

    memcpy(blob_ptr(session, blob) + chunk_offset, payload + cursor.offset, chunk_length);
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
    arm_cmsis_nn_status status = run_kernel_once(session);
    session->last_kernel_status = status;
    if (kernel_status_is_fatal(session, status))
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    if (expects_exact_status(session))
    {
        session->output_length = 0u;
    }
    else if (status == ARM_CMSIS_NN_SUCCESS && session->output_length != session->output_capacity_bytes)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    return queue_correctness_output(session);
}

static hctp_status_t handle_run_performance(hct_server_session_t *session)
{
    uint32_t pass_index;
    uint32_t sample_index;
    uint32_t warmup;
    const uint32_t iterations = resolve_iterations(session);

    for (pass_index = 0u; pass_index < session->pass_count; ++pass_index)
    {
        const hct_pmu_pass_t *pass = &session->passes[pass_index];
        const uint32_t slot_mask = pmu_pass_slot_mask(pass);
        enable_dwt();
        pmu_pass_program(pass);
        for (warmup = 0u; warmup < session->planned_warmups; ++warmup)
        {
            arm_cmsis_nn_status status = run_kernel_once(session);
            session->last_kernel_status = status;
            if (kernel_status_is_fatal(session, status))
            {
                return HCTP_STATUS_INVALID_ARGUMENT;
            }
        }
        for (sample_index = 0u; sample_index < session->planned_samples; ++sample_index)
        {
            hct_pmu_sample_t sample;
            uint32_t iter;
            uint32_t start;
            uint32_t end;
            pmu_sample_start(slot_mask);
            start = dwt_cycles();
            for (iter = 0u; iter < iterations; ++iter)
            {
                arm_cmsis_nn_status status = run_kernel_once(session);
                session->last_kernel_status = status;
                if (kernel_status_is_fatal(session, status))
                {
                    pmu_sample_stop(pass, slot_mask, 0u, &sample);
                    return HCTP_STATUS_INVALID_ARGUMENT;
                }
            }
            end = dwt_cycles();
            pmu_sample_stop(pass, slot_mask, end - start, &sample);
            if (queue_sample_result(session, (uint16_t)sample_index, iterations, (uint64_t)(end - start), pass, &sample) != HCTP_STATUS_OK)
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
                             void *workspace,
                             uint32_t workspace_bytes)
{
    size_t frame_length = 0u;
    memset(session, 0, sizeof(*session));
    session->session_id = session_id;
    session->max_frame_payload = max_frame_payload;
    session->workspace = (uint8_t *)workspace;
    session->workspace_bytes = workspace_bytes;
    session->runtime_arena_capacity = workspace_bytes;
    session->state = HCT_SERVER_STATE_WAIT_TARGET_INFO_ACK;
    hct_build_target_info_frame(session_id,
                                session->next_outgoing_sequence++,
                                max_frame_payload,
                                session->runtime_arena_capacity,
                                HCT_SERVER_MAX_RX_PAYLOAD_BYTES,
                                HCT_SERVER_MAX_CASES,
                                HCT_SERVER_MAX_PASSES,
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
        case HCTP_MSG_TARGET_INFO_ACK:
            if (session->state != HCT_SERVER_STATE_WAIT_TARGET_INFO_ACK)
            {
                queue_error_frame(session, frame.header.message_type, HCTP_STATUS_INVALID_ARGUMENT);
                return HCTP_STATUS_INVALID_ARGUMENT;
            }
            session->state = HCT_SERVER_STATE_WAIT_PLAN;
            {
                /* Emit the full catalog as one or more paginated KERNEL_CATALOG frames;
                 * loop until the chunk builder reports is_final so every entry reaches
                 * the host regardless of how many chunks that takes. */
                size_t chunk_start_index = 0u;
                bool chunk_is_final = false;
                do
                {
                    size_t chunk_next_index = 0u;
                    if (hct_build_catalog_frame_chunk(session->session_id,
                                                      session->next_outgoing_sequence++,
                                                      chunk_start_index,
                                                      frame_buffer,
                                                      sizeof(frame_buffer),
                                                      &catalog_frame_length,
                                                      &chunk_next_index,
                                                      &chunk_is_final) != HCTP_STATUS_OK)
                    {
                        session->state = HCT_SERVER_STATE_ERROR;
                        queue_error_frame(session, frame.header.message_type, HCTP_STATUS_TRUNCATED_FRAME);
                        return HCTP_STATUS_TRUNCATED_FRAME;
                    }
                    status = append_frame(session, frame_buffer, catalog_frame_length);
                    if (status != HCTP_STATUS_OK)
                    {
                        session->state = HCT_SERVER_STATE_ERROR;
                        queue_error_frame(session, frame.header.message_type, status);
                        return status;
                    }
                    chunk_start_index = chunk_next_index;
                } while (!chunk_is_final);
            }
            return HCTP_STATUS_OK;
        case HCTP_MSG_SESSION_PLAN:
            if (session->state != HCT_SERVER_STATE_WAIT_PLAN)
            {
                queue_error_frame(session, frame.header.message_type, HCTP_STATUS_INVALID_ARGUMENT);
                return HCTP_STATUS_INVALID_ARGUMENT;
            }
            status = handle_session_plan(session, frame.payload, frame.header.payload_length);
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
    if (session->outbox_length == 0u && session->output_stream_active != 0u)
    {
        if (pump_correctness_output(session) != HCTP_STATUS_OK)
        {
            session->state = HCT_SERVER_STATE_ERROR;
            return 0u;
        }
    }
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

    if (session->outbox_length == 0u && session->output_stream_active != 0u)
    {
        if (pump_correctness_output(session) != HCTP_STATUS_OK)
        {
            session->state = HCT_SERVER_STATE_ERROR;
            return 0u;
        }
    }

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
