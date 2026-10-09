/* Host harness for MRAM weights placement. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arm_nn_types.h"
#include "benchmark_server_adapters.h"
#include "benchmark_server_catalog.h"
#include "benchmark_server_mram.h"
#include "benchmark_server_session.h"
#include "hct_mram_stub.h"

#define POOL_BYTES (512u * 1024u)
#define BIG_WEIGHTS (POOL_BYTES + 16u)

typedef struct
{
    const char *role;
    const uint8_t *data;
    uint32_t length;
    uint32_t alignment;
} blob_spec_t;

static const int8_t kInput[12] = {-12, -1, 0, 7, -99, 5, -8, 3, -4, 11, -2, 100};
static const uint8_t kWeights[13] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
static const uint8_t kBias[8] = {0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7};
static const blob_spec_t kCase[3] = {
    {"input_0", (const uint8_t *)kInput, sizeof(kInput), 1u},
    {"weights", kWeights, sizeof(kWeights), 1u},
    {"bias", kBias, sizeof(kBias), 4u},
};

static uint8_t workspace[32768u];
#if defined(HCT_PLACEMENT_MRAM)
static uint8_t probe_workspace[sizeof(workspace)];
static uint8_t big_workspace[600u * 1024u];
static uint8_t big_weights[BIG_WEIGHTS];
#endif
static uint8_t frame[2048];
static uint8_t payload[1024];
static uint32_t next_sequence;
/* Store digests sent with CASE_META, if set. */
static const uint64_t *meta_keys;
/* REQUEST_BLOBs that start a blob. */
static uint32_t blob_requests;

static void put_u8(size_t *at, uint8_t value) { payload[(*at)++] = value; }

static void put_u16(size_t *at, uint16_t value)
{
    put_u8(at, (uint8_t)value);
    put_u8(at, (uint8_t)(value >> 8));
}

static void put_u32(size_t *at, uint32_t value)
{
    put_u16(at, (uint16_t)value);
    put_u16(at, (uint16_t)(value >> 16));
}

static void put_text(size_t *at, const char *text)
{
    const uint16_t length = (uint16_t)strlen(text);
    put_u16(at, length);
    memcpy(&payload[*at], text, length);
    *at += length;
}

static uint32_t get_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static hctp_status_t send(hct_server_session_t *s, uint16_t type, size_t length)
{
    hctp_frame_header_t header = {
        .magic = HCTP_MAGIC_U32,
        .protocol_version = HCTP_SUPPORTED_VERSION,
        .message_type = type,
        .flags = HCTP_FLAG_NONE,
        .session_id = s->session_id,
        .sequence_id = next_sequence++,
        .payload_length = (uint32_t)length,
        .payload_crc32 = hctp_crc32(payload, length),
        .header_crc32 = 0u,
    };
    hctp_encode_header(frame, &header);
    memcpy(frame + HCTP_HEADER_SIZE, payload, length);
    return hct_server_session_accept_frame(s, frame, HCTP_HEADER_SIZE + length);
}

/* Drop queued frames; return the last type. */
static uint16_t drain(hct_server_session_t *s)
{
    uint8_t out[2048];
    hctp_frame_view_t view;
    uint16_t last = 0u;
    size_t length;
    while ((length = hct_server_session_take_next_frame(s, out, sizeof(out))) != 0u)
    {
        if (hctp_decode_frame(out, length, HCTP_DEFAULT_MAX_PAYLOAD, &view) != HCTP_STATUS_OK) return 0xFFFFu;
        last = view.header.message_type;
    }
    return last;
}

/* Init, handshake, one-case plan. */
static int open_session(hct_server_session_t *s, uint8_t *ws, uint32_t ws_bytes, const char *case_id)
{
    static const hct_boot_info_t boot = {0, 96000000u, 0x03040000u, 0x00040000u};
    size_t at = 0u;
    next_sequence = 0u;
    hct_server_session_init(s, 0xC0DE1234u, 256u, ws, ws_bytes, &boot);
    if (drain(s) != HCTP_MSG_TARGET_INFO) return 1;
    if (send(s, HCTP_MSG_TARGET_INFO_ACK, 0u) != HCTP_STATUS_OK) return 2;
    if (drain(s) != HCTP_MSG_KERNEL_CATALOG) return 3;
    put_u16(&at, 1u);
    put_u8(&at, 1u);
    put_u16(&at, 1u);
    put_u16(&at, 1u);
    put_u32(&at, 1u);
    put_u32(&at, 1u);
    put_u32(&at, 1u);
    put_u8(&at, 0u);
    put_text(&at, case_id);
    put_u32(&at, HCT_KERNEL_ID_ABS_S8);
    if (send(s, HCTP_MSG_SESSION_PLAN, at) != HCTP_STATUS_OK) return 4;
    return drain(s) == HCTP_MSG_REQUEST_CASE ? 0 : 5;
}

/* CASE_META for an abs_s8 case. */
static hctp_status_t send_meta(hct_server_session_t *s, const char *case_id, const blob_spec_t *blobs, uint16_t count)
{
    size_t at = 0u;
    put_text(&at, case_id);
    put_u32(&at, HCT_KERNEL_ID_ABS_S8);
    put_u16(&at, 1u);
    put_u8(&at, 1u);
    put_u32(&at, 0u);
    put_u32(&at, 0u);
    put_u32(&at, 0u);
    put_u8(&at, 1u);
    put_text(&at, "output_capacity_bytes");
    put_u32(&at, blobs[0].length);
    put_u16(&at, count);
    for (uint16_t index = 0u; index < count; ++index)
    {
        put_u32(&at, index + 1u);
        put_text(&at, blobs[index].role);
        put_text(&at, "S8");
        put_u8(&at, 1u);
        put_u32(&at, blobs[index].length);
        for (int dim = 1; dim < 6; ++dim) put_u32(&at, 0u);
        put_u32(&at, blobs[index].length);
        put_u32(&at, blobs[index].alignment);
        put_u32(&at, hctp_crc32(blobs[index].data, blobs[index].length));
        put_u8(&at, 0u);
    }
    put_u32(&at, 0u);
    if (meta_keys != NULL)
    {
        put_u8(&at, 1u);
        for (uint16_t index = 0u; index < count; ++index)
        {
            put_u32(&at, (uint32_t)meta_keys[index]);
            put_u32(&at, (uint32_t)(meta_keys[index] >> 32));
        }
    }
    return send(s, HCTP_MSG_CASE_META, at);
}

/* Answer blob requests; first error wins. */
static hctp_status_t stream(hct_server_session_t *s, const blob_spec_t *blobs)
{
    uint8_t out[2048];
    hctp_frame_view_t view;
    while (s->state == HCT_SERVER_STATE_WAIT_BLOB_CHUNK)
    {
        const size_t length = hct_server_session_take_next_frame(s, out, sizeof(out));
        uint32_t id;
        uint32_t offset;
        uint32_t count;
        size_t at = 0u;
        hctp_status_t status;
        if (hctp_decode_frame(out, length, HCTP_DEFAULT_MAX_PAYLOAD, &view) != HCTP_STATUS_OK) return HCTP_STATUS_TRUNCATED_FRAME;
        if (view.header.message_type != HCTP_MSG_REQUEST_BLOB) return HCTP_STATUS_INVALID_ARGUMENT;
        id = get_u32(&view.payload[0]);
        offset = get_u32(&view.payload[4]);
        count = (uint32_t)view.payload[8] | ((uint32_t)view.payload[9] << 8);
        blob_requests += offset == 0u ? 1u : 0u;
        if (count > blobs[id - 1u].length - offset) count = blobs[id - 1u].length - offset;
        put_u32(&at, id);
        put_u32(&at, offset);
        put_u32(&at, count);
        memcpy(&payload[at], blobs[id - 1u].data + offset, count);
        status = send(s, HCTP_MSG_BLOB_CHUNK, at + count);
        if (status != HCTP_STATUS_OK) return status;
    }
    return HCTP_STATUS_OK;
}

#if defined(HCT_PLACEMENT_MRAM)
/* Stream a session copy; expect a status. */
static int probe_stream(const hct_server_session_t *s, hctp_status_t expected)
{
    static hct_server_session_t probe;
    const uint32_t saved_sequence = next_sequence;
    const uint32_t calls = hct_mram_stub.program_calls;
    hctp_status_t status;
    memcpy(&probe, s, sizeof(probe));
    memcpy(probe_workspace, workspace, sizeof(workspace));
    probe.workspace = probe_workspace;
    status = stream(&probe, kCase);
    next_sequence = saved_sequence;
    hct_window.cold_count = 0u;
    if (status != expected) return 1;
    return hct_mram_stub.program_calls == calls + 1u ? 0 : 2;
}

/* Placed copy: pool, line-aligned, exact. */
static int check_placed(const hct_server_blob_t *blob, const blob_spec_t *spec, uintptr_t base)
{
    const uintptr_t at = (uintptr_t)blob->placed;
    if (blob->placed == NULL || at < base || at + 16u > base + POOL_BYTES) return 1;
    if ((at % 32u) != 0u) return 2;
    return memcmp(blob->placed, spec->data, spec->length) == 0 ? 0 : 3;
}

#endif

/* Correctness, ack, performance; then drain. */
static int run_case(hct_server_session_t *s)
{
    const uint8_t *out;
    if (drain(s) != HCTP_MSG_CASE_READY) return 1;
    if (send(s, HCTP_MSG_RUN_CORRECTNESS, 0u) != HCTP_STATUS_OK) return 2;
    out = hct_output_ptr(s);
    for (size_t index = 0u; index < sizeof(kInput); ++index)
    {
        const int8_t want = kInput[index] < 0 ? (int8_t)-kInput[index] : kInput[index];
        if ((int8_t)out[index] != want) return 3;
    }
    drain(s);
    payload[0] = 1u;
    if (send(s, HCTP_MSG_CORRECTNESS_ACK, 1u) != HCTP_STATUS_OK) return 4;
    return send(s, HCTP_MSG_RUN_PERFORMANCE, 0u) == HCTP_STATUS_OK ? 0 : 5;
}

#if defined(HCT_PLACEMENT_MRAM)
static int test_place_and_reuse(uintptr_t base)
{
    static hct_server_session_t s;
    const uint8_t *first[2];
    uint32_t calls;
    uint32_t invalidations;
    int status = open_session(&s, workspace, sizeof(workspace), "mram_a");
    if (status != 0) return 100 + status;
    if (send_meta(&s, "mram_a", kCase, 3u) != HCTP_STATUS_OK) return 110;
    /* Weights and bias pad to 16-byte rows. */
    printf("workspace=%u\n", (unsigned)s.workspace_used_bytes);
    hct_mram_stub.corrupt_program = 1;
    if (probe_stream(&s, HCTP_STATUS_PAYLOAD_CRC_MISMATCH) != 0) return 111;
    hct_mram_stub.fail_program = 1;
    if (probe_stream(&s, HCTP_STATUS_INVALID_ARGUMENT) != 0) return 112;

    calls = hct_mram_stub.program_calls;
    invalidations = hct_mram_stub.invalidate_calls;
    if (stream(&s, kCase) != HCTP_STATUS_OK) return 113;
    if (hct_mram_stub.program_calls != calls + 2u || hct_mram_stub.invalidate_calls != invalidations + 2u) return 114;
    if (s.blobs[0].placed != NULL) return 115;
    if (check_placed(&s.blobs[1], &kCase[1], base) != 0 || check_placed(&s.blobs[2], &kCase[2], base) != 0) return 116;
    if (hct_window.cold_count != 2u || hct_window.cold_addr[0] != s.blobs[1].placed || hct_window.cold_bytes[0] != 16) return 117;
    first[0] = s.blobs[1].placed;
    first[1] = s.blobs[2].placed;

    /* Timed calls evict both cold ranges. */
    invalidations = hct_mram_stub.invalidate_calls;
    status = run_case(&s);
    if (status != 0) return 120 + status;
    if (hct_mram_stub.invalidate_calls <= invalidations || (hct_mram_stub.invalidate_calls - invalidations) % 2u != 0u) return 130;
    if (hct_mram_stub.last_invalidate != (const volatile void *)first[1]) return 131;
    if (drain(&s) != HCTP_MSG_SESSION_COMPLETE) return 132;

    /* Same bytes: same rows, no program. */
    status = open_session(&s, workspace, sizeof(workspace), "mram_b");
    if (status != 0) return 140 + status;
    if (send_meta(&s, "mram_b", kCase, 3u) != HCTP_STATUS_OK) return 150;
    calls = hct_mram_stub.program_calls;
    if (stream(&s, kCase) != HCTP_STATUS_OK) return 151;
    if (hct_mram_stub.program_calls != calls) return 152;
    if (s.blobs[1].placed != first[0] || s.blobs[2].placed != first[1]) return 153;
    status = run_case(&s);
    if (status != 0) return 160 + status;
    printf("placed reused\n");
    return drain(&s) == HCTP_MSG_SESSION_COMPLETE ? 0 : 170;
}

/* Stream rows; return calls and rows programmed. */
static int stream_rows(const blob_spec_t *blobs, uint32_t *calls, uint32_t *rows)
{
    static hct_server_session_t s;
    const uint32_t calls_before = hct_mram_stub.program_calls;
    const uint32_t rows_before = hct_mram_rows_programmed;
    if (open_session(&s, workspace, sizeof(workspace), "mram_rows") != 0) return 1;
    if (send_meta(&s, "mram_rows", blobs, 3u) != HCTP_STATUS_OK) return 2;
    if (stream(&s, blobs) != HCTP_STATUS_OK) return 3;
    if (memcmp(s.blobs[2].placed, blobs[2].data, blobs[2].length) != 0) return 4;
    *calls = hct_mram_stub.program_calls - calls_before;
    *rows = hct_mram_rows_programmed - rows_before;
    return 0;
}

/* Only changed rows get programmed. */
static int test_row_skip(void)
{
    static uint8_t bias[80];
    const blob_spec_t blobs[3] = {
        {"input_0", (const uint8_t *)kInput, sizeof(kInput), 1u},
        {"weights", kWeights, sizeof(kWeights), 1u},
        {"bias", bias, sizeof(bias), 4u},
    };
    uint32_t calls;
    uint32_t rows;
    /* Bias changes keep the weights-hashed offset. */
    for (uint32_t index = 0u; index < sizeof(bias); ++index) bias[index] = (uint8_t)(index * 3u + 1u);
    if (stream_rows(blobs, &calls, &rows) != 0) return 300;
    if (stream_rows(blobs, &calls, &rows) != 0 || calls != 0u || rows != 0u) return 301;
    bias[37] ^= 0x5Au;
    if (stream_rows(blobs, &calls, &rows) != 0 || calls != 1u || rows != 1u) return 302;
    if (hct_mram_stub.last_invalidate_bytes != 16) return 303;
    /* Rows 0, 3, 4: two runs, three rows. */
    bias[0] ^= 1u;
    bias[50] ^= 1u;
    bias[79] ^= 1u;
    if (stream_rows(blobs, &calls, &rows) != 0 || calls != 2u || rows != 3u) return 304;
    printf("rows skipped\n");
    return 0;
}

static int test_pool_limits(uintptr_t base)
{
    static hct_server_session_t s;
    const blob_spec_t big[2] = {
        {"input_0", (const uint8_t *)kInput, sizeof(kInput), 1u},
        {"weights", big_weights, sizeof(big_weights), 1u},
    };
    uint32_t calls = hct_mram_stub.program_calls;
    int status;
    /* A blob larger than the pool. */
    for (uint32_t index = 0u; index < sizeof(big_weights); ++index) big_weights[index] = (uint8_t)(index * 7u);
    status = open_session(&s, big_workspace, sizeof(big_workspace), "mram_big");
    if (status != 0) return 200 + status;
    if (send_meta(&s, "mram_big", big, 2u) != HCTP_STATUS_OK) return 210;
    if (stream(&s, big) != HCTP_STATUS_INVALID_ARGUMENT) return 211;
    if (hct_mram_stub.program_calls != calls || hct_window.cold_count != 0u) return 212;

    /* Pool past MRAM end: refuse. */
    hct_stub_mram_end = base + POOL_BYTES - 1u;
    status = open_session(&s, workspace, sizeof(workspace), "mram_overlap");
    if (status != 0) return 220 + status;
    if (send_meta(&s, "mram_overlap", kCase, 3u) != HCTP_STATUS_OK) return 221;
    if (stream(&s, kCase) != HCTP_STATUS_INVALID_ARGUMENT) return 222;
    if (hct_mram_stub.program_calls != calls) return 223;
    hct_stub_mram_end = (uintptr_t)hct_stub_mram + HCT_STUB_MRAM_BYTES;
    printf("pool limits refused\n");
    return 0;
}

#endif

#if HCT_BLOB_STORE_BYTES > 0
static const uint64_t kKeys[3] = {0x1111111111111111ull, 0x2222222222222222ull, 0x3333333333333333ull};

static uint8_t *store_base(void)
{
    return (uint8_t *)(hct_stub_mram_end - HCT_BLOB_STORE_BYTES);
}

/* One keyed case; count requests and programs. */
static int store_case(const char *case_id, const uint64_t *keys, const blob_spec_t *blobs, uint32_t *requests, uint32_t *programs)
{
    static hct_server_session_t s;
    const uint32_t calls = hct_mram_stub.program_calls;
    int status = open_session(&s, workspace, sizeof(workspace), case_id);
    if (status != 0) return 1;
    meta_keys = keys;
    blob_requests = 0u;
    if (send_meta(&s, case_id, blobs, 3u) != HCTP_STATUS_OK) return 2;
    meta_keys = NULL;
    if (stream(&s, blobs) != HCTP_STATUS_OK) return 3;
    *requests = blob_requests;
    *programs = hct_mram_stub.program_calls - calls;
    /* Hits must equal what streaming sends. */
    for (uint16_t index = 0u; index < 3u; ++index)
    {
        const hct_server_blob_t *blob = &s.blobs[index];
        if (memcmp(blob->placed != NULL ? blob->placed : &s.workspace[blob->arena_offset], blobs[index].data, blobs[index].length) != 0) return 4;
    }
    status = run_case(&s);
    if (status != 0) return 10 + status;
    return drain(&s) == HCTP_MSG_SESSION_COMPLETE ? 0 : 5;
}

static int test_store(void)
{
    /* Prototype store bytes at the base. */
    static const uint32_t kStale[4] = {0x53544348u, 0x12345678u, 12u, 0xA5A5A5A5u};
    uint64_t other[3] = {kKeys[0], kKeys[1], kKeys[2]};
    uint32_t requests;
    uint32_t programs;
    uint32_t generation;
    memcpy(store_base(), kStale, sizeof(kStale));
    if ((hct_benchmark_server_capability_flags() & HCT_CAP_BLOB_STORE) == 0u) return 400;

    /* Cold: stream all, save all. */
    if (store_case("store_cold", kKeys, kCase, &requests, &programs) != 0 || requests != 3u || programs == 0u) return 401;
    if (memcmp(store_base(), "HBS1", 4u) != 0) return 402;
    generation = *(const uint32_t *)(store_base() + 16u);

    /* Warm: nothing streams or programs. */
    if (store_case("store_warm", kKeys, kCase, &requests, &programs) != 0 || requests != 0u || programs != 0u) return 403;

    /* No keys: stream all, save none. */
    if (store_case("store_unkeyed", NULL, kCase, &requests, &programs) != 0 || requests != 3u || programs != 0u) return 404;

    /* Same CRC and length, new digest: miss. */
    other[0] ^= 1u;
    if (store_case("store_digest", other, kCase, &requests, &programs) != 0 || requests != 1u || programs == 0u) return 405;

    /* Corrupt copy fails its CRC: stream it. */
    store_base()[64] ^= 0x40u;
    if (store_case("store_corrupt", kKeys, kCase, &requests, &programs) != 0 || requests != 1u || programs == 0u) return 406;
    if (store_case("store_healed", kKeys, kCase, &requests, &programs) != 0 || requests != 0u) return 407;
    if (*(const uint32_t *)(store_base() + 16u) != generation) return 408;
    printf("store hits\n");
    return 0;
}

/* Full store wipes; big blobs skip it. */
static int test_store_wipe(void)
{
    static uint8_t big[HCT_BLOB_STORE_BYTES / 3u];
    const blob_spec_t blobs[3] = {
        {"input_0", (const uint8_t *)kInput, sizeof(kInput), 1u},
        {"weights", big, sizeof(big), 1u},
        {"bias", kBias, sizeof(kBias), 4u},
    };
    const uint32_t generation = *(const uint32_t *)(store_base() + 16u);
    uint64_t keys[3] = {0x4444u, 0u, 0x5555u};
    uint32_t requests;
    uint32_t programs;
    for (uint32_t round = 0u; round < 4u; ++round)
    {
        for (uint32_t index = 0u; index < sizeof(big); ++index) big[index] = (uint8_t)(index * 13u + round);
        keys[1] = 0x6666u + round;
        if (store_case("store_fill", keys, blobs, &requests, &programs) != 0 || requests == 0u) return 500 + (int)round;
    }
    if (*(const uint32_t *)(store_base() + 16u) == generation) return 510;
    /* The newest blob survives the wipe. */
    if (store_case("store_fill", keys, blobs, &requests, &programs) != 0 || requests != 0u) return 511;
    printf("store wiped\n");
    return 0;
}

/* Exact fill, then one more: wipe. */
static int test_store_exact_fill(void)
{
    /* Head, then 3 entries end at the top. */
    static uint8_t mid[HCT_BLOB_STORE_BYTES - 6u * 32u];
    const blob_spec_t blobs[3] = {
        {"input_0", (const uint8_t *)kInput, sizeof(kInput), 1u},
        {"weights", mid, sizeof(mid), 1u},
        {"bias", kBias, sizeof(kBias), 4u},
    };
    uint64_t keys[3] = {0x7777u, 0x8888u, 0x9999u};
    uint32_t requests;
    uint32_t programs;
    for (uint32_t index = 0u; index < sizeof(mid); ++index) mid[index] = (uint8_t)(index * 5u);
    /* Foreign head: start an empty store. */
    store_base()[0] ^= 0xFFu;
    if (store_case("store_exact", keys, blobs, &requests, &programs) != 0 || requests != 3u) return 700;
    /* New input wipes; all three restream. */
    keys[0] = 0xAAAAu;
    if (store_case("store_exact", keys, blobs, &requests, &programs) != 0 || requests != 3u) return 701;
    if (store_case("store_exact", keys, blobs, &requests, &programs) != 0 || requests != 0u) return 702;
    /* Zeroed head twice: old entries stay dead. */
    memset(store_base(), 0, 32u);
    if (store_case("store_exact", keys, blobs, &requests, &programs) != 0 || requests != 3u) return 703;
    memset(store_base(), 0, 32u);
    if (store_case("store_exact", keys, blobs, &requests, &programs) != 0 || requests != 3u) return 704;
    printf("store exact fill\n");
    return 0;
}

/* Pool reaching the store: store unused. */
static int test_store_overlap(void)
{
    const uintptr_t saved = hct_stub_mram_end;
    uint32_t requests;
    uint32_t programs;
    hct_stub_mram_end = (((uintptr_t)hct_stub_mram + 0xFFFFu) & ~(uintptr_t)0xFFFFu) + POOL_BYTES + HCT_BLOB_STORE_BYTES - 32u;
    if (store_case("store_overlap", kKeys, kCase, &requests, &programs) != 0 || requests != 3u || programs != 0u) return 600;
    hct_stub_mram_end = saved;
    printf("store overlap refused\n");
    return 0;
}

#endif

int main(void)
{
    int status = 0;
    hct_stub_mram_end = (uintptr_t)hct_stub_mram + HCT_STUB_MRAM_BYTES;
#if defined(HCT_PLACEMENT_MRAM)
    {
        const uintptr_t base = ((uintptr_t)hct_stub_mram + 0xFFFFu) & ~(uintptr_t)0xFFFFu;
        if ((hct_benchmark_server_capability_flags() & HCT_CAP_WEIGHTS_MRAM) == 0u) return 10;
        status = test_place_and_reuse(base);
        if (status == 0) status = test_row_skip();
        if (status == 0) status = test_pool_limits(base);
    }
#endif
#if HCT_BLOB_STORE_BYTES > 0
    if (status == 0) status = test_store();
    if (status == 0) status = test_store_wipe();
    if (status == 0) status = test_store_exact_fill();
    if (status == 0) status = test_store_overlap();
#endif
    return status;
}
