#ifndef HCT_BENCHMARK_SERVER_SESSION_H
#define HCT_BENCHMARK_SERVER_SESSION_H

#include <stddef.h>
#include <stdint.h>

#include "hctp_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HCT_SERVER_MAX_CASE_ID 96u
#define HCT_SERVER_MAX_CASES 4u
#define HCT_SERVER_MAX_GROUPS 4u
#define HCT_SERVER_MAX_GROUP_NAME 16u
#define HCT_SERVER_MAX_BLOBS 8u
#define HCT_SERVER_MAX_INPUT_BYTES 4096u
#define HCT_SERVER_MAX_ARENA_BYTES 8192u
#define HCT_SERVER_MAX_OUTPUT_BYTES 4096u
#define HCT_SERVER_MAX_OUTBOX_BYTES 16384u
#define HCT_SERVER_BLOB_CHUNK_BYTES 64u

typedef enum
{
    HCT_SERVER_STATE_WAIT_HELLO_ACK = 0,
    HCT_SERVER_STATE_WAIT_PLAN = 1,
    HCT_SERVER_STATE_WAIT_CASE_META = 2,
    HCT_SERVER_STATE_WAIT_BLOB_CHUNK = 3,
    HCT_SERVER_STATE_WAIT_RUN_CORRECTNESS = 4,
    HCT_SERVER_STATE_WAIT_CORRECTNESS_ACK = 5,
    HCT_SERVER_STATE_WAIT_RUN_PERFORMANCE = 6,
    HCT_SERVER_STATE_COMPLETE = 7,
    HCT_SERVER_STATE_ERROR = 8
} hct_server_state_t;

typedef struct
{
    uint32_t blob_id;
    uint8_t role;
    uint8_t dtype;
    uint8_t rank;
    uint8_t mutable_data;
    uint32_t dimensions[6];
    uint32_t byte_length;
    uint32_t alignment;
    uint32_t crc32;
    uint32_t arena_offset;
    uint32_t bytes_received;
} hct_server_blob_t;

typedef struct
{
    uint32_t session_id;
    uint32_t max_frame_payload;
    uint32_t runtime_arena_capacity;
    uint32_t next_outgoing_sequence;
    hct_server_state_t state;
    uint16_t planned_case_count;
    uint16_t current_case_index;
    uint32_t planned_iterations;
    uint16_t planned_warmups;
    uint16_t planned_samples;
    uint32_t min_cycles;
    uint32_t max_iterations;
    uint8_t requested_group_count;
    char requested_groups[HCT_SERVER_MAX_GROUPS][HCT_SERVER_MAX_GROUP_NAME];
    char planned_case_ids[HCT_SERVER_MAX_CASES][HCT_SERVER_MAX_CASE_ID];
    uint32_t planned_kernel_ids[HCT_SERVER_MAX_CASES];
    uint32_t expected_kernel_id;
    char current_case_id[HCT_SERVER_MAX_CASE_ID];
    uint8_t comparison_mode;
    int32_t tolerance;
    uint32_t atol_q16;
    uint32_t rtol_q16;
    int32_t stride_h;
    int32_t stride_w;
    int32_t padding;
    /* Ground-truth output dims and "before" padding sent explicitly by the host
     * (see serialized_scalar_parameters in generated_test_bridge.py/case_bundle.py),
     * used directly instead of re-deriving them from the `padding` VALID/SAME flag
     * above, which can silently diverge from the real generator's padding/output-size
     * convention (asymmetric splits, rounding, etc.) for real generated test cases. */
    int32_t pad_h;
    int32_t pad_w;
    int32_t output_h;
    int32_t output_w;
    int32_t output_c;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t input_offset;
    int32_t output_offset;
    int32_t activation_min;
    int32_t activation_max;
    /* BasicMathFunctions elementwise-binary (Add/Sub) scalar params -- see parse_scalar()
     * and run_elementwise_binary_once() in benchmark_server_session.c. Reuses output_offset/
     * activation_min/activation_max above (same semantics as convolve's output clamp). */
    int32_t input1_offset;
    int32_t input1_mult;
    int32_t input1_shift;
    int32_t input2_offset;
    int32_t input2_mult;
    int32_t input2_shift;
    int32_t left_shift;
    int32_t out_mult;
    int32_t out_shift;
    /* ConvolutionFunctions DepthwiseConv scalar param -- see parse_scalar() and
     * run_depthwise_conv_once() in benchmark_server_session.c. Reuses stride_h/w,
     * pad_h/w, dilation_h/w, output_h/w/c, input_offset, output_offset,
     * activation_min/max above (same semantics as convolve's). */
    int32_t ch_mult;
    uint16_t blob_count;
    uint16_t current_blob_index;
    uint32_t scratch_bytes;
    uint32_t scratch_offset;
    uint32_t case_arena_used_bytes;
    uint32_t output_length;
    hct_server_blob_t blobs[HCT_SERVER_MAX_BLOBS];
    uint8_t case_arena[HCT_SERVER_MAX_ARENA_BYTES];
    uint8_t output_buffer[HCT_SERVER_MAX_OUTPUT_BYTES];
    uint8_t outbox[HCT_SERVER_MAX_OUTBOX_BYTES];
    size_t outbox_length;
} hct_server_session_t;

void hct_server_session_init(hct_server_session_t *session,
                             uint32_t session_id,
                             uint32_t max_frame_payload,
                             uint32_t runtime_arena_capacity);

hctp_status_t hct_server_session_accept_frame(hct_server_session_t *session,
                                              const uint8_t *frame_bytes,
                                              size_t frame_length);

size_t hct_server_session_take_outbound(hct_server_session_t *session,
                                        uint8_t *buffer,
                                        size_t capacity);

size_t hct_server_session_take_next_frame(hct_server_session_t *session,
                                          uint8_t *buffer,
                                          size_t capacity);

#ifdef __cplusplus
}
#endif

#endif
