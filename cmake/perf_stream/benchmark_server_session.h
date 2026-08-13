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
#define HCT_SERVER_MAX_ARENA_BYTES 49152u
#define HCT_SERVER_MAX_OUTPUT_BYTES 20480u
#define HCT_SERVER_MAX_OUTBOX_BYTES 32768u
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
    int32_t last_kernel_status;
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
    int32_t pad_offset_h;
    int32_t pad_offset_w;
    int32_t output_n;
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
    /* PoolingFunctions AvgPool/MaxPool scalar params (pool window size -- there is no
     * weights blob for pooling, so these can't be read off a blob's dimensions like
     * Convolve/DepthwiseConv's filter dims are). Reuses stride_h/w, pad_h/w,
     * activation_min/max, output_h/w/c above (same semantics). See parse_scalar() and
     * run_pooling_once() in benchmark_server_session.c. */
    int32_t pool_h;
    int32_t pool_w;
    /* ActivationFunctions scalar params -- see parse_scalar() and run_activation_once() in
     * benchmark_server_session.c. Relu/Relu6/Clamp/LeakyRelu/HardSwish* all reuse
     * input_offset/output_offset/out_mult/out_shift/activation_min/max above where semantics
     * match; these fields cover the remaining op-specific quantized params:
     *  - LeakyRelu: out_mult/out_shift above is the "identity" branch; out_mult_alpha/
     *    out_shift_alpha below is the "alpha" (negative-slope) branch.
     *  - HardSwishCompat: out_mult_fp/out_mult_exp (output branch) and relu_mult_fp/
     *    relu_mult_exp (relu branch) are its own Q15 mantissa/exponent pairs.
     *  - HardSwishPrecise: relu_q3/relu_q6/prescale are its own quantized breakpoints.
     *  - Logistic/Tanh (S16-only, no offsets at all): input_mult/input_left_shift. */
    int32_t out_mult_alpha;
    int32_t out_shift_alpha;
    int32_t out_mult_fp;
    int32_t out_mult_exp;
    int32_t relu_mult_fp;
    int32_t relu_mult_exp;
    int32_t relu_q3;
    int32_t relu_q6;
    int32_t prescale;
    int32_t input_mult;
    int32_t input_left_shift;
    /* PReLU/PReLUScalar scalar params -- see parse_scalar() and run_prelu_once() in
     * benchmark_server_session.c. Reuses input_offset/output_offset (own tensor's zero
     * points) and out_mult/out_shift (the "identity" branch) above, plus out_mult_alpha/
     * out_shift_alpha above (already added for LeakyRelu's alpha branch -- PReLU's alpha
     * branch has the exact same semantics). alpha_offset is the one genuinely new
     * quantized param; block_size is PReLUScalar's flat-vector element count (its
     * `scalar_is_input` argument is always true in every real generated test, so it's
     * hardcoded in firmware rather than added as a session field). */
    int32_t alpha_offset;
    int32_t block_size;
    /* QuantizationFunctions Quantize/Dequantize scalar params -- see parse_scalar() and
     * run_quantize_once()/run_dequantize_once() in benchmark_server_session.c. Reuses
     * input_offset/output_offset above for the kernel's zero_point (whichever side is the
     * quantized tensor). scale_bits is the float scale reinterpreted bit-for-bit as int32
     * (scalar params are transmitted as int32 only; this avoids any precision loss a fixed-
     * point encoding like atol_q16/rtol_q16 would introduce). activation_kind selects the
     * float-domain clamp applied by the generated test around the kernel call
     * (0=NONE, 1=RELU, 2=RELU6) -- Quantize applies it to the input before quantizing,
     * Dequantize applies it to the output after dequantizing. */
    int32_t scale_bits;
    int32_t activation_kind;
    /* SoftmaxFunctions Softmax scalar params -- see parse_scalar() and run_softmax_once() in
     * benchmark_server_session.c. Reuses out_mult/out_shift above for the kernel's
     * mult/shift requantization pair. num_rows/row_size are the flattened 2D view CMSIS-NN
     * softmax always operates on (softmax runs over the last dimension; num_rows is the
     * product of every other dimension). diff_min is the S8/S8-in-S16-out kernels' int8
     * saturation-radius clamp (unused, left 0, for the pure-S16 kernel which has no such
     * parameter). The S16 kernel's LUT tables are fixed CMSIS-NN reference constants
     * (identical across every generated test case), so they are embedded once as static
     * firmware data rather than transmitted per case. */
    int32_t num_rows;
    int32_t row_size;
    int32_t diff_min;
    /* FullyConnectedFunctions FullyConnected scalar params -- see parse_scalar() and
     * run_fully_connected_once() in benchmark_server_session.c. Reuses input_offset/
     * output_offset/activation_min/activation_max above (same fc_params semantics as
     * convolve's conv_params). filter_offset is FullyConnected-specific (the weight
     * zero-point term some descriptors use, e.g. fully_connected_weight_offset_s8) --
     * Convolve/DepthwiseConv never need this since their weights are always
     * symmetric-quantized (zero_point 0). */
    int32_t filter_offset;
    /* Additional scalar params for newly-bridged BasicMathFunctions ops:
     * - output_n / axis{n,h,w,c} / axis are used by reduction ops (ArgMax/ArgMin/Mean/
     *   ReduceMax/ReduceMin) whose output shape is not always the implicit n=1 convention
     *   older adapters hardcode.
     * - pad_offset_h / pad_offset_w are TransposeConv's additional padding-offset terms
     *   (distinct from the main padding.{h,w} values already shared with Conv/Pool).
     * - needs_rescale is shared by Abs and RsqrtUniversal's bool-like requantization flag.
     */
    int32_t axis_n;
    int32_t axis_h;
    int32_t axis_w;
    int32_t axis_c;
    int32_t axis;
    int32_t needs_rescale;
    /* Phase 7a invalid-argument status-assertion coverage reuses the normal streamed blobs
     * but sometimes must still pass a real NULL pointer into the kernel, matching the
     * standalone generated harness exactly (BroadcastTo/DynamicUpdateSlice null-input/
     * null-update/null-params/null-output cases). null_arg_mask selects those forced-NULL
     * pointer positions: bit0=input_0/operand, bit1=input_1/update, bit2=input_2/
     * start_indices, bit3=params struct, bit4=output buffer pointer. */
    int32_t null_arg_mask;
    /* FullyConnectedFunctions BatchMatMul scalar params -- see parse_scalar() and
     * run_batch_matmul_once() in benchmark_server_session.c. Reuses input_offset (lhs
     * zero point), filter_offset (rhs zero point), output_offset, activation_min/max,
     * and out_mult/out_shift (the kernel's single per-tensor requantization pair --
     * BatchMatMul has no per-channel variant, unlike FullyConnected) above. No adj_x/
     * adj_y field is needed: arm_batch_matmul_{s8,s16}() never reads bmm_params->adj_x/
     * adj_y (see the kernel source -- "Does not perform transposes"), so the real
     * generated test's transposed-operand descriptors simply pre-arrange their raw
     * `_input_lhs`/`_input_rhs` header array data (and dims) into the final row-major
     * layout the kernel expects; the bridge only ever needs to stream that already-
     * correct data/dims through unchanged.
     */
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
