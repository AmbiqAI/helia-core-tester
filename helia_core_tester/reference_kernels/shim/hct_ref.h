/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * C ABI over the vendored TFLM reference kernels (third_party/tflite_micro).
 *
 * Conventions mirror cmsis_nn_*_params so one parameter set feeds both the
 * reference call and the generated C harness:
 *   input_offset = -input_zero_point, output_offset = +output_zero_point,
 *   shift > 0 is a left shift.
 * Shapes are TFLite order: NHWC activations, OHWI conv / tconv filters,
 * 1HWC depthwise filters, [out, in] fully-connected weights.
 * Outputs are caller-allocated. Every entry validates its arguments and
 * returns an HCT_REF_E_* code instead of touching memory on bad input.
 */
#ifndef HCT_REF_H
#define HCT_REF_H

#include <stdint.h>

/* The library is built -fvisibility=hidden: only these entries are exported, so the
 * vendored tflite:: symbols inside it can neither interpose on nor be interposed by
 * another copy of TFLite loaded in the same process (TensorFlow, ai_edge_litert). */
#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

#define HCT_REF_ABI_VERSION 3
#define HCT_REF_MAX_RANK 6

#define HCT_REF_OK 0
#define HCT_REF_E_DIMS 1
#define HCT_REF_E_PARAM 2
#define HCT_REF_E_UNSUPPORTED 3
#define HCT_REF_E_NULL 4

#define HCT_REF_ACT_NONE 0
#define HCT_REF_ACT_RELU 1
#define HCT_REF_ACT_RELU_N1_TO_1 2
#define HCT_REF_ACT_RELU6 3

typedef struct
{
    int32_t rank;
    int32_t dims[HCT_REF_MAX_RANK];
} HctShape;

typedef struct
{
    int32_t min;
    int32_t max;
    float fmin;
    float fmax;
} HctActivation;

typedef struct
{
    int32_t stride_h;
    int32_t stride_w;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t pad_h_offset;
    int32_t pad_w_offset;
    int32_t input_offset;
    int32_t output_offset;
    HctActivation act;
} HctConvParams;

typedef struct
{
    HctConvParams conv;
    int32_t depth_multiplier;
} HctDwConvParams;

typedef struct
{
    int32_t input_offset;
    int32_t weights_offset;
    int32_t output_offset;
    HctActivation act;
} HctFcParams;

typedef struct
{
    int32_t stride_h;
    int32_t stride_w;
    int32_t filter_h;
    int32_t filter_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t pad_h_offset;
    int32_t pad_w_offset;
    HctActivation act;
} HctPoolParams;

typedef struct
{
    int32_t multiplier;
    int32_t shift;
} HctQuant;

/* Per-channel requantization: `count` entries each. */
typedef struct
{
    const int32_t *multiplier;
    const int32_t *shift;
    int32_t count;
} HctPerChannelQuant;

int32_t hct_ref_abi_version(void);

/* ---- parameter preparation (TFLM arithmetic, never re-implemented in Python) ---- */
int32_t hct_ref_quantize_multiplier(double real_multiplier, HctQuant *out);
int32_t hct_ref_quantize_multiplier_smaller_than_one_exp(double real_multiplier, HctQuant *out);
int32_t hct_ref_preprocess_softmax_scaling(double beta,
                                           double input_scale,
                                           int32_t input_integer_bits,
                                           HctQuant *out);
int32_t hct_ref_calculate_input_radius(int32_t input_integer_bits,
                                       int32_t input_left_shift,
                                       int32_t total_signed_bits,
                                       int32_t *out);
int32_t hct_ref_activation_range_quantized(int32_t activation,
                                           float scale,
                                           int32_t zero_point,
                                           int32_t qmin,
                                           int32_t qmax,
                                           int32_t *act_min,
                                           int32_t *act_max);
int32_t hct_ref_downscale_multiplier_to_s16(int32_t multiplier, int16_t *out);

/* ---- convolution (input NHWC, filter OHWI, bias [O]) ---- */
int32_t hct_ref_conv_s8(const HctConvParams *params,
                        const HctPerChannelQuant *quant,
                        const HctShape *input_shape,
                        const int8_t *input,
                        const HctShape *filter_shape,
                        const int8_t *filter,
                        const int32_t *bias,
                        int32_t bias_len,
                        const HctShape *output_shape,
                        int8_t *output);
/* Packed int4 filter (two values per byte, low nibble first). */
int32_t hct_ref_conv_s4(const HctConvParams *params,
                        const HctPerChannelQuant *quant,
                        const HctShape *input_shape,
                        const int8_t *input,
                        const HctShape *filter_shape,
                        const int8_t *packed_filter,
                        const int32_t *bias,
                        int32_t bias_len,
                        const HctShape *output_shape,
                        int8_t *output);
int32_t hct_ref_conv_s16(const HctConvParams *params,
                         const HctPerChannelQuant *quant,
                         const HctShape *input_shape,
                         const int16_t *input,
                         const HctShape *filter_shape,
                         const int8_t *filter,
                         const int64_t *bias,
                         int32_t bias_len,
                         const HctShape *output_shape,
                         int16_t *output);
int32_t hct_ref_conv_s16_b32(const HctConvParams *params,
                             const HctPerChannelQuant *quant,
                             const HctShape *input_shape,
                             const int16_t *input,
                             const HctShape *filter_shape,
                             const int8_t *filter,
                             const int32_t *bias,
                             int32_t bias_len,
                             const HctShape *output_shape,
                             int16_t *output);
int32_t hct_ref_conv_f32(const HctConvParams *params,
                         const HctShape *input_shape,
                         const float *input,
                         const HctShape *filter_shape,
                         const float *filter,
                         const float *bias,
                         int32_t bias_len,
                         const HctShape *output_shape,
                         float *output);

/* ---- depthwise convolution (input NHWC, filter 1HWC with C = in_ch * depth_multiplier) ---- */
int32_t hct_ref_dwconv_s8(const HctDwConvParams *params,
                          const HctPerChannelQuant *quant,
                          const HctShape *input_shape,
                          const int8_t *input,
                          const HctShape *filter_shape,
                          const int8_t *filter,
                          const int32_t *bias,
                          int32_t bias_len,
                          const HctShape *output_shape,
                          int8_t *output);
int32_t hct_ref_dwconv_s4(const HctDwConvParams *params,
                          const HctPerChannelQuant *quant,
                          const HctShape *input_shape,
                          const int8_t *input,
                          const HctShape *filter_shape,
                          const int8_t *packed_filter,
                          const int32_t *bias,
                          int32_t bias_len,
                          const HctShape *output_shape,
                          int8_t *output);
int32_t hct_ref_dwconv_s16(const HctDwConvParams *params,
                           const HctPerChannelQuant *quant,
                           const HctShape *input_shape,
                           const int16_t *input,
                           const HctShape *filter_shape,
                           const int8_t *filter,
                           const int64_t *bias,
                           int32_t bias_len,
                           const HctShape *output_shape,
                           int16_t *output);
int32_t hct_ref_dwconv_f32(const HctDwConvParams *params,
                           const HctShape *input_shape,
                           const float *input,
                           const HctShape *filter_shape,
                           const float *filter,
                           const float *bias,
                           int32_t bias_len,
                           const HctShape *output_shape,
                           float *output);

/* ---- fully connected (input [..., in], weights [out, in], output [..., out]) ----
 * quant->count == 1 selects the per-tensor kernel; quant->count == out the
 * per-channel one. Both honour weights_offset (the per-channel kernel runs on
 * weights widened with the offset applied, as CMSIS-NN computes it). */
int32_t hct_ref_fc_s8(const HctFcParams *params,
                      const HctPerChannelQuant *quant,
                      const HctShape *input_shape,
                      const int8_t *input,
                      const HctShape *filter_shape,
                      const int8_t *filter,
                      const int32_t *bias,
                      int32_t bias_len,
                      const HctShape *output_shape,
                      int8_t *output);
int32_t hct_ref_fc_s4(const HctFcParams *params,
                      const HctPerChannelQuant *quant,
                      const HctShape *input_shape,
                      const int8_t *input,
                      const HctShape *filter_shape,
                      const int8_t *packed_filter,
                      const int32_t *bias,
                      int32_t bias_len,
                      const HctShape *output_shape,
                      int8_t *output);
int32_t hct_ref_fc_s16(const HctFcParams *params,
                       const HctPerChannelQuant *quant,
                       const HctShape *input_shape,
                       const int16_t *input,
                       const HctShape *filter_shape,
                       const int8_t *filter,
                       const int64_t *bias,
                       int32_t bias_len,
                       const HctShape *output_shape,
                       int16_t *output);
int32_t hct_ref_fc_f32(const HctFcParams *params,
                       const HctShape *input_shape,
                       const float *input,
                       const HctShape *filter_shape,
                       const float *filter,
                       const float *bias,
                       int32_t bias_len,
                       const HctShape *output_shape,
                       float *output);

/* ---- transpose convolution (input NHWC, filter OHWI, output NHWC) ---- */
int32_t hct_ref_tconv_s8(const HctConvParams *params,
                         const HctPerChannelQuant *quant,
                         const HctShape *input_shape,
                         const int8_t *input,
                         const HctShape *filter_shape,
                         const int8_t *filter,
                         const int32_t *bias,
                         int32_t bias_len,
                         const HctShape *output_shape,
                         int8_t *output);
int32_t hct_ref_tconv_s16(const HctConvParams *params,
                          const HctPerChannelQuant *quant,
                          const HctShape *input_shape,
                          const int16_t *input,
                          const HctShape *filter_shape,
                          const int8_t *filter,
                          const int64_t *bias,
                          int32_t bias_len,
                          const HctShape *output_shape,
                          int16_t *output);
int32_t hct_ref_tconv_f32(const HctConvParams *params,
                          const HctShape *input_shape,
                          const float *input,
                          const HctShape *filter_shape,
                          const float *filter,
                          const float *bias,
                          int32_t bias_len,
                          const HctShape *output_shape,
                          float *output);

/* ---- pooling (NHWC; quantized output shares the input quantization) ---- */
int32_t hct_ref_avgpool_s8(const HctPoolParams *params,
                           const HctShape *input_shape,
                           const int8_t *input,
                           const HctShape *output_shape,
                           int8_t *output);
int32_t hct_ref_avgpool_s16(const HctPoolParams *params,
                            const HctShape *input_shape,
                            const int16_t *input,
                            const HctShape *output_shape,
                            int16_t *output);
int32_t hct_ref_avgpool_f32(const HctPoolParams *params,
                            const HctShape *input_shape,
                            const float *input,
                            const HctShape *output_shape,
                            float *output);
int32_t hct_ref_maxpool_s8(const HctPoolParams *params,
                           const HctShape *input_shape,
                           const int8_t *input,
                           const HctShape *output_shape,
                           int8_t *output);
int32_t hct_ref_maxpool_s16(const HctPoolParams *params,
                            const HctShape *input_shape,
                            const int16_t *input,
                            const HctShape *output_shape,
                            int16_t *output);
int32_t hct_ref_maxpool_f32(const HctPoolParams *params,
                            const HctShape *input_shape,
                            const float *input,
                            const HctShape *output_shape,
                            float *output);

/* ---- elementwise binary arithmetic (TFLM ProcessBroadcastShapes; rank <= 4) ----
 * Offsets follow the convention above: input_offset = -zp, output_offset = +zp.
 * Add/Sub use all fields (left_shift 20 for s8, 15 for s16); Mul ignores the
 * per-input multipliers and left_shift. s16 offsets must be 0. */
typedef struct
{
    int32_t left_shift;
    int32_t input1_offset;
    int32_t input1_multiplier;
    int32_t input1_shift;
    int32_t input2_offset;
    int32_t input2_multiplier;
    int32_t input2_shift;
    int32_t output_offset;
    int32_t output_multiplier;
    int32_t output_shift;
    HctActivation act;
} HctBinaryParams;

int32_t hct_ref_add_s8(const HctBinaryParams *params,
                       const HctShape *input1_shape,
                       const int8_t *input1,
                       const HctShape *input2_shape,
                       const int8_t *input2,
                       const HctShape *output_shape,
                       int8_t *output);
int32_t hct_ref_add_s16(const HctBinaryParams *params,
                        const HctShape *input1_shape,
                        const int16_t *input1,
                        const HctShape *input2_shape,
                        const int16_t *input2,
                        const HctShape *output_shape,
                        int16_t *output);
int32_t hct_ref_sub_s8(const HctBinaryParams *params,
                       const HctShape *input1_shape,
                       const int8_t *input1,
                       const HctShape *input2_shape,
                       const int8_t *input2,
                       const HctShape *output_shape,
                       int8_t *output);
int32_t hct_ref_sub_s16(const HctBinaryParams *params,
                        const HctShape *input1_shape,
                        const int16_t *input1,
                        const HctShape *input2_shape,
                        const int16_t *input2,
                        const HctShape *output_shape,
                        int16_t *output);
int32_t hct_ref_mul_s8(const HctBinaryParams *params,
                       const HctShape *input1_shape,
                       const int8_t *input1,
                       const HctShape *input2_shape,
                       const int8_t *input2,
                       const HctShape *output_shape,
                       int8_t *output);
int32_t hct_ref_mul_s16(const HctBinaryParams *params,
                        const HctShape *input1_shape,
                        const int16_t *input1,
                        const HctShape *input2_shape,
                        const int16_t *input2,
                        const HctShape *output_shape,
                        int16_t *output);

/* ---- activations: *_prepare mirrors the TFLM micro prepare, eval the reference kernel ---- */

/* Softmax: s8 output has zero point -128 and scale 1/256; s16 output zero point
 * 0 and scale 1/32768 (LUTs populated as TFLM's InitializeLutForInt16). */
typedef struct
{
    int32_t input_multiplier;
    int32_t input_left_shift;
    int32_t diff_min; /* s8 only */
} HctSoftmaxParams;

int32_t hct_ref_softmax_s8(const HctSoftmaxParams *params, const HctShape *shape, const int8_t *input, int8_t *output);
int32_t hct_ref_softmax_s16(const HctSoftmaxParams *params,
                            const HctShape *shape,
                            const int16_t *input,
                            int16_t *output);

/* Tanh / Logistic, int16 (zero point 0, output scale 2^-15). */
typedef struct
{
    int32_t input_zero_point;
    int32_t input_range_radius;
    int32_t input_multiplier;
    int32_t input_left_shift;
} HctLutActParams;

int32_t hct_ref_tanh_logistic_s16_prepare(int32_t logistic,
                                          float input_scale,
                                          float output_scale,
                                          HctLutActParams *out);
int32_t hct_ref_tanh_s16(const HctLutActParams *params, const HctShape *shape, const int16_t *input, int16_t *output);
int32_t hct_ref_logistic_s16(const HctLutActParams *params,
                             const HctShape *shape,
                             const int16_t *input,
                             int16_t *output);

/* LeakyRelu: zero points as stored (the kernel subtracts the input one). */
typedef struct
{
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t multiplier_alpha;
    int32_t shift_alpha;
    int32_t multiplier_identity;
    int32_t shift_identity;
} HctLeakyReluParams;

int32_t hct_ref_leaky_relu_prepare(float input_scale,
                                   int32_t input_zero_point,
                                   float alpha,
                                   float output_scale,
                                   int32_t output_zero_point,
                                   HctLeakyReluParams *out);
int32_t hct_ref_leaky_relu_s8(const HctLeakyReluParams *params,
                              const HctShape *shape,
                              const int8_t *input,
                              int8_t *output);
int32_t hct_ref_leaky_relu_s16(const HctLeakyReluParams *params,
                               const HctShape *shape,
                               const int16_t *input,
                               int16_t *output);

/* PReLU (BroadcastPrelu4DSlow; rank <= 4, alpha broadcast onto the input). */
typedef struct
{
    int32_t input_offset;
    int32_t alpha_offset;
    int32_t output_offset;
    int32_t multiplier_1;
    int32_t shift_1;
    int32_t multiplier_2;
    int32_t shift_2;
} HctPreluParams;

int32_t hct_ref_prelu_prepare(float input_scale,
                              int32_t input_zero_point,
                              float alpha_scale,
                              int32_t alpha_zero_point,
                              float output_scale,
                              int32_t output_zero_point,
                              HctPreluParams *out);
int32_t hct_ref_prelu_s8(const HctPreluParams *params,
                         const HctShape *input_shape,
                         const int8_t *input,
                         const HctShape *alpha_shape,
                         const int8_t *alpha,
                         const HctShape *output_shape,
                         int8_t *output);

/* HardSwish, int8. */
typedef struct
{
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t reluish_multiplier_fixedpoint_int16;
    int32_t reluish_multiplier_exponent;
    int32_t output_multiplier_fixedpoint_int16;
    int32_t output_multiplier_exponent;
} HctHardSwishParams;

int32_t hct_ref_hard_swish_prepare(float input_scale,
                                   int32_t input_zero_point,
                                   float output_scale,
                                   int32_t output_zero_point,
                                   HctHardSwishParams *out);
int32_t hct_ref_hard_swish_s8(const HctHardSwishParams *params,
                              const HctShape *shape,
                              const int8_t *input,
                              int8_t *output);

/* Relu / Relu6 (TFLM ReluQuantized = TFLite QuantizedReluX): requantize, then
 * clamp. act_max_real is +inf for Relu. */
typedef struct
{
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t output_multiplier;
    int32_t output_shift;
    int32_t act_min;
    int32_t act_max;
} HctReluParams;

int32_t hct_ref_relu_prepare(float input_scale,
                             int32_t input_zero_point,
                             float output_scale,
                             int32_t output_zero_point,
                             float act_min_real,
                             float act_max_real,
                             int32_t qmin,
                             int32_t qmax,
                             HctReluParams *out);
int32_t hct_ref_relu_s8(const HctReluParams *params, const HctShape *shape, const int8_t *input, int8_t *output);
int32_t hct_ref_relu_s16(const HctReluParams *params, const HctShape *shape, const int16_t *input, int16_t *output);

/* ---- integer Mean (QuantizedMeanOrSum; multiplier/shift = QuantizeMultiplier(si/so),
 * the 1/count fold happens inside, as hct_ref_mean_fold reproduces) ---- */
typedef struct
{
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t multiplier;
    int32_t shift;
    int32_t keep_dims;
} HctMeanParams;

int32_t hct_ref_mean_fold(int32_t multiplier, int32_t shift, int64_t count, HctQuant *out);
int32_t hct_ref_mean_s8(const HctMeanParams *params,
                        const HctShape *input_shape,
                        const int8_t *input,
                        const int32_t *axes,
                        int32_t num_axes,
                        const HctShape *output_shape,
                        int8_t *output);
int32_t hct_ref_mean_s16(const HctMeanParams *params,
                         const HctShape *input_shape,
                         const int16_t *input,
                         const int32_t *axes,
                         int32_t num_axes,
                         const HctShape *output_shape,
                         int16_t *output);

/* ---- Rsqrt as TFLite's elementwise.cc evaluates it (input >= zero point) ---- */
typedef struct
{
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t multiplier;
    int32_t shift;
} HctRsqrtParams;

int32_t hct_ref_rsqrt_prepare(float input_scale,
                              int32_t input_zero_point,
                              float output_scale,
                              int32_t output_zero_point,
                              HctRsqrtParams *out);
int32_t hct_ref_rsqrt_s8(const HctRsqrtParams *params, const HctShape *shape, const int8_t *input, int8_t *output);
/* int16 follows TFLite's LUT route (the table needs the scales); multiplier/shift unused. */
int32_t hct_ref_rsqrt_s16(const HctRsqrtParams *params,
                          float input_scale,
                          float output_scale,
                          const HctShape *shape,
                          const int16_t *input,
                          int16_t *output);

/* ---- Quantize float32 -> int (AffineQuantize: round(x / scale) + zero point, saturated) ---- */
int32_t hct_ref_quantize_f32_s8(float scale, int32_t zero_point, const HctShape *shape, const float *input, int8_t *output);
int32_t hct_ref_quantize_f32_s16(float scale,
                                 int32_t zero_point,
                                 const HctShape *shape,
                                 const float *input,
                                 int16_t *output);

/* ---- BatchMatMul (TFLM reference; adj flags resolved by the caller):
 * lhs [..., M, K], rhs [..., N, K], output [..., M, N]; lhs_offset = -lhs zp,
 * rhs_offset = -rhs zp, output_offset = +zp; int16 offsets are all zero ---- */
typedef struct
{
    int32_t lhs_offset;
    int32_t rhs_offset;
    int32_t output_offset;
    int32_t output_multiplier;
    int32_t output_shift;
    HctActivation act;
} HctBmmParams;

int32_t hct_ref_bmm_s8(const HctBmmParams *params,
                       const HctShape *lhs_shape,
                       const int8_t *lhs,
                       const HctShape *rhs_shape,
                       const int8_t *rhs,
                       const HctShape *output_shape,
                       int8_t *output);
int32_t hct_ref_bmm_s16(const HctBmmParams *params,
                        const HctShape *lhs_shape,
                        const int16_t *lhs,
                        const HctShape *rhs_shape,
                        const int16_t *rhs,
                        const HctShape *output_shape,
                        int16_t *output);

/* Struct sizes, asserted against the ctypes mirrors at load time. */
int32_t hct_ref_sizeof(const char *type_name);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif /* HCT_REF_H */
