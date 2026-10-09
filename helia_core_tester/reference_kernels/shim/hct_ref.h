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

#ifdef __cplusplus
extern "C" {
#endif

#define HCT_REF_ABI_VERSION 1
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

/* Struct sizes, asserted against the ctypes mirrors at load time. */
int32_t hct_ref_sizeof(const char *type_name);

#ifdef __cplusplus
}
#endif

#endif /* HCT_REF_H */
