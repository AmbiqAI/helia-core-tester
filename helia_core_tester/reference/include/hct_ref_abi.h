/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated from spec/entries.yaml by helia_core_tester.generation.reference.abi.
 * Do not edit; run `python -m helia_core_tester.generation.reference.abi --write`.
 */
#ifndef HCT_REF_ABI_H
#define HCT_REF_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#define HCT_REF_ABI_VERSION 1
#define HCT_MAX_RANK 8

typedef enum
{
    HCT_INT8 = 1,
    HCT_INT16 = 2,
    HCT_INT32 = 3,
    HCT_INT64 = 4,
    HCT_FLOAT32 = 5,
    HCT_FLOAT16 = 6,
    HCT_BOOL = 7
} HctDtype;

typedef enum
{
    HCT_OK = 0,
    HCT_E_NULL = 1,
    HCT_E_COUNT = 2,
    HCT_E_DTYPE = 3,
    HCT_E_SHAPE = 4,
    HCT_E_PARAM = 5,
    HCT_E_SIZE = 6
} HctStatus;

typedef enum
{
    HCT_ACT_NONE = 0,
    HCT_ACT_RELU = 1,
    HCT_ACT_RELU6 = 2,
    HCT_ACT_RELU_N1_TO_1 = 3
} HctActivation;

typedef enum
{
    HCT_CMP_EQUAL = 0,
    HCT_CMP_NOT_EQUAL = 1,
    HCT_CMP_GREATER = 2,
    HCT_CMP_GREATER_EQUAL = 3,
    HCT_CMP_LESS = 4,
    HCT_CMP_LESS_EQUAL = 5
} HctComparison;

typedef enum
{
    HCT_FACT_NONE = 0,
    HCT_FACT_RELU = 1,
    HCT_FACT_RELU6 = 2,
    HCT_FACT_LEAKY_RELU = 3,
    HCT_FACT_SIGMOID = 4,
    HCT_FACT_TANH = 5,
    HCT_FACT_HARDSWISH = 6
} HctFloatActivation;

/* A tensor: dtype is an HctDtype, dims[0..rank-1] row-major (NHWC for 4-D), data
 * points at rank-product elements (1 for rank 0). */
typedef struct
{
    int32_t dtype;
    int32_t rank;
    int32_t dims[HCT_MAX_RANK];
    void *data;
} HctTensor;

/* A real multiplier, as TFLite's QuantizeMultiplier takes it. */
typedef struct
{
    double real_multiplier;
} HctQuantizeMultiplierIn;

typedef struct
{
    int32_t multiplier;
    int32_t shift;
} HctQuantizeMultiplierOut;

/* CalculateActivationRangeQuantized. activation is an HctActivation value. */
typedef struct
{
    int32_t activation;
    int32_t dtype;
    float scale;
    int32_t zero_point;
} HctActivationRangeIn;

typedef struct
{
    int32_t min;
    int32_t max;
} HctActivationRangeOut;

/* Per-tensor quantization of an int8/int16 Add, scales as float32 as TFLite stores them. */
typedef struct
{
    int32_t dtype;
    int32_t activation;
    float input1_scale;
    int32_t input1_zero_point;
    float input2_scale;
    int32_t input2_zero_point;
    float output_scale;
    int32_t output_zero_point;
} HctAddQuant;

/* Offsets follow CMSIS-NN; input offsets are -zero_point, output offset is +zero_point. */
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
    int32_t activation_min;
    int32_t activation_max;
} HctAddParams;

/* Float clamp; +/-INFINITY leaves the side unbounded. */
typedef struct
{
    float activation_min;
    float activation_max;
} HctFloatActivationParams;

/* Entries that take no parameters; the field is ignored. */
typedef struct
{
    int32_t unused;
} HctNoParams;

typedef struct
{
    int32_t dtype;
    int32_t activation;
    float input1_scale;
    int32_t input1_zero_point;
    float input2_scale;
    int32_t input2_zero_point;
    float output_scale;
    int32_t output_zero_point;
} HctMulQuant;

typedef struct
{
    int32_t input1_offset;
    int32_t input2_offset;
    int32_t output_offset;
    int32_t output_multiplier;
    int32_t output_shift;
    int32_t activation_min;
    int32_t activation_max;
} HctMulParams;

/* As HctAddParams; left_shift is 7 (int8) or 0 (int16) and the output multiplier is general. */
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
    int32_t activation_min;
    int32_t activation_max;
} HctSquaredDifferenceParams;

/* operation is an HctComparison value. */
typedef struct
{
    int32_t dtype;
    int32_t operation;
    float input1_scale;
    int32_t input1_zero_point;
    float input2_scale;
    int32_t input2_zero_point;
} HctComparisonQuant;

typedef struct
{
    int32_t operation;
    int32_t left_shift;
    int32_t input1_offset;
    int32_t input1_multiplier;
    int32_t input1_shift;
    int32_t input2_offset;
    int32_t input2_multiplier;
    int32_t input2_shift;
} HctComparisonParams;

typedef struct
{
    int32_t dtype;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
} HctAbsQuant;

/* TFLite's Abs; input_zero_point is subtracted, the result rescaled only when needs_rescale. */
typedef struct
{
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t needs_rescale;
    int32_t multiplier;
    int32_t shift;
} HctAbsParams;

typedef struct
{
    int32_t activation_min;
    int32_t activation_max;
} HctClampParams;

/* Relu and Relu6 (TFLM CalculateReluOpData); act_max is +INFINITY for Relu, 6 for Relu6. */
typedef struct
{
    int32_t dtype;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
    float act_max;
} HctReluQuant;

/* input_offset/output_offset are the zero points, as TFLite's ReluQuantized takes them. */
typedef struct
{
    int32_t input_offset;
    int32_t output_offset;
    int32_t output_multiplier;
    int32_t output_shift;
    int32_t activation_min;
    int32_t activation_max;
} HctReluParams;

typedef struct
{
    int32_t dtype;
    float alpha;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
} HctLeakyReluQuant;

typedef struct
{
    int32_t dtype;
    float input_scale;
    int32_t input_zero_point;
    float alpha_scale;
    int32_t alpha_zero_point;
    float output_scale;
    int32_t output_zero_point;
} HctPreluQuant;

/* TFLite's BroadcastPrelu4D; input/alpha offsets are -zero_point, output offset +zero_point. */
typedef struct
{
    int32_t input_offset;
    int32_t alpha_offset;
    int32_t output_offset;
    int32_t identity_multiplier;
    int32_t identity_shift;
    int32_t alpha_multiplier;
    int32_t alpha_shift;
} HctPreluParams;

/* TFLite's QuantizeLeakyRelu; offsets are the zero points. */
typedef struct
{
    int32_t input_offset;
    int32_t output_offset;
    int32_t alpha_multiplier;
    int32_t alpha_shift;
    int32_t identity_multiplier;
    int32_t identity_shift;
} HctLeakyReluParams;

typedef struct
{
    int32_t dtype;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
} HctHardSwishQuant;

/* TFLite's HardSwishParams (int8); Q15 multipliers with frexp exponents, offsets are the zero points. */
typedef struct
{
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t output_multiplier_fixedpoint_int16;
    int32_t output_multiplier_exponent;
    int32_t reluish_multiplier_fixedpoint_int16;
    int32_t reluish_multiplier_exponent;
} HctHardSwishParams;

/* CMSIS-NN arm_hard_swish_precise_* (no TFLite counterpart): relu_q3/relu_q6 are round(3|6 / input_scale), the output multiplier is input_scale^2 / (6 output_scale) * 2^prescale. */
typedef struct
{
    int32_t input_offset;
    int32_t output_offset;
    int32_t output_multiplier;
    int32_t output_shift;
    int32_t relu_q3;
    int32_t relu_q6;
    int32_t prescale;
} HctHardSwishPreciseParams;

/* int16 Tanh/Logistic (TFLM prepare); zero points must be 0 and the output scale 2^-15. */
typedef struct
{
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
} HctTanhLogisticQuant;

/* reference_integer_ops::Tanh/Logistic(int16): x * input_multiplier, rounding right shift by input_left_shift; input_multiplier 0 means 3 << input_left_shift and no shift. */
typedef struct
{
    int32_t input_multiplier;
    int32_t input_left_shift;
} HctTanhLogisticParams;

/* TFLM CalculateSoftmaxParams. int8 -> int8 needs output 1/256 at -128, int8 -> int16 1/65536 at -32768, int16 -> int16 1/32768 at 0. */
typedef struct
{
    int32_t input_dtype;
    int32_t output_dtype;
    float beta;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
} HctSoftmaxQuant;

/* SoftmaxParams: int8 rescales x - max by MultiplyByQuantizedMultiplierGreaterThanOne and drops differences below diff_min; int16 by MultiplyByQuantizedMultiplier (diff_min unused, 0). */
typedef struct
{
    int32_t input_multiplier;
    int32_t input_left_shift;
    int32_t diff_min;
} HctSoftmaxParams;

/* Per-tensor input and output quantization, for entries that work in float on the dequantized value. */
typedef struct
{
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
} HctUnaryQuantParams;

/* AffineQuantize / Dequantize at (scale, zero_point); the float side is clamped to [activation_min, activation_max] (before quantizing, after dequantizing). */
typedef struct
{
    float scale;
    int32_t zero_point;
    float activation_min;
    float activation_max;
} HctQuantizeParams;

/* CMSIS-NN arm_requantize_* - the zero points subtracted and added around arm_nn_requantize. */
typedef struct
{
    int32_t multiplier;
    int32_t shift;
    int32_t input_zero_point;
    int32_t output_zero_point;
} HctRequantizeParams;

/* An HctFloatActivation and its parameter (the LeakyRelu slope; ignored otherwise). */
typedef struct
{
    int32_t activation_type;
    float act_param;
} HctFloatActivationTypeParams;

/* PopulateConvolutionQuantizationParams: each channel's (input_scale * filter_scale[c]) / output_scale, formed in double from the float32 scales, through QuantizeMultiplier. */
typedef struct
{
    float input_scale;
    float output_scale;
} HctPerChannelQuantParams;

/* TFLite ConvParams for the quantized convolutions: input_offset is -input_zero_point, output_offset the output zero point (both 0 for int16); groups follow from input depth / filter input depth. */
typedef struct
{
    int32_t stride_h;
    int32_t stride_w;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t input_offset;
    int32_t output_offset;
    int32_t activation_min;
    int32_t activation_max;
} HctConvParams;

typedef struct
{
    int32_t stride_h;
    int32_t stride_w;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t pad_h;
    int32_t pad_w;
    float activation_min;
    float activation_max;
} HctConvFloatParams;

/* TFLite FullyConnectedParams: input_offset is -input_zero_point, filter_offset -filter_zero_point (0 when the filter is symmetric, as per-channel filters are), output_offset the output zero point. */
typedef struct
{
    int32_t input_offset;
    int32_t filter_offset;
    int32_t output_offset;
    int32_t activation_min;
    int32_t activation_max;
} HctFullyConnectedParams;

/* adj_x / adj_y (0 or 1) mean the stored operand is transposed in its last two dims; offsets are -zero_point for the operands, +zero_point for the output; one per-tensor multiplier and shift. */
typedef struct
{
    int32_t adj_x;
    int32_t adj_y;
    int32_t lhs_offset;
    int32_t rhs_offset;
    int32_t output_offset;
    int32_t multiplier;
    int32_t shift;
    int32_t activation_min;
    int32_t activation_max;
} HctBatchMatMulParams;

typedef struct
{
    int32_t adj_x;
    int32_t adj_y;
    float activation_min;
    float activation_max;
} HctBatchMatMulFloatParams;

typedef struct
{
    int32_t stride_h;
    int32_t stride_w;
    int32_t filter_h;
    int32_t filter_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t activation_min;
    int32_t activation_max;
} HctPoolParams;

typedef struct
{
    int32_t stride_h;
    int32_t stride_w;
    int32_t filter_h;
    int32_t filter_w;
    int32_t pad_h;
    int32_t pad_w;
    float activation_min;
    float activation_max;
} HctPoolFloatParams;

/* Bit d set reduces input dim d. */
typedef struct
{
    int32_t axis_mask;
} HctAxisMaskParams;

typedef struct
{
    int32_t axis;
} HctAxisParams;

/* QuantizedMeanOrSum's requantization for a reduction over `count` elements. */
typedef struct
{
    float input_scale;
    float output_scale;
    int64_t count;
} HctMeanQuant;

/* input_scale / output_scale through QuantizeMultiplier, with 1 / count folded in as TFLM folds it. */
typedef struct
{
    int32_t multiplier;
    int32_t shift;
} HctMeanParams;

typedef struct
{
    int32_t axis_mask;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t multiplier;
    int32_t shift;
} HctMeanQuantizedParams;

int32_t hct_ref_abi_version(void);

int32_t hct_ref_quantize_multiplier(const HctQuantizeMultiplierIn *in, HctQuantizeMultiplierOut *out);
int32_t hct_ref_activation_range_quantized(const HctActivationRangeIn *in, HctActivationRangeOut *out);
int32_t hct_ref_add_prepare(const HctAddQuant *in, HctAddParams *out);
int32_t hct_ref_sub_prepare(const HctAddQuant *in, HctAddParams *out);
int32_t hct_ref_mul_prepare(const HctMulQuant *in, HctMulParams *out);
int32_t hct_ref_squared_difference_prepare(const HctAddQuant *in, HctSquaredDifferenceParams *out);
int32_t hct_ref_comparison_prepare(const HctComparisonQuant *in, HctComparisonParams *out);
int32_t hct_ref_abs_prepare(const HctAbsQuant *in, HctAbsParams *out);
int32_t hct_ref_relu_prepare(const HctReluQuant *in, HctReluParams *out);
int32_t hct_ref_leaky_relu_prepare(const HctLeakyReluQuant *in, HctLeakyReluParams *out);
int32_t hct_ref_prelu_prepare(const HctPreluQuant *in, HctPreluParams *out);
int32_t hct_ref_hard_swish_prepare(const HctHardSwishQuant *in, HctHardSwishParams *out);
int32_t hct_ref_hard_swish_precise_prepare(const HctHardSwishQuant *in, HctHardSwishPreciseParams *out);
int32_t hct_ref_tanh_prepare(const HctTanhLogisticQuant *in, HctTanhLogisticParams *out);
int32_t hct_ref_logistic_prepare(const HctTanhLogisticQuant *in, HctTanhLogisticParams *out);
int32_t hct_ref_softmax_prepare(const HctSoftmaxQuant *in, HctSoftmaxParams *out);
int32_t hct_ref_mean_prepare(const HctMeanQuant *in, HctMeanParams *out);

/* inputs [input1:int8, input2:int8] -> outputs [output:int8] */
int32_t hct_ref_add_s8(const HctAddParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int16, input2:int16] -> outputs [output:int16] */
int32_t hct_ref_add_s16(const HctAddParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float32, input2:float32] -> outputs [output:float32] */
int32_t hct_ref_add_f32(const HctFloatActivationParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float16, input2:float16] -> outputs [output:float16] */
int32_t hct_ref_add_f16(const HctFloatActivationParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int8, input2:int8] -> outputs [output:int8] */
int32_t hct_ref_sub_s8(const HctAddParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int16, input2:int16] -> outputs [output:int16] */
int32_t hct_ref_sub_s16(const HctAddParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int8, input2:int8] -> outputs [output:int8] */
int32_t hct_ref_mul_s8(const HctMulParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int16, input2:int16] -> outputs [output:int16] */
int32_t hct_ref_mul_s16(const HctMulParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float32, input2:float32] -> outputs [output:float32] */
int32_t hct_ref_sub_f32(const HctFloatActivationParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float16, input2:float16] -> outputs [output:float16] */
int32_t hct_ref_sub_f16(const HctFloatActivationParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float32, input2:float32] -> outputs [output:float32] */
int32_t hct_ref_mul_f32(const HctFloatActivationParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float16, input2:float16] -> outputs [output:float16] */
int32_t hct_ref_mul_f16(const HctFloatActivationParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int8, input2:int8] -> outputs [output:int8] */
int32_t hct_ref_maximum_s8(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int16, input2:int16] -> outputs [output:int16] */
int32_t hct_ref_maximum_s16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float32, input2:float32] -> outputs [output:float32] */
int32_t hct_ref_maximum_f32(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float16, input2:float16] -> outputs [output:float16] */
int32_t hct_ref_maximum_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int8, input2:int8] -> outputs [output:int8] */
int32_t hct_ref_minimum_s8(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int16, input2:int16] -> outputs [output:int16] */
int32_t hct_ref_minimum_s16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float32, input2:float32] -> outputs [output:float32] */
int32_t hct_ref_minimum_f32(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float16, input2:float16] -> outputs [output:float16] */
int32_t hct_ref_minimum_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int8, input2:int8] -> outputs [output:int8] */
int32_t hct_ref_squared_difference_s8(const HctSquaredDifferenceParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int8, input2:int8] -> outputs [output:bool] */
int32_t hct_ref_comparison_s8(const HctComparisonParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_abs_s8(const HctAbsParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_clamp_s8(const HctClampParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int16, input2:int16] -> outputs [output:int16] */
int32_t hct_ref_squared_difference_s16(const HctSquaredDifferenceParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int16, input2:int16] -> outputs [output:bool] */
int32_t hct_ref_comparison_s16(const HctComparisonParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_abs_s16(const HctAbsParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_clamp_s16(const HctClampParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float16, input2:float16] -> outputs [output:float16] */
int32_t hct_ref_squared_difference_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_abs_f32(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_abs_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_relu_s8(const HctReluParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_relu_s16(const HctReluParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_leaky_relu_s8(const HctLeakyReluParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_leaky_relu_s16(const HctLeakyReluParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8, alpha:int8] -> outputs [output:int8] */
int32_t hct_ref_prelu_s8(const HctPreluParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16, alpha:int16] -> outputs [output:int16] */
int32_t hct_ref_prelu_s16(const HctPreluParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32, alpha:float32] -> outputs [output:float32] */
int32_t hct_ref_prelu_f32(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16, alpha:float16] -> outputs [output:float16] */
int32_t hct_ref_prelu_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32, scale:float32, bias:float32] -> outputs [output:float32] */
int32_t hct_ref_batch_norm_f32(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16, scale:float16, bias:float16] -> outputs [output:float16] */
int32_t hct_ref_batch_norm_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_hard_swish_s8(const HctHardSwishParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_hard_swish_precise_s8(const HctHardSwishPreciseParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_hard_swish_precise_s16(const HctHardSwishPreciseParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_hard_swish_f32(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_hard_swish_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_tanh_s16(const HctTanhLogisticParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_logistic_s16(const HctTanhLogisticParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_softmax_s8(const HctSoftmaxParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int16] */
int32_t hct_ref_softmax_s8_s16(const HctSoftmaxParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_softmax_s16(const HctSoftmaxParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_softmax_f32(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_softmax_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_sqrt_s8(const HctUnaryQuantParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_sqrt_s16(const HctUnaryQuantParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_rsqrt_s16(const HctUnaryQuantParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_sqrt_f32(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_sqrt_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_rsqrt_f32(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_rsqrt_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:int8] */
int32_t hct_ref_quantize_f32_s8(const HctQuantizeParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:int16] */
int32_t hct_ref_quantize_f32_s16(const HctQuantizeParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:float32] */
int32_t hct_ref_dequantize_s8_f32(const HctQuantizeParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:float32] */
int32_t hct_ref_dequantize_s16_f32(const HctQuantizeParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float32] */
int32_t hct_ref_dequantize_f16_f32(const HctFloatActivationParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_requantize_s8(const HctRequantizeParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_requantize_s16(const HctRequantizeParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_nn_activation_f32(const HctFloatActivationTypeParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_nn_activation_f16(const HctFloatActivationTypeParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_tanh_lut_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_tanh_lut_mve_f16(const HctNoParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [filter_scale:float32] -> outputs [multiplier:int32, shift:int32] */
int32_t hct_ref_per_channel_quant(const HctPerChannelQuantParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8, filter:int8, bias:int32, multiplier:int32, shift:int32] -> outputs [output:int8] */
int32_t hct_ref_conv_s8(const HctConvParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16, filter:int8, bias:int64, multiplier:int32, shift:int32] -> outputs [output:int16] */
int32_t hct_ref_conv_s16(const HctConvParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32, filter:float32, bias:float32] -> outputs [output:float32] */
int32_t hct_ref_conv_f32(const HctConvFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16, filter:float16, bias:float16] -> outputs [output:float16] */
int32_t hct_ref_conv_f16(const HctConvFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8, filter:int8, bias:int32, multiplier:int32, shift:int32] -> outputs [output:int8] */
int32_t hct_ref_depthwise_conv_s8(const HctConvParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16, filter:int8, bias:int64, multiplier:int32, shift:int32] -> outputs [output:int16] */
int32_t hct_ref_depthwise_conv_s16(const HctConvParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32, filter:float32, bias:float32] -> outputs [output:float32] */
int32_t hct_ref_depthwise_conv_f32(const HctConvFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16, filter:float16, bias:float16] -> outputs [output:float16] */
int32_t hct_ref_depthwise_conv_f16(const HctConvFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8, filter:int8, bias:int32, multiplier:int32, shift:int32] -> outputs [output:int8] */
int32_t hct_ref_transpose_conv_s8(const HctConvParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16, filter:int8, bias:int64, multiplier:int32, shift:int32] -> outputs [output:int16] */
int32_t hct_ref_transpose_conv_s16(const HctConvParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32, filter:float32, bias:float32] -> outputs [output:float32] */
int32_t hct_ref_transpose_conv_f32(const HctConvFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16, filter:float16, bias:float16] -> outputs [output:float16] */
int32_t hct_ref_transpose_conv_f16(const HctConvFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8, filter:int8, bias:int32, multiplier:int32, shift:int32] -> outputs [output:int8] */
int32_t hct_ref_fully_connected_s8(const HctFullyConnectedParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16, filter:int8, bias:int64, multiplier:int32, shift:int32] -> outputs [output:int16] */
int32_t hct_ref_fully_connected_s16(const HctFullyConnectedParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32, filter:float32, bias:float32] -> outputs [output:float32] */
int32_t hct_ref_fully_connected_f32(const HctFloatActivationParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16, filter:float16, bias:float16] -> outputs [output:float16] */
int32_t hct_ref_fully_connected_f16(const HctFloatActivationParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [lhs:int8, rhs:int8] -> outputs [output:int8] */
int32_t hct_ref_batch_matmul_s8(const HctBatchMatMulParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [lhs:int16, rhs:int16] -> outputs [output:int16] */
int32_t hct_ref_batch_matmul_s16(const HctBatchMatMulParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [lhs:float32, rhs:float32] -> outputs [output:float32] */
int32_t hct_ref_batch_matmul_f32(const HctBatchMatMulFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [lhs:float16, rhs:float16] -> outputs [output:float16] */
int32_t hct_ref_batch_matmul_f16(const HctBatchMatMulFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_avg_pool_s8(const HctPoolParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_avg_pool_s16(const HctPoolParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_avg_pool_f32(const HctPoolFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_avg_pool_f16(const HctPoolFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_max_pool_s8(const HctPoolParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_max_pool_s16(const HctPoolParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_max_pool_f32(const HctPoolFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_max_pool_f16(const HctPoolFloatParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_mean_s8(const HctMeanQuantizedParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_mean_s16(const HctMeanQuantizedParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_mean_f32(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_mean_f16(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_reduce_sum_f32(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_reduce_sum_f16(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_reduce_max_f32(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_reduce_max_f16(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:float32] */
int32_t hct_ref_reduce_min_f32(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:float16] */
int32_t hct_ref_reduce_min_f16(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_reduce_max_s8(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_reduce_max_s16(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int8] */
int32_t hct_ref_reduce_min_s8(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int16] */
int32_t hct_ref_reduce_min_s16(const HctAxisMaskParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int32] */
int32_t hct_ref_arg_max_s8(const HctAxisParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int32] */
int32_t hct_ref_arg_max_s16(const HctAxisParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:int32] */
int32_t hct_ref_arg_max_f32(const HctAxisParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:int32] */
int32_t hct_ref_arg_max_f16(const HctAxisParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int8] -> outputs [output:int32] */
int32_t hct_ref_arg_min_s8(const HctAxisParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:int16] -> outputs [output:int32] */
int32_t hct_ref_arg_min_s16(const HctAxisParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float32] -> outputs [output:int32] */
int32_t hct_ref_arg_min_f32(const HctAxisParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input:float16] -> outputs [output:int32] */
int32_t hct_ref_arg_min_f16(const HctAxisParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif /* HCT_REF_ABI_H */
