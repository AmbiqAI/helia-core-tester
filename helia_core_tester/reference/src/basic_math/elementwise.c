/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Elementwise Mul, Maximum, Minimum, SquaredDifference, Comparison, Abs and
 * Clamp, as TFLite's reference kernels define them. Binary ops broadcast
 * numpy-style.
 */
#include <math.h>

#include "hct_ref_internal.h"

static int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Zero points of int8 tensors lie in the int8 range; int16 tensors are symmetric. */
static int32_t check_zero_points(int32_t dtype, const int32_t *zps, int n)
{
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    for (int i = 0; i < n; ++i)
    {
        if (dtype == HCT_INT16 ? zps[i] != 0 : (zps[i] < qmin || zps[i] > qmax))
        {
            return HCT_E_PARAM;
        }
    }
    return HCT_OK;
}

static int32_t check_scales(const float *scales, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (!(isfinite(scales[i]) && scales[i] > 0.0f))
        {
            return HCT_E_PARAM;
        }
    }
    return HCT_OK;
}

static int32_t check_activation(int32_t dtype, int32_t lo, int32_t hi)
{
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    return (lo <= hi && lo >= qmin && hi <= qmax) ? HCT_OK : HCT_E_PARAM;
}

static int32_t check_shift(int32_t multiplier, int32_t shift, int32_t max_shift)
{
    return (multiplier >= 0 && shift >= -31 && shift <= max_shift) ? HCT_OK : HCT_E_PARAM;
}

/* ---------------------------------------------------------------- Mul */

int32_t hct_ref_mul_prepare(const HctMulQuant *in, HctMulParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    const float scales[3] = {in->input1_scale, in->input2_scale, in->output_scale};
    const int32_t zps[3] = {in->input1_zero_point, in->input2_zero_point, in->output_zero_point};
    HCT_TRY(check_zero_points(in->dtype, zps, 3));
    HCT_TRY(check_scales(scales, 3));
    out->input1_offset = -in->input1_zero_point;
    out->input2_offset = -in->input2_zero_point;
    out->output_offset = in->output_zero_point;
    const double real = (double)in->input1_scale * (double)in->input2_scale / (double)in->output_scale;
    HCT_TRY(hct_quantize_multiplier_impl(real, &out->output_multiplier, &out->output_shift));
    return hct_activation_range_quantized(in->activation, in->dtype, in->output_scale, in->output_zero_point,
                                          &out->activation_min, &out->activation_max);
}

static int32_t mul_quantized(const HctMulParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                             int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HctBroadcast2 bc;
    HCT_TRY(hct_binary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &bc));
    const int32_t zps[3] = {-p->input1_offset, -p->input2_offset, p->output_offset};
    HCT_TRY(check_zero_points(dtype, zps, 3));
    HCT_TRY(check_shift(p->output_multiplier, p->output_shift, 30));
    HCT_TRY(check_activation(dtype, p->activation_min, p->activation_max));
    for (int64_t i = 0; i < bc.count; ++i)
    {
        int64_t o1 = 0;
        int64_t o2 = 0;
        hct_broadcast2_offsets(&bc, i, &o1, &o2);
        const int32_t product =
            (p->input1_offset + hct_load_i32(&inputs[0], o1)) * (p->input2_offset + hct_load_i32(&inputs[1], o2));
        const int32_t out =
            p->output_offset + hct_multiply_by_quantized_multiplier(product, p->output_multiplier, p->output_shift);
        hct_store_i32(&outputs[0], i, clamp_i32(out, p->activation_min, p->activation_max));
    }
    return HCT_OK;
}

static int32_t mul_float(const HctFloatActivationParams *p, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HctBroadcast2 bc;
    HCT_TRY(hct_binary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &bc));
    HCT_TRY(hct_check_float_activation(p->activation_min, p->activation_max));
    for (int64_t i = 0; i < bc.count; ++i)
    {
        int64_t o1 = 0;
        int64_t o2 = 0;
        hct_broadcast2_offsets(&bc, i, &o1, &o2);
        const float product = hct_load_f32(&inputs[0], o1) * hct_load_f32(&inputs[1], o2);
        hct_store_f32(&outputs[0], i, hct_clamp_f32(product, p->activation_min, p->activation_max));
    }
    return HCT_OK;
}

int32_t hct_ref_mul_s8(const HctMulParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                       int32_t num_outputs)
{
    return mul_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_mul_s16(const HctMulParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs)
{
    return mul_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

int32_t hct_ref_mul_f32(const HctFloatActivationParams *params, const HctTensor *inputs, int32_t num_inputs,
                        HctTensor *outputs, int32_t num_outputs)
{
    return mul_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_mul_f16(const HctFloatActivationParams *params, const HctTensor *inputs, int32_t num_inputs,
                        HctTensor *outputs, int32_t num_outputs)
{
    return mul_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}

/* ------------------------------------------------------- Maximum / Minimum */

/* Quantized operands share one quantization, so the codes compare directly. Floats
 * follow IEEE 754-2019 maximum/minimum: a NaN operand propagates (the first if
 * both), and -0 orders below +0. */
static int32_t min_max(const HctNoParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                       int32_t num_outputs, int32_t dtype, int is_max)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HctBroadcast2 bc;
    HCT_TRY(hct_binary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &bc));
    const int is_float = dtype == HCT_FLOAT32 || dtype == HCT_FLOAT16;
    for (int64_t i = 0; i < bc.count; ++i)
    {
        int64_t o1 = 0;
        int64_t o2 = 0;
        hct_broadcast2_offsets(&bc, i, &o1, &o2);
        if (is_float)
        {
            const float a = hct_load_f32(&inputs[0], o1);
            const float b = hct_load_f32(&inputs[1], o2);
            int take_a = 0;
            if (isnan(a) || isnan(b))
            {
                take_a = isnan(a);
            }
            else if (a == b)
            {
                /* equal values differ at most in the sign of zero */
                take_a = is_max ? !signbit(a) : signbit(a);
            }
            else
            {
                take_a = is_max ? a > b : a < b;
            }
            hct_store_f32(&outputs[0], i, take_a ? a : b);
        }
        else
        {
            const int32_t a = hct_load_i32(&inputs[0], o1);
            const int32_t b = hct_load_i32(&inputs[1], o2);
            hct_store_i32(&outputs[0], i, is_max ? (a >= b ? a : b) : (a <= b ? a : b));
        }
    }
    return HCT_OK;
}

#define HCT_MIN_MAX_ENTRY(op, suffix, dtype, is_max)                                                                   \
    int32_t hct_ref_##op##_##suffix(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,           \
                                    HctTensor *outputs, int32_t num_outputs)                                           \
    {                                                                                                                  \
        return min_max(params, inputs, num_inputs, outputs, num_outputs, dtype, is_max);                               \
    }

HCT_MIN_MAX_ENTRY(maximum, s8, HCT_INT8, 1)
HCT_MIN_MAX_ENTRY(maximum, s16, HCT_INT16, 1)
HCT_MIN_MAX_ENTRY(maximum, f32, HCT_FLOAT32, 1)
HCT_MIN_MAX_ENTRY(maximum, f16, HCT_FLOAT16, 1)
HCT_MIN_MAX_ENTRY(minimum, s8, HCT_INT8, 0)
HCT_MIN_MAX_ENTRY(minimum, s16, HCT_INT16, 0)
HCT_MIN_MAX_ENTRY(minimum, f32, HCT_FLOAT32, 0)
HCT_MIN_MAX_ENTRY(minimum, f16, HCT_FLOAT16, 0)

/* ------------------------------------------------------- SquaredDifference */

int32_t hct_ref_squared_difference_prepare(const HctAddQuant *in, HctSquaredDifferenceParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    const float scales[3] = {in->input1_scale, in->input2_scale, in->output_scale};
    const int32_t zps[3] = {in->input1_zero_point, in->input2_zero_point, in->output_zero_point};
    HCT_TRY(check_zero_points(in->dtype, zps, 3));
    HCT_TRY(check_scales(scales, 3));
    out->left_shift = in->dtype == HCT_INT16 ? 0 : 7;
    out->input1_offset = -in->input1_zero_point;
    out->input2_offset = -in->input2_zero_point;
    out->output_offset = in->output_zero_point;
    const double s1 = (double)in->input1_scale;
    const double s2 = (double)in->input2_scale;
    const double twice_max = 2.0 * (s1 > s2 ? s1 : s2);
    const double real_output =
        twice_max * twice_max / ((double)(1 << (out->left_shift * 2)) * (double)in->output_scale);
    HCT_TRY(hct_quantize_multiplier_smaller_than_one(s1 / twice_max, &out->input1_multiplier, &out->input1_shift));
    HCT_TRY(hct_quantize_multiplier_smaller_than_one(s2 / twice_max, &out->input2_multiplier, &out->input2_shift));
    HCT_TRY(hct_quantize_multiplier_impl(real_output, &out->output_multiplier, &out->output_shift));
    return hct_activation_range_quantized(in->activation, in->dtype, in->output_scale, in->output_zero_point,
                                          &out->activation_min, &out->activation_max);
}

static int32_t squared_difference_quantized(const HctSquaredDifferenceParams *p, const HctTensor *inputs,
                                            int32_t num_inputs, HctTensor *outputs, int32_t num_outputs,
                                            int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HctBroadcast2 bc;
    HCT_TRY(hct_binary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &bc));
    const int32_t zps[3] = {-p->input1_offset, -p->input2_offset, p->output_offset};
    HCT_TRY(check_zero_points(dtype, zps, 3));
    if (p->left_shift != (dtype == HCT_INT16 ? 0 : 7))
    {
        return HCT_E_PARAM;
    }
    HCT_TRY(check_shift(p->input1_multiplier, p->input1_shift, 0));
    HCT_TRY(check_shift(p->input2_multiplier, p->input2_shift, 0));
    HCT_TRY(check_shift(p->output_multiplier, p->output_shift, 30));
    HCT_TRY(check_activation(dtype, p->activation_min, p->activation_max));
    for (int64_t i = 0; i < bc.count; ++i)
    {
        int64_t o1 = 0;
        int64_t o2 = 0;
        hct_broadcast2_offsets(&bc, i, &o1, &o2);
        const int32_t shifted1 = (p->input1_offset + hct_load_i32(&inputs[0], o1)) * (1 << p->left_shift);
        const int32_t shifted2 = (p->input2_offset + hct_load_i32(&inputs[1], o2)) * (1 << p->left_shift);
        const int32_t scaled1 = hct_multiply_by_quantized_multiplier(shifted1, p->input1_multiplier, p->input1_shift);
        const int32_t scaled2 = hct_multiply_by_quantized_multiplier(shifted2, p->input2_multiplier, p->input2_shift);
        const int32_t raw_diff = scaled1 - scaled2;
        const int32_t out =
            hct_multiply_by_quantized_multiplier(raw_diff * raw_diff, p->output_multiplier, p->output_shift) +
            p->output_offset;
        hct_store_i32(&outputs[0], i, clamp_i32(out, p->activation_min, p->activation_max));
    }
    return HCT_OK;
}

int32_t hct_ref_squared_difference_s8(const HctSquaredDifferenceParams *params, const HctTensor *inputs,
                                      int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return squared_difference_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_squared_difference_s16(const HctSquaredDifferenceParams *params, const HctTensor *inputs,
                                       int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return squared_difference_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

/* (a - b) * (a - b) in binary16 arithmetic: each operation is one IEEE binary16 result. */
int32_t hct_ref_squared_difference_f16(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                                       HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HctBroadcast2 bc;
    HCT_TRY(hct_binary_setup(inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16, HCT_FLOAT16, &bc));
    for (int64_t i = 0; i < bc.count; ++i)
    {
        int64_t o1 = 0;
        int64_t o2 = 0;
        hct_broadcast2_offsets(&bc, i, &o1, &o2);
        const float diff = hct_round_f16(hct_load_f32(&inputs[0], o1) - hct_load_f32(&inputs[1], o2));
        hct_store_f32(&outputs[0], i, diff * diff);
    }
    return HCT_OK;
}

/* -------------------------------------------------------------- Comparison */

int32_t hct_ref_comparison_prepare(const HctComparisonQuant *in, HctComparisonParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(in->dtype, &qmin, &qmax));
    const float scales[2] = {in->input1_scale, in->input2_scale};
    HCT_TRY(check_scales(scales, 2));
    if (in->input1_zero_point < qmin || in->input1_zero_point > qmax || in->input2_zero_point < qmin ||
        in->input2_zero_point > qmax || in->operation < HCT_CMP_EQUAL || in->operation > HCT_CMP_LESS_EQUAL)
    {
        return HCT_E_PARAM;
    }
    out->operation = in->operation;
    out->left_shift = 8;
    out->input1_offset = -in->input1_zero_point;
    out->input2_offset = -in->input2_zero_point;
    HCT_TRY(hct_quantize_multiplier_smaller_than_one((double)in->input1_scale, &out->input1_multiplier,
                                                     &out->input1_shift));
    return hct_quantize_multiplier_smaller_than_one((double)in->input2_scale, &out->input2_multiplier,
                                                    &out->input2_shift);
}

static int32_t comparison(const HctComparisonParams *p, const HctTensor *inputs, int32_t num_inputs,
                          HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HctBroadcast2 bc;
    HCT_TRY(hct_binary_setup(inputs, num_inputs, outputs, num_outputs, dtype, HCT_BOOL, &bc));
    if (p->operation < HCT_CMP_EQUAL || p->operation > HCT_CMP_LESS_EQUAL || p->left_shift != 8)
    {
        return HCT_E_PARAM;
    }
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (-p->input1_offset < qmin || -p->input1_offset > qmax || -p->input2_offset < qmin || -p->input2_offset > qmax)
    {
        return HCT_E_PARAM;
    }
    HCT_TRY(check_shift(p->input1_multiplier, p->input1_shift, 0));
    HCT_TRY(check_shift(p->input2_multiplier, p->input2_shift, 0));
    for (int64_t i = 0; i < bc.count; ++i)
    {
        int64_t o1 = 0;
        int64_t o2 = 0;
        hct_broadcast2_offsets(&bc, i, &o1, &o2);
        const int32_t a = hct_multiply_by_quantized_multiplier(
            (p->input1_offset + hct_load_i32(&inputs[0], o1)) * (1 << p->left_shift), p->input1_multiplier,
            p->input1_shift);
        const int32_t b = hct_multiply_by_quantized_multiplier(
            (p->input2_offset + hct_load_i32(&inputs[1], o2)) * (1 << p->left_shift), p->input2_multiplier,
            p->input2_shift);
        int32_t result = 0;
        switch (p->operation)
        {
        case HCT_CMP_EQUAL:
            result = a == b;
            break;
        case HCT_CMP_NOT_EQUAL:
            result = a != b;
            break;
        case HCT_CMP_GREATER:
            result = a > b;
            break;
        case HCT_CMP_GREATER_EQUAL:
            result = a >= b;
            break;
        case HCT_CMP_LESS:
            result = a < b;
            break;
        default:
            result = a <= b;
            break;
        }
        hct_store_i32(&outputs[0], i, result);
    }
    return HCT_OK;
}

int32_t hct_ref_comparison_s8(const HctComparisonParams *params, const HctTensor *inputs, int32_t num_inputs,
                              HctTensor *outputs, int32_t num_outputs)
{
    return comparison(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_comparison_s16(const HctComparisonParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    return comparison(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

/* --------------------------------------------------------------------- Abs */

int32_t hct_ref_abs_prepare(const HctAbsQuant *in, HctAbsParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    const float scales[2] = {in->input_scale, in->output_scale};
    const int32_t zps[2] = {in->input_zero_point, in->output_zero_point};
    HCT_TRY(check_zero_points(in->dtype, zps, 2));
    HCT_TRY(check_scales(scales, 2));
    out->input_zero_point = in->input_zero_point;
    out->output_zero_point = in->output_zero_point;
    out->needs_rescale = in->input_scale != in->output_scale;
    /* TFLite forms the ratio in float. */
    const float scale = in->input_scale / in->output_scale;
    return hct_quantize_multiplier_impl((double)scale, &out->multiplier, &out->shift);
}

static int32_t abs_quantized(const HctAbsParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                             int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    const int32_t zps[2] = {p->input_zero_point, p->output_zero_point};
    HCT_TRY(check_zero_points(dtype, zps, 2));
    if (p->needs_rescale != 0 && p->needs_rescale != 1)
    {
        return HCT_E_PARAM;
    }
    HCT_TRY(check_shift(p->multiplier, p->shift, 30));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    for (int64_t i = 0; i < count; ++i)
    {
        int32_t value = hct_load_i32(&inputs[0], i) - p->input_zero_point;
        value = value < 0 ? -value : value;
        const int32_t out =
            (p->needs_rescale ? hct_multiply_by_quantized_multiplier(value, p->multiplier, p->shift) : value) +
            p->output_zero_point;
        hct_store_i32(&outputs[0], i, clamp_i32(out, qmin, qmax));
    }
    return HCT_OK;
}

static int32_t abs_float(const HctNoParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                         int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    for (int64_t i = 0; i < count; ++i)
    {
        hct_store_f32(&outputs[0], i, fabsf(hct_load_f32(&inputs[0], i)));
    }
    return HCT_OK;
}

int32_t hct_ref_abs_s8(const HctAbsParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                       int32_t num_outputs)
{
    return abs_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_abs_s16(const HctAbsParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs)
{
    return abs_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

int32_t hct_ref_abs_f32(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs)
{
    return abs_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_abs_f16(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs)
{
    return abs_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}

/* ------------------------------------------------------------------- Clamp */

static int32_t clamp_quantized(const HctClampParams *p, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    HCT_TRY(check_activation(dtype, p->activation_min, p->activation_max));
    for (int64_t i = 0; i < count; ++i)
    {
        hct_store_i32(&outputs[0], i, clamp_i32(hct_load_i32(&inputs[0], i), p->activation_min, p->activation_max));
    }
    return HCT_OK;
}

int32_t hct_ref_clamp_s8(const HctClampParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return clamp_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_clamp_s16(const HctClampParams *params, const HctTensor *inputs, int32_t num_inputs,
                          HctTensor *outputs, int32_t num_outputs)
{
    return clamp_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}
