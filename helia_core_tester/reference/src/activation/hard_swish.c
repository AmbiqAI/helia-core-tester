/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * HardSwish, x * relu6(x + 3) / 6.
 *   hard_swish_s8:          TFLite's reference_ops::HardSwish<int8_t> with HardSwishPrepare's params.
 *   hard_swish_precise_*:   CMSIS-NN's int32 variant (no TFLite counterpart), named rather than toleranced.
 *   hard_swish_f32/f16:     the exact result rounded once to the output type.
 */
#include <math.h>

#include "hct_ref_internal.h"

static int32_t check_quant(const HctHardSwishQuant *in)
{
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(in->dtype, &qmin, &qmax));
    if (!(isfinite(in->input_scale) && in->input_scale > 0.0f && isfinite(in->output_scale) &&
          in->output_scale > 0.0f))
    {
        return HCT_E_PARAM;
    }
    if (in->dtype == HCT_INT16 ? (in->input_zero_point != 0 || in->output_zero_point != 0)
                               : (in->input_zero_point < qmin || in->input_zero_point > qmax ||
                                  in->output_zero_point < qmin || in->output_zero_point > qmax))
    {
        return HCT_E_PARAM;
    }
    return HCT_OK;
}

static int32_t check_offsets(int32_t dtype, int32_t in_offset, int32_t out_offset)
{
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (dtype == HCT_INT16 ? (in_offset != 0 || out_offset != 0)
                           : (in_offset < qmin || in_offset > qmax || out_offset < qmin || out_offset > qmax))
    {
        return HCT_E_PARAM;
    }
    return HCT_OK;
}

static int32_t clamp_i32(int64_t v, int32_t lo, int32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : (int32_t)v);
}

/* ------------------------------------------------------- TFLite (int8) */

/* The int16 exponents stay within what gemmlowp's int16 shifts are defined for. */
static int32_t check_params(const HctHardSwishParams *p)
{
    HCT_TRY(check_offsets(HCT_INT8, p->input_zero_point, p->output_zero_point));
    if (p->output_multiplier_fixedpoint_int16 < 0 || p->output_multiplier_fixedpoint_int16 > INT16_MAX ||
        p->reluish_multiplier_fixedpoint_int16 < 0 || p->reluish_multiplier_fixedpoint_int16 > INT16_MAX ||
        p->output_multiplier_exponent > 0 || p->output_multiplier_exponent < -15 ||
        p->reluish_multiplier_exponent < -15 || p->reluish_multiplier_exponent > 15)
    {
        return HCT_E_PARAM;
    }
    return HCT_OK;
}

/* QuantizeMultiplier then DownScaleInt32ToInt16Multiplier. */
static int32_t q15_multiplier(float real, int32_t *fixedpoint, int32_t *exponent)
{
    int32_t q31 = 0;
    HCT_TRY(hct_quantize_multiplier_impl((double)real, &q31, exponent));
    *fixedpoint = q31 >= INT32_MAX - (1 << 15) ? INT16_MAX : (q31 + (1 << 15)) >> 16;
    return HCT_OK;
}

int32_t hct_ref_hard_swish_prepare(const HctHardSwishQuant *in, HctHardSwishParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    if (in->dtype != HCT_INT8)
    {
        return HCT_E_DTYPE;
    }
    HCT_TRY(check_quant(in));
    /* As HardSwishPrepare: every ratio is formed in float. */
    const float hires_input_scale = (1.0f / 128.0f) * in->input_scale;
    const float reluish_scale = 3.0f / 32768.0f;
    out->input_zero_point = in->input_zero_point;
    out->output_zero_point = in->output_zero_point;
    HCT_TRY(q15_multiplier(hires_input_scale / in->output_scale, &out->output_multiplier_fixedpoint_int16,
                           &out->output_multiplier_exponent));
    HCT_TRY(q15_multiplier(hires_input_scale / reluish_scale, &out->reluish_multiplier_fixedpoint_int16,
                           &out->reluish_multiplier_exponent));
    return check_params(out);
}

static int16_t srdhm16(int16_t a, int16_t b)
{
    if (a == INT16_MIN && b == INT16_MIN)
    {
        return INT16_MAX;
    }
    const int32_t ab = (int32_t)a * (int32_t)b;
    const int32_t nudge = ab >= 0 ? (1 << 14) : (1 - (1 << 14));
    return (int16_t)((ab + nudge) / (1 << 15));
}

static int16_t sdhm16(int16_t a, int16_t b)
{
    if (a == INT16_MIN && b == INT16_MIN)
    {
        return INT16_MAX;
    }
    return (int16_t)(((int32_t)a * (int32_t)b) / (1 << 15));
}

static int16_t saturating_left_shift16(int16_t v, int32_t amount)
{
    return (int16_t)clamp_i32((int64_t)v * (1LL << amount), INT16_MIN, INT16_MAX);
}

int32_t hct_ref_hard_swish_s8(const HctHardSwishParams *params, const HctTensor *inputs, int32_t num_inputs,
                              HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, HCT_INT8, HCT_INT8, &count));
    HCT_TRY(check_params(params));
    const HctHardSwishParams *p = params;
    for (int64_t i = 0; i < count; ++i)
    {
        const int16_t x = (int16_t)(hct_load_i32(&inputs[0], i) - p->input_zero_point);
        const int16_t hires = (int16_t)(x * (1 << 7));
        const int16_t preshift = srdhm16(hires, (int16_t)p->output_multiplier_fixedpoint_int16);
        int16_t reluish = hires;
        if (p->reluish_multiplier_exponent > 0)
        {
            reluish = saturating_left_shift16(reluish, p->reluish_multiplier_exponent - 1);
        }
        reluish = srdhm16(reluish, (int16_t)p->reluish_multiplier_fixedpoint_int16);
        if (p->reluish_multiplier_exponent > 0)
        {
            reluish = saturating_left_shift16(reluish, 1);
        }
        if (p->reluish_multiplier_exponent < 0)
        {
            reluish = (int16_t)hct_rounding_divide_by_pot(reluish, -p->reluish_multiplier_exponent);
        }
        reluish = (int16_t)((reluish + (1 << 15)) >> 1);
        const int16_t y = sdhm16(reluish, preshift);
        const int32_t v = hct_rounding_divide_by_pot(y, -p->output_multiplier_exponent) + p->output_zero_point;
        hct_store_i32(&outputs[0], i, clamp_i32(v, INT8_MIN, INT8_MAX));
    }
    return HCT_OK;
}

/* ----------------------------------------------------- CMSIS-NN precise */

#define HCT_HARD_SWISH_MAX_ABS_X 65535

int32_t hct_ref_hard_swish_precise_prepare(const HctHardSwishQuant *in, HctHardSwishPreciseParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(check_quant(in));
    const double s_in = (double)in->input_scale;
    const double q3 = floor(3.0 / s_in + 0.5);
    const double q6 = floor(6.0 / s_in + 0.5);
    if (q6 > (double)(INT32_MAX - HCT_HARD_SWISH_MAX_ABS_X))
    {
        return HCT_E_PARAM;
    }
    int32_t prescale = 0;
    for (int64_t prod_max = 32767LL * (int64_t)q6; prod_max > INT32_MAX; prod_max >>= 1)
    {
        ++prescale;
    }
    out->input_offset = in->input_zero_point;
    out->output_offset = in->output_zero_point;
    out->relu_q3 = (int32_t)q3;
    out->relu_q6 = (int32_t)q6;
    out->prescale = prescale;
    const double real = s_in * s_in / (6.0 * (double)in->output_scale) * (double)(1LL << prescale);
    return hct_quantize_multiplier_impl(real, &out->output_multiplier, &out->output_shift);
}

static int32_t hard_swish_precise(const HctHardSwishPreciseParams *p, const HctTensor *inputs, int32_t num_inputs,
                                  HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    HCT_TRY(check_offsets(dtype, p->input_offset, p->output_offset));
    if (p->relu_q3 < 0 || p->relu_q6 < p->relu_q3 || p->relu_q6 > INT32_MAX - HCT_HARD_SWISH_MAX_ABS_X ||
        p->prescale < 0 || p->prescale > 31 || p->output_multiplier < 0 || p->output_shift < -31 ||
        p->output_shift > 30)
    {
        return HCT_E_PARAM;
    }
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    for (int64_t i = 0; i < count; ++i)
    {
        const int32_t x = hct_load_i32(&inputs[0], i) - p->input_offset;
        int64_t xr = clamp_i32((int64_t)x + p->relu_q3, 0, p->relu_q6);
        if (p->prescale > 0)
        {
            xr = (xr + (1LL << (p->prescale - 1))) >> p->prescale;
        }
        /* The kernel's int32 product; prepare's prescale keeps it in range. */
        const int32_t y = (int32_t)(uint32_t)((uint64_t)(int64_t)x * (uint64_t)xr);
        const int64_t v = (int64_t)hct_cmsis_requantize(y, p->output_multiplier, p->output_shift) + p->output_offset;
        hct_store_i32(&outputs[0], i, clamp_i32(v, qmin, qmax));
    }
    return HCT_OK;
}

int32_t hct_ref_hard_swish_precise_s8(const HctHardSwishPreciseParams *params, const HctTensor *inputs,
                                      int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return hard_swish_precise(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_hard_swish_precise_s16(const HctHardSwishPreciseParams *params, const HctTensor *inputs,
                                       int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return hard_swish_precise(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

/* ------------------------------------------------------------------ float */

/* x + 3 and x * r are exact in binary64 for binary32/16 x, so the division's one
 * rounding, then one correctly rounded narrowing (53 >= 2*24 + 2 for binary32), yields
 * the exact result rounded once to the output type. NaN propagates; -inf gives NaN (-inf * 0). */
static int32_t hard_swish_float(const HctNoParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
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
        const double x = (double)hct_load_f32(&inputs[0], i);
        double r = x + 3.0;
        r = r < 0.0 ? 0.0 : (r > 6.0 ? 6.0 : r);
        const double y = x * r / 6.0;
        if (dtype == HCT_FLOAT16)
        {
            ((uint16_t *)outputs[0].data)[i] = hct_f64_to_f16(y);
        }
        else
        {
            ((float *)outputs[0].data)[i] = (float)y;
        }
    }
    return HCT_OK;
}

int32_t hct_ref_hard_swish_f32(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    return hard_swish_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_hard_swish_f16(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    return hard_swish_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
