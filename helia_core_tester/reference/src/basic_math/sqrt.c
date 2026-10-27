/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Sqrt and Rsqrt.
 *   sqrt_s8/s16:  TFLite's SqrtEvalQuantized: sqrtf of the dequantized value, truncated on requantizing.
 *   rsqrt_s16:    TFLite's int16 Rsqrt: LUTPopulate<int16_t> of 1 / sqrtf(x), then LUTLookup.
 *   (r)sqrt_f32/f16: the binary64 result rounded once, with the ns-cmsis-nn#295 special values:
 *                 a negative non-zero input gives the default quiet NaN, a NaN is quieted with its
 *                 sign and payload kept, +-0 gives +-0 (sqrt) or +-inf (rsqrt), +inf gives +inf or +0.
 */
#include <math.h>
#include <string.h>

#include "hct_ref_internal.h"

static int32_t check_quant(const HctUnaryQuantParams *p, int32_t dtype)
{
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (!(isfinite(p->input_scale) && p->input_scale > 0.0f && isfinite(p->output_scale) && p->output_scale > 0.0f))
    {
        return HCT_E_PARAM;
    }
    /* int16 tensors are symmetric (GenericPrepare). */
    if (dtype == HCT_INT16 ? (p->input_zero_point != 0 || p->output_zero_point != 0)
                           : (p->input_zero_point < qmin || p->input_zero_point > qmax ||
                              p->output_zero_point < qmin || p->output_zero_point > qmax))
    {
        return HCT_E_PARAM;
    }
    return HCT_OK;
}

/* --------------------------------------------------------- quantized sqrt */

static int32_t sqrt_quantized(const HctUnaryQuantParams *p, const HctTensor *inputs, int32_t num_inputs,
                              HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    HCT_TRY(check_quant(p, dtype));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    /* TFLite fails the whole op on a negative value; so does the reference, before writing. */
    for (int64_t i = 0; i < count; ++i)
    {
        if (p->input_scale * (float)(hct_load_i32(&inputs[0], i) - p->input_zero_point) < 0.0f)
        {
            return HCT_E_PARAM;
        }
    }
    for (int64_t i = 0; i < count; ++i)
    {
        const float dequantized = p->input_scale * (float)(hct_load_i32(&inputs[0], i) - p->input_zero_point);
        const float scaled = sqrtf(dequantized) / p->output_scale;
        /* static_cast<int> truncates; the clamp to the dtype absorbs anything past int32. */
        const int64_t truncated = scaled >= 2147483648.0f ? INT32_MAX : (int64_t)scaled;
        const int64_t v = truncated + p->output_zero_point;
        hct_store_i32(&outputs[0], i, v < qmin ? qmin : (v > qmax ? qmax : (int32_t)v));
    }
    return HCT_OK;
}

int32_t hct_ref_sqrt_s8(const HctUnaryQuantParams *params, const HctTensor *inputs, int32_t num_inputs,
                        HctTensor *outputs, int32_t num_outputs)
{
    return sqrt_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_sqrt_s16(const HctUnaryQuantParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return sqrt_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

/* -------------------------------------------------------- quantized rsqrt */

static float rsqrt_transform(float value, const void *params)
{
    if (value <= 0.0f)
    {
        return (float)INT16_MAX * *(const float *)params;
    }
    return 1.0f / sqrtf(value);
}

int32_t hct_ref_rsqrt_s16(const HctUnaryQuantParams *params, const HctTensor *inputs, int32_t num_inputs,
                          HctTensor *outputs, int32_t num_outputs)
{
    const HctUnaryQuantParams *p = params;
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, HCT_INT16, HCT_INT16, &count));
    HCT_TRY(check_quant(p, HCT_INT16));
    for (int64_t i = 0; i < count; ++i)
    {
        if (hct_load_i32(&inputs[0], i) < p->input_zero_point)
        {
            return HCT_E_PARAM; /* "Rsqrt is only defined for positive values" */
        }
    }
    int16_t lut[HCT_LUT_S16_SIZE];
    const float output_scale = p->output_scale;
    hct_lut_populate_s16(p->input_scale, p->input_zero_point, p->output_scale, p->output_zero_point, rsqrt_transform,
                         &output_scale, lut);
    for (int64_t i = 0; i < count; ++i)
    {
        hct_store_i32(&outputs[0], i, hct_lut_lookup_s16(hct_load_i32(&inputs[0], i), lut));
    }
    return HCT_OK;
}

/* ------------------------------------------------------------------ float */

static uint32_t special_bits(uint32_t bits, uint32_t result, int reciprocal, uint32_t sign, uint32_t inf,
                             uint32_t quiet)
{
    const uint32_t magnitude = bits & (sign - 1u);
    if ((bits & sign) != 0u && magnitude != 0u)
    {
        result = inf | quiet;
    }
    if (magnitude > inf)
    {
        result = bits | quiet;
    }
    if (magnitude == 0u)
    {
        result = bits | (reciprocal ? inf : 0u);
    }
    if (bits == inf)
    {
        result = reciprocal ? 0u : inf;
    }
    return result;
}

static int32_t sqrt_float(const HctNoParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                          int32_t num_outputs, int32_t dtype, int reciprocal)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    for (int64_t i = 0; i < count; ++i)
    {
        double r = sqrt((double)hct_load_f32(&inputs[0], i));
        if (reciprocal)
        {
            r = 1.0 / r;
        }
        if (dtype == HCT_FLOAT16)
        {
            const uint16_t bits = ((const uint16_t *)inputs[0].data)[i];
            ((uint16_t *)outputs[0].data)[i] =
                (uint16_t)special_bits(bits, hct_f64_to_f16(r), reciprocal, 0x8000u, 0x7C00u, 0x0200u);
        }
        else
        {
            uint32_t bits = 0;
            uint32_t out = 0;
            const float f = (float)r;
            memcpy(&bits, &((const float *)inputs[0].data)[i], sizeof(bits));
            memcpy(&out, &f, sizeof(out));
            out = special_bits(bits, out, reciprocal, 0x80000000u, 0x7F800000u, 0x00400000u);
            memcpy(&((float *)outputs[0].data)[i], &out, sizeof(out));
        }
    }
    return HCT_OK;
}

int32_t hct_ref_sqrt_f32(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                         int32_t num_outputs)
{
    return sqrt_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32, 0);
}

int32_t hct_ref_sqrt_f16(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                         int32_t num_outputs)
{
    return sqrt_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16, 0);
}

int32_t hct_ref_rsqrt_f32(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                          int32_t num_outputs)
{
    return sqrt_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32, 1);
}

int32_t hct_ref_rsqrt_f16(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                          int32_t num_outputs)
{
    return sqrt_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16, 1);
}
