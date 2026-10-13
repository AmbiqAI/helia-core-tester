/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Quantize (TFLite AffineQuantize), Dequantize (TFLite Dequantize, and the exact binary16
 * widening, NaNs quieted) with a float clamp for a fused activation, and CMSIS-NN's
 * arm_requantize_* (a named variant: TFLite has no counterpart with its rounding).
 */
#include <math.h>
#include <string.h>

#include "hct_ref_internal.h"

static int32_t check_params(const HctQuantizeParams *p, int32_t dtype)
{
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (!(isfinite(p->scale) && p->scale > 0.0f) || p->zero_point < qmin || p->zero_point > qmax ||
        (dtype == HCT_INT16 && p->zero_point != 0))
    {
        return HCT_E_PARAM;
    }
    return hct_check_float_activation(p->activation_min, p->activation_max);
}

/* --------------------------------------------------------------- quantize */

static int32_t quantize(const HctQuantizeParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs, int32_t out_dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32, out_dtype, &count));
    HCT_TRY(check_params(p, out_dtype));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(out_dtype, &qmin, &qmax));
    /* The float-to-int conversion is undefined for a non-finite value (and host-dependent
     * in practice), so a value the clamp leaves non-finite is refused, before writing. */
    for (int64_t i = 0; i < count; ++i)
    {
        if (!isfinite(hct_clamp_f32(hct_load_f32(&inputs[0], i), p->activation_min, p->activation_max)))
        {
            return HCT_E_PARAM;
        }
    }
    for (int64_t i = 0; i < count; ++i)
    {
        const float val = hct_clamp_f32(hct_load_f32(&inputs[0], i), p->activation_min, p->activation_max);
        const float r = roundf(val / p->scale);
        /* Saturated in float first: past int32 the C conversion would be undefined. */
        const int64_t q = r >= 2147483648.0f ? INT32_MAX : (r < -2147483648.0f ? INT32_MIN : (int64_t)r);
        const int64_t v = q + p->zero_point;
        hct_store_i32(&outputs[0], i, v < qmin ? qmin : (v > qmax ? qmax : (int32_t)v));
    }
    return HCT_OK;
}

int32_t hct_ref_quantize_f32_s8(const HctQuantizeParams *params, const HctTensor *inputs, int32_t num_inputs,
                                HctTensor *outputs, int32_t num_outputs)
{
    return quantize(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_quantize_f32_s16(const HctQuantizeParams *params, const HctTensor *inputs, int32_t num_inputs,
                                 HctTensor *outputs, int32_t num_outputs)
{
    return quantize(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

/* ------------------------------------------------------------- dequantize */

static int32_t dequantize(const HctQuantizeParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                          int32_t num_outputs, int32_t in_dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, in_dtype, HCT_FLOAT32, &count));
    HCT_TRY(check_params(p, in_dtype));
    const double scale = (double)p->scale;
    for (int64_t i = 0; i < count; ++i)
    {
        const float result = (float)(scale * (double)(hct_load_i32(&inputs[0], i) - p->zero_point));
        hct_store_f32(&outputs[0], i, hct_clamp_f32(result, p->activation_min, p->activation_max));
    }
    return HCT_OK;
}

int32_t hct_ref_dequantize_s8_f32(const HctQuantizeParams *params, const HctTensor *inputs, int32_t num_inputs,
                                  HctTensor *outputs, int32_t num_outputs)
{
    return dequantize(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_dequantize_s16_f32(const HctQuantizeParams *params, const HctTensor *inputs, int32_t num_inputs,
                                   HctTensor *outputs, int32_t num_outputs)
{
    return dequantize(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

int32_t hct_ref_dequantize_f16_f32(const HctFloatActivationParams *params, const HctTensor *inputs,
                                   int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16, HCT_FLOAT32, &count));
    HCT_TRY(hct_check_float_activation(params->activation_min, params->activation_max));
    for (int64_t i = 0; i < count; ++i)
    {
        const uint16_t h = ((const uint16_t *)inputs[0].data)[i];
        if ((h & 0x7C00u) == 0x7C00u && (h & 0x3FFu) != 0u)
        {
            /* A NaN widens with its sign and payload and the quiet bit set (the IEEE conversion),
             * written as bits so no float move can alter a signaling pattern. */
            const uint32_t bits = ((uint32_t)(h & 0x8000u) << 16) | 0x7FC00000u | ((uint32_t)(h & 0x3FFu) << 13);
            memcpy(&((float *)outputs[0].data)[i], &bits, sizeof(bits));
            continue;
        }
        hct_store_f32(&outputs[0], i,
                      hct_clamp_f32(hct_f16_to_f32(h), params->activation_min, params->activation_max));
    }
    return HCT_OK;
}

/* ------------------------------------------------------------- requantize */

static int32_t requantize(const HctRequantizeParams *p, const HctTensor *inputs, int32_t num_inputs,
                          HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (p->multiplier < 0 || p->shift < -31 || p->shift > 30 || p->input_zero_point < qmin ||
        p->input_zero_point > qmax || p->output_zero_point < qmin || p->output_zero_point > qmax)
    {
        return HCT_E_PARAM;
    }
    for (int64_t i = 0; i < count; ++i)
    {
        const int64_t v = (int64_t)hct_cmsis_requantize(hct_load_i32(&inputs[0], i) - p->input_zero_point,
                                                        p->multiplier, p->shift) +
                          p->output_zero_point;
        hct_store_i32(&outputs[0], i, v < qmin ? qmin : (v > qmax ? qmax : (int32_t)v));
    }
    return HCT_OK;
}

int32_t hct_ref_requantize_s8(const HctRequantizeParams *params, const HctTensor *inputs, int32_t num_inputs,
                              HctTensor *outputs, int32_t num_outputs)
{
    return requantize(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_requantize_s16(const HctRequantizeParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    return requantize(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}
