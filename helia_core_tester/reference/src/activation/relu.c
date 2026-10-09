/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Relu/Relu6 (TFLite ReluQuantized) and LeakyRelu (TFLite QuantizeLeakyRelu),
 * with the parameters TFLM's prepare derives.
 */
#include <math.h>

#include "hct_ref_internal.h"

static int32_t check_quant(int32_t dtype, float in_scale, int32_t in_zp, float out_scale, int32_t out_zp)
{
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (!(isfinite(in_scale) && in_scale > 0.0f && isfinite(out_scale) && out_scale > 0.0f))
    {
        return HCT_E_PARAM;
    }
    if (dtype == HCT_INT16 ? (in_zp != 0 || out_zp != 0)
                           : (in_zp < qmin || in_zp > qmax || out_zp < qmin || out_zp > qmax))
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

static int32_t check_multiplier(int32_t multiplier, int32_t shift)
{
    return (multiplier >= 0 && shift >= -31 && shift <= 30) ? HCT_OK : HCT_E_PARAM;
}

/* ------------------------------------------------------------ Relu, Relu6 */

int32_t hct_ref_relu_prepare(const HctReluQuant *in, HctReluParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(check_quant(in->dtype, in->input_scale, in->input_zero_point, in->output_scale, in->output_zero_point));
    if (isnan(in->act_max) || in->act_max < 0.0f)
    {
        return HCT_E_PARAM;
    }
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(in->dtype, &qmin, &qmax));
    out->input_offset = in->input_zero_point;
    out->output_offset = in->output_zero_point;
    HCT_TRY(hct_quantize_multiplier_impl((double)in->input_scale / (double)in->output_scale, &out->output_multiplier,
                                         &out->output_shift));
    /* As TFLM: the bounds are quantized in float, zero_point + roundf(act / scale). */
    const int32_t lo = in->output_zero_point + (int32_t)roundf(0.0f / in->output_scale);
    out->activation_min = lo > qmin ? lo : qmin;
    if (isinf(in->act_max))
    {
        out->activation_max = qmax;
    }
    else
    {
        /* zero_point + roundf(act_max / scale), saturated before the int conversion. */
        const float hi = (float)in->output_zero_point + roundf(in->act_max / in->output_scale);
        out->activation_max = hi >= (float)qmax ? qmax : (int32_t)hi;
    }
    return HCT_OK;
}

static int32_t relu(const HctReluParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                    int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    HCT_TRY(check_offsets(dtype, p->input_offset, p->output_offset));
    HCT_TRY(check_multiplier(p->output_multiplier, p->output_shift));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (p->activation_min > p->activation_max || p->activation_min < qmin || p->activation_max > qmax)
    {
        return HCT_E_PARAM;
    }
    for (int64_t i = 0; i < count; ++i)
    {
        int32_t v = p->output_offset + hct_multiply_by_quantized_multiplier(hct_load_i32(&inputs[0], i) - p->input_offset,
                                                                            p->output_multiplier, p->output_shift);
        v = v < p->activation_min ? p->activation_min : (v > p->activation_max ? p->activation_max : v);
        hct_store_i32(&outputs[0], i, v);
    }
    return HCT_OK;
}

int32_t hct_ref_relu_s8(const HctReluParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs)
{
    return relu(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_relu_s16(const HctReluParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                         int32_t num_outputs)
{
    return relu(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

/* -------------------------------------------------------------- LeakyRelu */

int32_t hct_ref_leaky_relu_prepare(const HctLeakyReluQuant *in, HctLeakyReluParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(check_quant(in->dtype, in->input_scale, in->input_zero_point, in->output_scale, in->output_zero_point));
    if (!isfinite(in->alpha))
    {
        return HCT_E_PARAM;
    }
    out->input_offset = in->input_zero_point;
    out->output_offset = in->output_zero_point;
    /* TFLM forms both ratios in float before widening them. */
    const float alpha_ratio = in->input_scale * in->alpha / in->output_scale;
    const float identity_ratio = in->input_scale / in->output_scale;
    if (alpha_ratio < 0.0f)
    {
        return HCT_E_PARAM; /* QuantizeMultiplier takes a non-negative real */
    }
    HCT_TRY(hct_quantize_multiplier_impl((double)alpha_ratio, &out->alpha_multiplier, &out->alpha_shift));
    return hct_quantize_multiplier_impl((double)identity_ratio, &out->identity_multiplier, &out->identity_shift);
}

static int32_t leaky_relu(const HctLeakyReluParams *p, const HctTensor *inputs, int32_t num_inputs,
                          HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    HCT_TRY(check_offsets(dtype, p->input_offset, p->output_offset));
    HCT_TRY(check_multiplier(p->alpha_multiplier, p->alpha_shift));
    HCT_TRY(check_multiplier(p->identity_multiplier, p->identity_shift));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    for (int64_t i = 0; i < count; ++i)
    {
        const int32_t x = hct_load_i32(&inputs[0], i) - p->input_offset;
        const int32_t scaled =
            x >= 0 ? hct_multiply_by_quantized_multiplier(x, p->identity_multiplier, p->identity_shift)
                   : hct_multiply_by_quantized_multiplier(x, p->alpha_multiplier, p->alpha_shift);
        const int32_t v = p->output_offset + scaled;
        hct_store_i32(&outputs[0], i, v < qmin ? qmin : (v > qmax ? qmax : v));
    }
    return HCT_OK;
}

int32_t hct_ref_leaky_relu_s8(const HctLeakyReluParams *params, const HctTensor *inputs, int32_t num_inputs,
                              HctTensor *outputs, int32_t num_outputs)
{
    return leaky_relu(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_leaky_relu_s16(const HctLeakyReluParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    return leaky_relu(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

/* ------------------------------------------------------------------ PReLU */

int32_t hct_ref_prelu_prepare(const HctPreluQuant *in, HctPreluParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(check_quant(in->dtype, in->input_scale, in->input_zero_point, in->output_scale, in->output_zero_point));
    HCT_TRY(check_quant(in->dtype, in->alpha_scale, in->alpha_zero_point, in->output_scale, in->output_zero_point));
    out->input_offset = -in->input_zero_point;
    out->alpha_offset = -in->alpha_zero_point;
    out->output_offset = in->output_zero_point;
    const double in_scale = (double)in->input_scale;
    const double out_scale = (double)in->output_scale;
    HCT_TRY(hct_quantize_multiplier_impl(in_scale / out_scale, &out->identity_multiplier, &out->identity_shift));
    return hct_quantize_multiplier_impl(in_scale * (double)in->alpha_scale / out_scale, &out->alpha_multiplier,
                                        &out->alpha_shift);
}

/* Input and alpha broadcast against each other (TFLite BroadcastPrelu4D); the output
 * takes the broadcast shape. */
static int32_t prelu_setup(const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs,
                           int32_t dtype, HctBroadcast2 *bc)
{
    return hct_binary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, bc);
}

static int32_t prelu_quantized(const HctPreluParams *p, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HctBroadcast2 bc;
    HCT_TRY(prelu_setup(inputs, num_inputs, outputs, num_outputs, dtype, &bc));
    HCT_TRY(check_offsets(dtype, -p->input_offset, p->output_offset));
    HCT_TRY(check_offsets(dtype, -p->alpha_offset, p->output_offset));
    HCT_TRY(check_multiplier(p->identity_multiplier, p->identity_shift));
    HCT_TRY(check_multiplier(p->alpha_multiplier, p->alpha_shift));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    for (int64_t i = 0; i < bc.count; ++i)
    {
        int64_t oi = 0;
        int64_t oa = 0;
        hct_broadcast2_offsets(&bc, i, &oi, &oa);
        const int32_t x = p->input_offset + hct_load_i32(&inputs[0], oi);
        int32_t v = 0;
        if (x >= 0)
        {
            v = hct_multiply_by_quantized_multiplier(x, p->identity_multiplier, p->identity_shift);
        }
        else
        {
            const int32_t a = p->alpha_offset + hct_load_i32(&inputs[1], oa);
            v = hct_multiply_by_quantized_multiplier(x * a, p->alpha_multiplier, p->alpha_shift);
        }
        v += p->output_offset;
        hct_store_i32(&outputs[0], i, v < qmin ? qmin : (v > qmax ? qmax : v));
    }
    return HCT_OK;
}

static int32_t prelu_float(const HctNoParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                           int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HctBroadcast2 bc;
    HCT_TRY(prelu_setup(inputs, num_inputs, outputs, num_outputs, dtype, &bc));
    for (int64_t i = 0; i < bc.count; ++i)
    {
        int64_t oi = 0;
        int64_t oa = 0;
        hct_broadcast2_offsets(&bc, i, &oi, &oa);
        const float x = hct_load_f32(&inputs[0], oi);
        hct_store_f32(&outputs[0], i, x >= 0.0f ? x : x * hct_load_f32(&inputs[1], oa));
    }
    return HCT_OK;
}

int32_t hct_ref_prelu_s8(const HctPreluParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return prelu_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_prelu_s16(const HctPreluParams *params, const HctTensor *inputs, int32_t num_inputs,
                          HctTensor *outputs, int32_t num_outputs)
{
    return prelu_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

int32_t hct_ref_prelu_f32(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                          int32_t num_outputs)
{
    return prelu_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_prelu_f16(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                          int32_t num_outputs)
{
    return prelu_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
