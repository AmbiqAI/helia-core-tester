/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Elementwise Add and Sub with numpy broadcasting.
 *
 * int8/int16: TFLite's general quantized add/sub. Each input is offset, shifted up
 * by left_shift (20 for int8, 15 for int16), rescaled to a common scale twice the
 * larger input scale, added or subtracted, rescaled to the output and clamped.
 * float32: a +/- b, clamped. float16: the binary16 operands are widened, combined
 * and clamped in binary32, and the result rounded to binary16 once.
 */
#include <math.h>

#include "hct_ref_internal.h"

static int32_t check_offsets(int32_t dtype, int32_t in1, int32_t in2, int32_t out)
{
    if (dtype == HCT_INT16)
    {
        /* TFLite int16 tensors are symmetric. */
        return (in1 == 0 && in2 == 0 && out == 0) ? HCT_OK : HCT_E_PARAM;
    }
    const int ok = in1 >= -INT8_MAX && in1 <= -INT8_MIN && in2 >= -INT8_MAX && in2 <= -INT8_MIN && out >= INT8_MIN &&
                   out <= INT8_MAX;
    return ok ? HCT_OK : HCT_E_PARAM;
}

static int32_t check_params(const HctAddParams *p, int32_t dtype)
{
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    HCT_TRY(check_offsets(dtype, p->input1_offset, p->input2_offset, p->output_offset));
    if (p->left_shift != (dtype == HCT_INT16 ? 15 : 20))
    {
        return HCT_E_PARAM;
    }
    const int32_t mults[3] = {p->input1_multiplier, p->input2_multiplier, p->output_multiplier};
    const int32_t shifts[3] = {p->input1_shift, p->input2_shift, p->output_shift};
    for (int i = 0; i < 3; ++i)
    {
        if (mults[i] < 0 || shifts[i] > 0 || shifts[i] < -31)
        {
            return HCT_E_PARAM;
        }
    }
    if (p->activation_min > p->activation_max || p->activation_min < qmin || p->activation_max > qmax)
    {
        return HCT_E_PARAM;
    }
    return HCT_OK;
}

static int32_t add_sub_quantized(const HctAddParams *p, const HctTensor *inputs, int32_t num_inputs,
                                 HctTensor *outputs, int32_t num_outputs, int32_t dtype, int32_t sign)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HctBroadcast2 bc;
    HCT_TRY(hct_binary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &bc));
    HCT_TRY(check_params(p, dtype));
    for (int64_t i = 0; i < bc.count; ++i)
    {
        int64_t o1 = 0;
        int64_t o2 = 0;
        hct_broadcast2_offsets(&bc, i, &o1, &o2);
        const int32_t shifted1 = (p->input1_offset + hct_load_i32(&inputs[0], o1)) * (1 << p->left_shift);
        const int32_t shifted2 = (p->input2_offset + hct_load_i32(&inputs[1], o2)) * (1 << p->left_shift);
        const int32_t scaled1 = hct_multiply_by_quantized_multiplier(shifted1, p->input1_multiplier, p->input1_shift);
        const int32_t scaled2 = hct_multiply_by_quantized_multiplier(shifted2, p->input2_multiplier, p->input2_shift);
        int32_t out = hct_multiply_by_quantized_multiplier(scaled1 + sign * scaled2, p->output_multiplier,
                                                           p->output_shift) +
                      p->output_offset;
        out = out < p->activation_min ? p->activation_min : (out > p->activation_max ? p->activation_max : out);
        hct_store_i32(&outputs[0], i, out);
    }
    return HCT_OK;
}

static int32_t add_sub_float(const HctFloatActivationParams *p, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs, int32_t dtype, float sign)
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
        const float result = hct_load_f32(&inputs[0], o1) + sign * hct_load_f32(&inputs[1], o2);
        hct_store_f32(&outputs[0], i, hct_clamp_f32(result, p->activation_min, p->activation_max));
    }
    return HCT_OK;
}

static int32_t add_sub_prepare(const HctAddQuant *in, HctAddParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(in->dtype, &qmin, &qmax));
    const float scales[3] = {in->input1_scale, in->input2_scale, in->output_scale};
    const int32_t zps[3] = {in->input1_zero_point, in->input2_zero_point, in->output_zero_point};
    for (int i = 0; i < 3; ++i)
    {
        if (!(isfinite(scales[i]) && scales[i] > 0.0f) || zps[i] < qmin || zps[i] > qmax)
        {
            return HCT_E_PARAM;
        }
    }
    out->left_shift = in->dtype == HCT_INT16 ? 15 : 20;
    out->input1_offset = -in->input1_zero_point;
    out->input2_offset = -in->input2_zero_point;
    out->output_offset = in->output_zero_point;
    HCT_TRY(check_offsets(in->dtype, out->input1_offset, out->input2_offset, out->output_offset));

    /* As TFLite's prepare: the scales are float32, the arithmetic double. */
    const double s1 = (double)in->input1_scale;
    const double s2 = (double)in->input2_scale;
    const double twice_max_input_scale = 2.0 * (s1 > s2 ? s1 : s2);
    const double real_output = twice_max_input_scale / ((double)(1 << out->left_shift) * (double)in->output_scale);
    HCT_TRY(hct_quantize_multiplier_smaller_than_one(s1 / twice_max_input_scale, &out->input1_multiplier,
                                                     &out->input1_shift));
    HCT_TRY(hct_quantize_multiplier_smaller_than_one(s2 / twice_max_input_scale, &out->input2_multiplier,
                                                     &out->input2_shift));
    HCT_TRY(hct_quantize_multiplier_smaller_than_one(real_output, &out->output_multiplier, &out->output_shift));
    return hct_activation_range_quantized(in->activation, in->dtype, in->output_scale, in->output_zero_point,
                                          &out->activation_min, &out->activation_max);
}

int32_t hct_ref_add_prepare(const HctAddQuant *in, HctAddParams *out)
{
    return add_sub_prepare(in, out);
}

int32_t hct_ref_sub_prepare(const HctAddQuant *in, HctAddParams *out)
{
    return add_sub_prepare(in, out);
}

#define HCT_ADD_SUB_ENTRIES(op, sign)                                                                                  \
    int32_t hct_ref_##op##_s8(const HctAddParams *params, const HctTensor *inputs, int32_t num_inputs,                \
                              HctTensor *outputs, int32_t num_outputs)                                                 \
    {                                                                                                                  \
        return add_sub_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8, sign);                    \
    }                                                                                                                  \
    int32_t hct_ref_##op##_s16(const HctAddParams *params, const HctTensor *inputs, int32_t num_inputs,               \
                               HctTensor *outputs, int32_t num_outputs)                                                \
    {                                                                                                                  \
        return add_sub_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16, sign);                   \
    }                                                                                                                  \
    int32_t hct_ref_##op##_f32(const HctFloatActivationParams *params, const HctTensor *inputs, int32_t num_inputs,   \
                               HctTensor *outputs, int32_t num_outputs)                                                \
    {                                                                                                                  \
        return add_sub_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32, (float)(sign));            \
    }                                                                                                                  \
    int32_t hct_ref_##op##_f16(const HctFloatActivationParams *params, const HctTensor *inputs, int32_t num_inputs,   \
                               HctTensor *outputs, int32_t num_outputs)                                                \
    {                                                                                                                  \
        return add_sub_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16, (float)(sign));            \
    }

HCT_ADD_SUB_ENTRIES(add, 1)
HCT_ADD_SUB_ENTRIES(sub, -1)
