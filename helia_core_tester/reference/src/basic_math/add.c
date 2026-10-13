/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Elementwise Add with numpy broadcasting.
 *
 * int8/int16: TFLite's general quantized add. Each input is offset, shifted up
 * by left_shift (20 for int8, 15 for int16), rescaled to a common scale twice the
 * larger input scale, summed, rescaled to the output and clamped.
 * float32: a + b, clamped. float16: the binary16 operands are widened, added and
 * clamped in binary32, and the result rounded to binary16 once.
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

static int32_t check_add_params(const HctAddParams *p, int32_t dtype)
{
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    HCT_TRY(check_offsets(dtype, p->input1_offset, p->input2_offset, p->output_offset));
    const int32_t want_left_shift = dtype == HCT_INT16 ? 15 : 20;
    if (p->left_shift != want_left_shift)
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

static int32_t add_quantized(
    const HctAddParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs,
    int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    int64_t n1 = 0;
    int64_t n2 = 0;
    int64_t no = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &n1));
    HCT_TRY(hct_check_tensor(&inputs[1], dtype, &n2));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &no));
    HCT_TRY(check_add_params(p, dtype));
    HctBroadcast2 bc;
    HCT_TRY(hct_broadcast2_init(&bc, &inputs[0], &inputs[1], &outputs[0]));

    for (int64_t i = 0; i < bc.count; ++i)
    {
        int64_t o1 = 0;
        int64_t o2 = 0;
        hct_broadcast2_offsets(&bc, i, &o1, &o2);
        const int32_t x1 = dtype == HCT_INT8 ? ((const int8_t *)inputs[0].data)[o1] : ((const int16_t *)inputs[0].data)[o1];
        const int32_t x2 = dtype == HCT_INT8 ? ((const int8_t *)inputs[1].data)[o2] : ((const int16_t *)inputs[1].data)[o2];
        const int32_t shifted1 = (p->input1_offset + x1) * (1 << p->left_shift);
        const int32_t shifted2 = (p->input2_offset + x2) * (1 << p->left_shift);
        const int32_t scaled1 = hct_multiply_by_quantized_multiplier(shifted1, p->input1_multiplier, p->input1_shift);
        const int32_t scaled2 = hct_multiply_by_quantized_multiplier(shifted2, p->input2_multiplier, p->input2_shift);
        int32_t out = hct_multiply_by_quantized_multiplier(scaled1 + scaled2, p->output_multiplier, p->output_shift) +
                      p->output_offset;
        out = out < p->activation_min ? p->activation_min : (out > p->activation_max ? p->activation_max : out);
        if (dtype == HCT_INT8)
        {
            ((int8_t *)outputs[0].data)[i] = (int8_t)out;
        }
        else
        {
            ((int16_t *)outputs[0].data)[i] = (int16_t)out;
        }
    }
    return HCT_OK;
}

static int32_t add_float(
    const HctFloatActivationParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
    int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    int64_t n1 = 0;
    int64_t n2 = 0;
    int64_t no = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &n1));
    HCT_TRY(hct_check_tensor(&inputs[1], dtype, &n2));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &no));
    HCT_TRY(hct_check_float_activation(p->activation_min, p->activation_max));
    HctBroadcast2 bc;
    HCT_TRY(hct_broadcast2_init(&bc, &inputs[0], &inputs[1], &outputs[0]));

    for (int64_t i = 0; i < bc.count; ++i)
    {
        int64_t o1 = 0;
        int64_t o2 = 0;
        hct_broadcast2_offsets(&bc, i, &o1, &o2);
        if (dtype == HCT_FLOAT32)
        {
            const float sum = ((const float *)inputs[0].data)[o1] + ((const float *)inputs[1].data)[o2];
            ((float *)outputs[0].data)[i] = hct_clamp_f32(sum, p->activation_min, p->activation_max);
        }
        else
        {
            const float sum = hct_f16_to_f32(((const uint16_t *)inputs[0].data)[o1]) +
                              hct_f16_to_f32(((const uint16_t *)inputs[1].data)[o2]);
            ((uint16_t *)outputs[0].data)[i] =
                hct_f32_to_f16(hct_clamp_f32(sum, p->activation_min, p->activation_max));
        }
    }
    return HCT_OK;
}

int32_t hct_ref_add_prepare(const HctAddQuant *in, HctAddParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(in->dtype, &qmin, &qmax));
    const float scales[3] = {in->input1_scale, in->input2_scale, in->output_scale};
    for (int i = 0; i < 3; ++i)
    {
        if (!(isfinite(scales[i]) && scales[i] > 0.0f))
        {
            return HCT_E_PARAM;
        }
    }
    const int32_t zps[3] = {in->input1_zero_point, in->input2_zero_point, in->output_zero_point};
    for (int i = 0; i < 3; ++i)
    {
        if (zps[i] < qmin || zps[i] > qmax)
        {
            return HCT_E_PARAM;
        }
    }
    out->left_shift = in->dtype == HCT_INT16 ? 15 : 20;
    out->input1_offset = -in->input1_zero_point;
    out->input2_offset = -in->input2_zero_point;
    out->output_offset = in->output_zero_point;
    HCT_TRY(check_offsets(in->dtype, out->input1_offset, out->input2_offset, out->output_offset));

    /* As TFLite's Add prepare: the scales are float32, the arithmetic double. */
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

int32_t hct_ref_add_s8(
    const HctAddParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return add_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_add_s16(
    const HctAddParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return add_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

int32_t hct_ref_add_f32(const HctFloatActivationParams *params, const HctTensor *inputs, int32_t num_inputs,
                        HctTensor *outputs, int32_t num_outputs)
{
    return add_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_add_f16(const HctFloatActivationParams *params, const HctTensor *inputs, int32_t num_inputs,
                        HctTensor *outputs, int32_t num_outputs)
{
    return add_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
