/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * FullyConnected: TFLite's reference_integer_ops::FullyConnected / FullyConnectedPerChannel.
 * The accumulator is the bias type (int32 for int8 activations, int64 for int16), so the
 * requantization is the matching MultiplyByQuantizedMultiplier overload. Per-tensor is the
 * per-channel form with one multiplier repeated; filter_offset (-filter zero point) is 0 for a
 * symmetric filter. Float: the exact dot products rounded once. Input [..., accum] with
 * batches = elements / accum, filter [out, accum], output [batches, out].
 */
#include "hct_ref_internal.h"

typedef struct
{
    int64_t batches;
    int32_t accum, out_c;
} FcShape;

static int32_t fc_shape(const HctTensor *input, int64_t in_count, const HctTensor *filter, const HctTensor *output,
                        int64_t out_count, FcShape *s)
{
    if (filter->rank != 2 || output->rank < 1 || input->rank < 1)
    {
        return HCT_E_SHAPE;
    }
    s->out_c = filter->dims[0];
    s->accum = filter->dims[1];
    if (s->accum <= 0 || s->out_c <= 0 || in_count % s->accum != 0 || output->dims[output->rank - 1] != s->out_c)
    {
        return HCT_E_SHAPE;
    }
    s->batches = in_count / s->accum;
    return s->batches * s->out_c == out_count ? HCT_OK : HCT_E_SHAPE;
}

static int32_t fc_quantized(const HctFullyConnectedParams *p, const HctTensor *inputs, int32_t num_inputs,
                            HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 5, outputs, num_outputs, 1));
    int64_t in_count = 0;
    int64_t out_count = 0;
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &in_count));
    HCT_TRY(hct_check_tensor(&inputs[1], HCT_INT8, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &out_count));
    FcShape s;
    HCT_TRY(fc_shape(&inputs[0], in_count, &inputs[1], &outputs[0], out_count, &s));
    int64_t n_bias = 0;
    HCT_TRY(hct_check_tensor(&inputs[2], dtype == HCT_INT16 ? HCT_INT64 : HCT_INT32, &n_bias));
    if (n_bias != 0 && n_bias != s.out_c)
    {
        return HCT_E_SHAPE;
    }
    int64_t n_mult = 0;
    int64_t n_shift = 0;
    HCT_TRY(hct_check_tensor(&inputs[3], HCT_INT32, &n_mult));
    HCT_TRY(hct_check_tensor(&inputs[4], HCT_INT32, &n_shift));
    if (n_mult != s.out_c || n_shift != s.out_c)
    {
        return HCT_E_SHAPE;
    }
    const int32_t *mult = (const int32_t *)inputs[3].data;
    const int32_t *shift = (const int32_t *)inputs[4].data;
    for (int32_t c = 0; c < s.out_c; ++c)
    {
        if (mult[c] < 0 || shift[c] < -31 || shift[c] > (dtype == HCT_INT16 ? 7 : 30))
        {
            return HCT_E_PARAM;
        }
    }
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (p->activation_min > p->activation_max || p->activation_min < qmin || p->activation_max > qmax ||
        p->filter_offset < -127 || p->filter_offset > 128 ||
        (dtype == HCT_INT16 ? (p->input_offset != 0 || p->output_offset != 0 || p->filter_offset != 0)
                            : (p->input_offset < -127 || p->input_offset > 128 || p->output_offset < qmin ||
                               p->output_offset > qmax)))
    {
        return HCT_E_PARAM;
    }
    const int8_t *w = (const int8_t *)inputs[1].data;
    for (int64_t b = 0; b < s.batches; ++b)
    {
        for (int32_t oc = 0; oc < s.out_c; ++oc)
        {
            int64_t acc = 0;
            for (int32_t d = 0; d < s.accum; ++d)
            {
                acc += (int64_t)(w[(int64_t)oc * s.accum + d] + p->filter_offset) *
                       (hct_load_i32(&inputs[0], b * s.accum + d) + p->input_offset);
            }
            if (n_bias)
            {
                acc += hct_load_i64(&inputs[2], oc);
            }
            int64_t scaled = 0;
            if (dtype == HCT_INT16)
            {
                if (acc < -((int64_t)1 << 47) || acc >= ((int64_t)1 << 47))
                {
                    return HCT_E_PARAM;
                }
                scaled = hct_multiply_by_quantized_multiplier_64(acc, mult[oc], shift[oc]);
                if (scaled < INT32_MIN || scaled > INT32_MAX)
                {
                    return HCT_E_PARAM;
                }
            }
            else
            {
                if (acc < INT32_MIN || acc > INT32_MAX)
                {
                    return HCT_E_PARAM;
                }
                scaled = hct_multiply_by_quantized_multiplier((int32_t)acc, mult[oc], shift[oc]);
            }
            int64_t v = scaled + p->output_offset;
            v = v < p->activation_min ? p->activation_min : (v > p->activation_max ? p->activation_max : v);
            hct_store_i32(&outputs[0], b * s.out_c + oc, (int32_t)v);
        }
    }
    return HCT_OK;
}

int32_t hct_ref_fully_connected_s8(const HctFullyConnectedParams *params, const HctTensor *inputs, int32_t num_inputs,
                                   HctTensor *outputs, int32_t num_outputs)
{
    return fc_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_fully_connected_s16(const HctFullyConnectedParams *params, const HctTensor *inputs,
                                    int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return fc_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

static int32_t fc_float(const HctFloatActivationParams *p, const HctTensor *inputs, int32_t num_inputs,
                        HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    int64_t in_count = 0;
    int64_t out_count = 0;
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &in_count));
    HCT_TRY(hct_check_tensor(&inputs[1], dtype, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &out_count));
    FcShape s;
    HCT_TRY(fc_shape(&inputs[0], in_count, &inputs[1], &outputs[0], out_count, &s));
    int64_t n_bias = 0;
    HCT_TRY(hct_check_tensor(&inputs[2], dtype, &n_bias));
    if (n_bias != 0 && n_bias != s.out_c)
    {
        return HCT_E_SHAPE;
    }
    HCT_TRY(hct_check_float_activation(p->activation_min, p->activation_max));
    for (int64_t b = 0; b < s.batches; ++b)
    {
        for (int32_t oc = 0; oc < s.out_c; ++oc)
        {
            HctXsum total;
            hct_xsum_init(&total);
            for (int32_t d = 0; d < s.accum; ++d)
            {
                hct_xsum_add_product(&total, hct_load_f32(&inputs[0], b * s.accum + d),
                                     hct_load_f32(&inputs[1], (int64_t)oc * s.accum + d));
            }
            if (n_bias)
            {
                hct_xsum_add(&total, (double)hct_load_f32(&inputs[2], oc));
            }
            HCT_TRY(hct_store_xsum(&outputs[0], b * s.out_c + oc, &total, 1, p->activation_min, p->activation_max));
        }
    }
    return HCT_OK;
}

int32_t hct_ref_fully_connected_f32(const HctFloatActivationParams *params, const HctTensor *inputs,
                                    int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return fc_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_fully_connected_f16(const HctFloatActivationParams *params, const HctTensor *inputs,
                                    int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return fc_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
