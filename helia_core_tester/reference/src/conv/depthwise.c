/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * DepthwiseConv2D: TFLite's reference_integer_ops::DepthwiseConvPerChannel (int8 with an int32
 * accumulator, int16 with an int64 one). Float: the exact sum rounded once, as conv.c. Input NHWC,
 * filter 1HWO with O = input depth * depth multiplier (implied by the shapes), output NHWC.
 */
#include "conv_common.h"

typedef struct
{
    int32_t batches, in_h, in_w, in_c, filter_h, filter_w, out_h, out_w, out_c, depth_multiplier;
} DwShape;

static int32_t dw_shape(const HctTensor *input, const HctTensor *filter, const HctTensor *output, DwShape *s)
{
    if (input->rank != 4 || filter->rank != 4 || output->rank != 4 || filter->dims[0] != 1)
    {
        return HCT_E_SHAPE;
    }
    s->batches = input->dims[0];
    s->in_h = input->dims[1];
    s->in_w = input->dims[2];
    s->in_c = input->dims[3];
    s->filter_h = filter->dims[1];
    s->filter_w = filter->dims[2];
    s->out_c = filter->dims[3];
    s->out_h = output->dims[1];
    s->out_w = output->dims[2];
    if (output->dims[0] != s->batches || output->dims[3] != s->out_c || s->in_c <= 0 || s->out_c % s->in_c != 0)
    {
        return HCT_E_SHAPE;
    }
    s->depth_multiplier = s->out_c / s->in_c;
    return HCT_OK;
}

#define HCT_DW_TAPS(s, p, input, filter, b, oy, ox, ic, oc, BODY)                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        const int32_t in_y0_ = (oy) * (p)->stride_h - (p)->pad_h;                                                    \
        const int32_t in_x0_ = (ox) * (p)->stride_w - (p)->pad_w;                                                    \
        for (int32_t fy_ = 0; fy_ < (s).filter_h; ++fy_)                                                             \
        {                                                                                                              \
            for (int32_t fx_ = 0; fx_ < (s).filter_w; ++fx_)                                                         \
            {                                                                                                          \
                const int32_t in_x_ = in_x0_ + (p)->dilation_w * fx_;                                                \
                const int32_t in_y_ = in_y0_ + (p)->dilation_h * fy_;                                                \
                if (in_x_ < 0 || in_x_ >= (s).in_w || in_y_ < 0 || in_y_ >= (s).in_h)                                \
                {                                                                                                      \
                    continue;                                                                                          \
                }                                                                                                      \
                const int64_t ii = hct_offset4((input), (b), in_y_, in_x_, (ic));                                    \
                const int64_t fi = hct_offset4((filter), 0, fy_, fx_, (oc));                                         \
                BODY;                                                                                                  \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

static int32_t dw_quantized(const HctConvParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                            int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 5, outputs, num_outputs, 1));
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], HCT_INT8, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &n));
    DwShape s;
    HCT_TRY(dw_shape(&inputs[0], &inputs[1], &outputs[0], &s));
    int has_bias = 0;
    HCT_TRY(hct_check_bias(&inputs[2], dtype == HCT_INT16 ? HCT_INT64 : HCT_INT32, s.out_c, &has_bias));
    HCT_TRY(hct_check_per_channel(&inputs[3], &inputs[4], s.out_c, -31, dtype == HCT_INT16 ? 7 : 30));
    HCT_TRY(hct_check_window(p->stride_h, p->stride_w, p->dilation_h, p->dilation_w, p->pad_h, p->pad_w));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (p->activation_min > p->activation_max || p->activation_min < qmin || p->activation_max > qmax ||
        (dtype == HCT_INT16 ? (p->input_offset != 0 || p->output_offset != 0)
                            : (p->input_offset < -127 || p->input_offset > 128 || p->output_offset < qmin ||
                               p->output_offset > qmax)))
    {
        return HCT_E_PARAM;
    }
    const HctTensor *input = &inputs[0];
    const HctTensor *filter = &inputs[1];
    const int8_t *fdata = (const int8_t *)filter->data;
    const int32_t *mult = (const int32_t *)inputs[3].data;
    const int32_t *shift = (const int32_t *)inputs[4].data;
    for (int32_t b = 0; b < s.batches; ++b)
    {
        for (int32_t oy = 0; oy < s.out_h; ++oy)
        {
            for (int32_t ox = 0; ox < s.out_w; ++ox)
            {
                for (int32_t ic = 0; ic < s.in_c; ++ic)
                {
                    for (int32_t m = 0; m < s.depth_multiplier; ++m)
                    {
                        const int32_t oc = m + ic * s.depth_multiplier;
                        int64_t acc = 0;
                        HCT_DW_TAPS(s, p, input, filter, b, oy, ox, ic, oc,
                                    acc += (int64_t)fdata[fi] * (hct_load_i32(input, ii) + p->input_offset));
                        if (has_bias)
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
                        v = v < p->activation_min ? p->activation_min
                                                  : (v > p->activation_max ? p->activation_max : v);
                        hct_store_i32(&outputs[0], hct_offset4(&outputs[0], b, oy, ox, oc), (int32_t)v);
                    }
                }
            }
        }
    }
    return HCT_OK;
}

int32_t hct_ref_depthwise_conv_s8(const HctConvParams *params, const HctTensor *inputs, int32_t num_inputs,
                                  HctTensor *outputs, int32_t num_outputs)
{
    return dw_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_depthwise_conv_s16(const HctConvParams *params, const HctTensor *inputs, int32_t num_inputs,
                                   HctTensor *outputs, int32_t num_outputs)
{
    return dw_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

static int32_t dw_float(const HctConvFloatParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], dtype, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &n));
    DwShape s;
    HCT_TRY(dw_shape(&inputs[0], &inputs[1], &outputs[0], &s));
    int has_bias = 0;
    HCT_TRY(hct_check_bias(&inputs[2], dtype, s.out_c, &has_bias));
    HCT_TRY(hct_check_window(p->stride_h, p->stride_w, p->dilation_h, p->dilation_w, p->pad_h, p->pad_w));
    HCT_TRY(hct_check_float_activation(p->activation_min, p->activation_max));
    const HctTensor *input = &inputs[0];
    const HctTensor *filter = &inputs[1];
    for (int32_t b = 0; b < s.batches; ++b)
    {
        for (int32_t oy = 0; oy < s.out_h; ++oy)
        {
            for (int32_t ox = 0; ox < s.out_w; ++ox)
            {
                for (int32_t ic = 0; ic < s.in_c; ++ic)
                {
                    for (int32_t m = 0; m < s.depth_multiplier; ++m)
                    {
                        const int32_t oc = m + ic * s.depth_multiplier;
                        HctXsum total;
                        hct_xsum_init(&total);
                        HCT_DW_TAPS(s, p, input, filter, b, oy, ox, ic, oc,
                                    hct_xsum_add_product(&total, hct_load_f32(input, ii), hct_load_f32(filter, fi)));
                        if (has_bias)
                        {
                            hct_xsum_add(&total, (double)hct_load_f32(&inputs[2], oc));
                        }
                        HCT_TRY(hct_store_xsum(&outputs[0], hct_offset4(&outputs[0], b, oy, ox, oc), &total, 1,
                                               p->activation_min, p->activation_max));
                    }
                }
            }
        }
    }
    return HCT_OK;
}

int32_t hct_ref_depthwise_conv_f32(const HctConvFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                                   HctTensor *outputs, int32_t num_outputs)
{
    return dw_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_depthwise_conv_f16(const HctConvFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                                   HctTensor *outputs, int32_t num_outputs)
{
    return dw_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
