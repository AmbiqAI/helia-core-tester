/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Conv2D: TFLite's reference_integer_ops::ConvPerChannel (int8 with an int32 accumulator,
 * int16 with an int64 one). Float: the exact sum (products of binary32/binary16 operands are
 * exact in binary64, accumulated there) plus bias, rounded once to the output type and then
 * clamped; summation order is not part of the reference.
 * Input NHWC, filter OHWI with I = input depth / groups, output NHWC.
 */
#include <math.h>

#include "conv_common.h"

int32_t hct_ref_per_channel_quant(const HctPerChannelQuantParams *params, const HctTensor *inputs, int32_t num_inputs,
                                  HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 2));
    int64_t n = 0;
    int64_t n_mult = 0;
    int64_t n_shift = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], HCT_FLOAT32, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_INT32, &n_mult));
    HCT_TRY(hct_check_tensor(&outputs[1], HCT_INT32, &n_shift));
    if (n_mult != n || n_shift != n)
    {
        return HCT_E_SHAPE;
    }
    if (!(isfinite(params->input_scale) && params->input_scale > 0.0f && isfinite(params->output_scale) &&
          params->output_scale > 0.0f))
    {
        return HCT_E_PARAM;
    }
    for (int64_t c = 0; c < n; ++c)
    {
        const float filter_scale = ((const float *)inputs[0].data)[c];
        if (!(isfinite(filter_scale) && filter_scale >= 0.0f))
        {
            return HCT_E_PARAM;
        }
    }
    for (int64_t c = 0; c < n; ++c)
    {
        const double real = (double)params->input_scale * (double)((const float *)inputs[0].data)[c] /
                            (double)params->output_scale;
        HCT_TRY(hct_quantize_multiplier_impl(real, &((int32_t *)outputs[0].data)[c],
                                             &((int32_t *)outputs[1].data)[c]));
    }
    return HCT_OK;
}

typedef struct
{
    int32_t batches, in_h, in_w, in_c;
    int32_t filter_h, filter_w, filter_c;
    int32_t out_h, out_w, out_c;
    int32_t groups, filters_per_group;
} ConvShape;

static int32_t conv_shape(const HctTensor *input, const HctTensor *filter, const HctTensor *output, ConvShape *s)
{
    if (input->rank != 4 || filter->rank != 4 || output->rank != 4)
    {
        return HCT_E_SHAPE;
    }
    s->batches = input->dims[0];
    s->in_h = input->dims[1];
    s->in_w = input->dims[2];
    s->in_c = input->dims[3];
    s->out_c = filter->dims[0];
    s->filter_h = filter->dims[1];
    s->filter_w = filter->dims[2];
    s->filter_c = filter->dims[3];
    s->out_h = output->dims[1];
    s->out_w = output->dims[2];
    if (output->dims[0] != s->batches || output->dims[3] != s->out_c || s->filter_c <= 0 || s->in_c % s->filter_c != 0)
    {
        return HCT_E_SHAPE;
    }
    s->groups = s->in_c / s->filter_c;
    if (s->groups <= 0 || s->out_c % s->groups != 0)
    {
        return HCT_E_SHAPE;
    }
    s->filters_per_group = s->out_c / s->groups;
    return HCT_OK;
}

/* Calls visit(ctx, input_index, filter_index) for every tap of output element
 * (b, oy, ox, oc) that lands inside the image, in TFLite's loop order. */
#define HCT_CONV_TAPS(s, p, input, filter, b, oy, ox, oc, BODY)                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        const int32_t group_ = (oc) / (s).filters_per_group;                                                         \
        const int32_t in_y0_ = (oy) * (p)->stride_h - (p)->pad_h;                                                    \
        const int32_t in_x0_ = (ox) * (p)->stride_w - (p)->pad_w;                                                    \
        for (int32_t fy_ = 0; fy_ < (s).filter_h; ++fy_)                                                             \
        {                                                                                                              \
            const int32_t in_y_ = in_y0_ + (p)->dilation_h * fy_;                                                    \
            for (int32_t fx_ = 0; fx_ < (s).filter_w; ++fx_)                                                         \
            {                                                                                                          \
                const int32_t in_x_ = in_x0_ + (p)->dilation_w * fx_;                                                \
                if (in_x_ < 0 || in_x_ >= (s).in_w || in_y_ < 0 || in_y_ >= (s).in_h)                                \
                {                                                                                                      \
                    continue;                                                                                          \
                }                                                                                                      \
                for (int32_t ic_ = 0; ic_ < (s).filter_c; ++ic_)                                                     \
                {                                                                                                      \
                    const int64_t ii = hct_offset4((input), (b), in_y_, in_x_, ic_ + group_ * (s).filter_c);         \
                    const int64_t fi = hct_offset4((filter), (oc), fy_, fx_, ic_);                                   \
                    BODY;                                                                                              \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

static int32_t conv_quantized(const HctConvParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                              int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 5, outputs, num_outputs, 1));
    const int32_t bias_dtype = dtype == HCT_INT16 ? HCT_INT64 : HCT_INT32;
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], HCT_INT8, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &n));
    ConvShape s;
    HCT_TRY(conv_shape(&inputs[0], &inputs[1], &outputs[0], &s));
    int has_bias = 0;
    HCT_TRY(hct_check_bias(&inputs[2], bias_dtype, s.out_c, &has_bias));
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
                for (int32_t oc = 0; oc < s.out_c; ++oc)
                {
                    int64_t acc = 0;
                    HCT_CONV_TAPS(s, p, input, filter, b, oy, ox, oc,
                                  acc += (int64_t)fdata[fi] * (hct_load_i32(input, ii) + p->input_offset));
                    if (has_bias)
                    {
                        acc += hct_load_i64(&inputs[2], oc);
                    }
                    int64_t scaled = 0;
                    if (dtype == HCT_INT16)
                    {
                        /* The int64 MultiplyByQuantizedMultiplier is specified for |acc| < 2^47. */
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
                        /* TFLite accumulates in int32; a sum past it is outside the op's definition. */
                        if (acc < INT32_MIN || acc > INT32_MAX)
                        {
                            return HCT_E_PARAM;
                        }
                        scaled = hct_multiply_by_quantized_multiplier((int32_t)acc, mult[oc], shift[oc]);
                    }
                    int64_t v = scaled + p->output_offset;
                    v = v < p->activation_min ? p->activation_min : (v > p->activation_max ? p->activation_max : v);
                    hct_store_i32(&outputs[0], hct_offset4(&outputs[0], b, oy, ox, oc), (int32_t)v);
                }
            }
        }
    }
    return HCT_OK;
}

int32_t hct_ref_conv_s8(const HctConvParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs)
{
    return conv_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_conv_s16(const HctConvParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                         int32_t num_outputs)
{
    return conv_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

static int32_t conv_float(const HctConvFloatParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
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
    ConvShape s;
    HCT_TRY(conv_shape(&inputs[0], &inputs[1], &outputs[0], &s));
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
                for (int32_t oc = 0; oc < s.out_c; ++oc)
                {
                    HctXsum total;
                    hct_xsum_init(&total);
                    HCT_CONV_TAPS(s, p, input, filter, b, oy, ox, oc,
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
    return HCT_OK;
}

int32_t hct_ref_conv_f32(const HctConvFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return conv_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_conv_f16(const HctConvFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return conv_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
