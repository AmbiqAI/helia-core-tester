/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * TransposeConv2D: TFLite's reference_integer_ops::TransposeConv (int8: an int32 accumulator per
 * output element; int16: int64), scattering every input element through the filter, then the
 * per-channel requantization. Float: the exact sums rounded once. Input NHWC, filter OHWI with
 * I = input depth, output NHWC (its height and width given by the output tensor). TFLite's
 * transpose convolution has no dilation; the pad is the leading crop of the full output.
 */
#include <stdlib.h>

#include "conv_common.h"

typedef struct
{
    int32_t batches, in_h, in_w, in_c, filter_h, filter_w, out_h, out_w, out_c;
} TcShape;

static int32_t tc_shape(const HctTensor *input, const HctTensor *filter, const HctTensor *output, TcShape *s)
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
    s->out_h = output->dims[1];
    s->out_w = output->dims[2];
    if (filter->dims[3] != s->in_c || output->dims[0] != s->batches || output->dims[3] != s->out_c)
    {
        return HCT_E_SHAPE;
    }
    return HCT_OK;
}

static int32_t tc_check_window(int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
                               int32_t pad_h, int32_t pad_w)
{
    HCT_TRY(hct_check_window(stride_h, stride_w, dilation_h, dilation_w, pad_h, pad_w));
    return (dilation_h == 1 && dilation_w == 1) ? HCT_OK : HCT_E_PARAM;
}

/* Scatter-accumulates every in-bounds contribution into acc (one per output element). */
#define HCT_TC_SCATTER(s, p, input, filter, OUTPUT, BODY)                                                             \
    for (int32_t b_ = 0; b_ < (s).batches; ++b_)                                                                     \
        for (int32_t iy_ = 0; iy_ < (s).in_h; ++iy_)                                                                 \
            for (int32_t ix_ = 0; ix_ < (s).in_w; ++ix_)                                                             \
                for (int32_t ic_ = 0; ic_ < (s).in_c; ++ic_)                                                         \
                {                                                                                                      \
                    const int32_t ox0_ = ix_ * (p)->stride_w - (p)->pad_w;                                           \
                    const int32_t oy0_ = iy_ * (p)->stride_h - (p)->pad_h;                                           \
                    for (int32_t fy_ = 0; fy_ < (s).filter_h; ++fy_)                                                 \
                        for (int32_t fx_ = 0; fx_ < (s).filter_w; ++fx_)                                             \
                            for (int32_t oc_ = 0; oc_ < (s).out_c; ++oc_)                                            \
                            {                                                                                          \
                                const int32_t ox_ = ox0_ + fx_;                                                        \
                                const int32_t oy_ = oy0_ + fy_;                                                        \
                                if (ox_ < 0 || ox_ >= (s).out_w || oy_ < 0 || oy_ >= (s).out_h)                        \
                                {                                                                                      \
                                    continue;                                                                          \
                                }                                                                                      \
                                const int64_t ii = hct_offset4((input), b_, iy_, ix_, ic_);                            \
                                const int64_t fi = hct_offset4((filter), oc_, fy_, fx_, ic_);                          \
                                const int64_t oi = hct_offset4((OUTPUT), b_, oy_, ox_, oc_);                           \
                                BODY;                                                                                  \
                            }                                                                                          \
                }

static int32_t tc_quantized(const HctConvParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                            int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 5, outputs, num_outputs, 1));
    int64_t n = 0;
    int64_t out_count = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], HCT_INT8, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &out_count));
    TcShape s;
    HCT_TRY(tc_shape(&inputs[0], &inputs[1], &outputs[0], &s));
    int has_bias = 0;
    HCT_TRY(hct_check_bias(&inputs[2], dtype == HCT_INT16 ? HCT_INT64 : HCT_INT32, s.out_c, &has_bias));
    HCT_TRY(hct_check_per_channel(&inputs[3], &inputs[4], s.out_c, -31, dtype == HCT_INT16 ? 7 : 30));
    HCT_TRY(tc_check_window(p->stride_h, p->stride_w, p->dilation_h, p->dilation_w, p->pad_h, p->pad_w));
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
    if (out_count == 0)
    {
        return HCT_OK;
    }
    int64_t *acc = calloc((size_t)out_count, sizeof(int64_t));
    if (acc == NULL)
    {
        return HCT_E_SIZE;
    }
    const HctTensor *input = &inputs[0];
    const HctTensor *filter = &inputs[1];
    const HctTensor *output = &outputs[0];
    const int8_t *fdata = (const int8_t *)filter->data;
    HCT_TC_SCATTER(s, p, input, filter, output,
                   acc[oi] += (int64_t)(hct_load_i32(input, ii) + p->input_offset) * fdata[fi])
    const int32_t *mult = (const int32_t *)inputs[3].data;
    const int32_t *shift = (const int32_t *)inputs[4].data;
    int32_t status = HCT_OK;
    for (int64_t i = 0; i < out_count && status == HCT_OK; ++i)
    {
        const int32_t oc = (int32_t)(i % s.out_c);
        int64_t a = acc[i] + (has_bias ? hct_load_i64(&inputs[2], oc) : 0);
        int64_t scaled = 0;
        if (dtype == HCT_INT16)
        {
            if (a < -((int64_t)1 << 47) || a >= ((int64_t)1 << 47))
            {
                status = HCT_E_PARAM;
                break;
            }
            scaled = hct_multiply_by_quantized_multiplier_64(a, mult[oc], shift[oc]);
            if (scaled < INT32_MIN || scaled > INT32_MAX)
            {
                status = HCT_E_PARAM;
                break;
            }
        }
        else
        {
            /* TFLite's scratch accumulates in int32. */
            if (acc[i] < INT32_MIN || acc[i] > INT32_MAX || a < INT32_MIN || a > INT32_MAX)
            {
                status = HCT_E_PARAM;
                break;
            }
            scaled = hct_multiply_by_quantized_multiplier((int32_t)a, mult[oc], shift[oc]);
        }
        int64_t v = scaled + p->output_offset;
        v = v < p->activation_min ? p->activation_min : (v > p->activation_max ? p->activation_max : v);
        hct_store_i32(&outputs[0], i, (int32_t)v);
    }
    free(acc);
    return status;
}

int32_t hct_ref_transpose_conv_s8(const HctConvParams *params, const HctTensor *inputs, int32_t num_inputs,
                                  HctTensor *outputs, int32_t num_outputs)
{
    return tc_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_transpose_conv_s16(const HctConvParams *params, const HctTensor *inputs, int32_t num_inputs,
                                   HctTensor *outputs, int32_t num_outputs)
{
    return tc_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

static int32_t tc_float(const HctConvFloatParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    int64_t n = 0;
    int64_t out_count = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], dtype, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &out_count));
    TcShape s;
    HCT_TRY(tc_shape(&inputs[0], &inputs[1], &outputs[0], &s));
    int has_bias = 0;
    HCT_TRY(hct_check_bias(&inputs[2], dtype, s.out_c, &has_bias));
    HCT_TRY(tc_check_window(p->stride_h, p->stride_w, p->dilation_h, p->dilation_w, p->pad_h, p->pad_w));
    HCT_TRY(hct_check_float_activation(p->activation_min, p->activation_max));
    if (out_count == 0)
    {
        return HCT_OK;
    }
    HctXsum *acc = calloc((size_t)out_count, sizeof(HctXsum));
    if (acc == NULL)
    {
        return HCT_E_SIZE;
    }
    const HctTensor *input = &inputs[0];
    const HctTensor *filter = &inputs[1];
    const HctTensor *output = &outputs[0];
    HCT_TC_SCATTER(s, p, input, filter, output,
                   hct_xsum_add_product(&acc[oi], hct_load_f32(input, ii), hct_load_f32(filter, fi)))
    int32_t status = HCT_OK;
    for (int64_t i = 0; i < out_count && status == HCT_OK; ++i)
    {
        if (has_bias)
        {
            hct_xsum_add(&acc[i], (double)hct_load_f32(&inputs[2], i % s.out_c));
        }
        status = hct_store_xsum(&outputs[0], i, &acc[i], 1, p->activation_min, p->activation_max);
    }
    free(acc);
    return status;
}

int32_t hct_ref_transpose_conv_f32(const HctConvFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                                   HctTensor *outputs, int32_t num_outputs)
{
    return tc_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_transpose_conv_f16(const HctConvFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                                   HctTensor *outputs, int32_t num_outputs)
{
    return tc_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
