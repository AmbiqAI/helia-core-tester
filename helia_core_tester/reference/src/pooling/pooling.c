/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * AveragePool2D / MaxPool2D: TFLite's reference_integer_ops::AveragePool / MaxPool (int8, int16),
 * reference_ops for float. The window is clipped to the image; an average divides by the taps it
 * covered, rounding half away from zero for integers (the output keeps the input quantization)
 * and once from the exact mean for floats. Float max is std::max's: a NaN tap never replaces the
 * running maximum. Input and output NHWC.
 */
#include <stdlib.h>

#include "hct_ref_internal.h"

typedef struct
{
    int32_t batches, in_h, in_w, depth, out_h, out_w;
} PoolShape;

static int32_t pool_shape(const HctTensor *in, const HctTensor *out, const int32_t *window, PoolShape *s)
{
    if (in->rank != 4 || out->rank != 4 || in->dims[0] != out->dims[0] || in->dims[3] != out->dims[3])
    {
        return HCT_E_SHAPE;
    }
    for (int i = 0; i < 4; ++i)
    {
        if (window[i] < 1)
        {
            return HCT_E_PARAM;
        }
    }
    s->batches = in->dims[0];
    s->in_h = in->dims[1];
    s->in_w = in->dims[2];
    s->depth = in->dims[3];
    s->out_h = out->dims[1];
    s->out_w = out->dims[2];
    return HCT_OK;
}

static int64_t at4(int32_t h, int32_t w, int32_t c, int32_t b, int32_t y, int32_t x, int32_t ch)
{
    return (((int64_t)b * h + y) * w + x) * c + ch;
}

/* Visits every output element with its clipped window [fy0, fy1) x [fx0, fx1). */
#define HCT_POOL_LOOP(s, p, BODY)                                                                                     \
    for (int32_t b = 0; b < (s).batches; ++b)                                                                         \
        for (int32_t oy = 0; oy < (s).out_h; ++oy)                                                                    \
            for (int32_t ox = 0; ox < (s).out_w; ++ox)                                                                \
                for (int32_t ch = 0; ch < (s).depth; ++ch)                                                            \
                {                                                                                                      \
                    const int32_t iy0 = oy * (p)->stride_h - (p)->pad_h;                                             \
                    const int32_t ix0 = ox * (p)->stride_w - (p)->pad_w;                                             \
                    const int32_t fy0 = iy0 < 0 ? -iy0 : 0;                                                          \
                    const int32_t fx0 = ix0 < 0 ? -ix0 : 0;                                                          \
                    const int32_t fy1 = (p)->filter_h < (s).in_h - iy0 ? (p)->filter_h : (s).in_h - iy0;             \
                    const int32_t fx1 = (p)->filter_w < (s).in_w - ix0 ? (p)->filter_w : (s).in_w - ix0;             \
                    const int64_t oi = at4((s).out_h, (s).out_w, (s).depth, b, oy, ox, ch);                          \
                    BODY;                                                                                              \
                }

static int32_t pool_quantized(const HctPoolParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                              int32_t num_outputs, int32_t dtype, int is_max)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &n));
    const int32_t window[4] = {p->stride_h, p->stride_w, p->filter_h, p->filter_w};
    PoolShape s;
    HCT_TRY(pool_shape(&inputs[0], &outputs[0], window, &s));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (p->pad_h < 0 || p->pad_w < 0 || p->activation_min > p->activation_max || p->activation_min < qmin ||
        p->activation_max > qmax)
    {
        return HCT_E_PARAM;
    }
    HCT_POOL_LOOP(s, p, {
        int32_t count = 0;
        int64_t acc = 0;
        int32_t max = qmin;
        for (int32_t fy = fy0; fy < fy1; ++fy)
        {
            for (int32_t fx = fx0; fx < fx1; ++fx)
            {
                const int32_t v = hct_load_i32(&inputs[0], at4(s.in_h, s.in_w, s.depth, b, iy0 + fy, ix0 + fx, ch));
                acc += v;
                max = v > max ? v : max;
                ++count;
            }
        }
        if (count == 0)
        {
            return HCT_E_PARAM; /* TFLite's AveragePool fails on an empty window */
        }
        int64_t v = is_max ? max : (acc > 0 ? (acc + count / 2) / count : (acc - count / 2) / count);
        v = v < p->activation_min ? p->activation_min : (v > p->activation_max ? p->activation_max : v);
        hct_store_i32(&outputs[0], oi, (int32_t)v);
    })
    return HCT_OK;
}

static int32_t pool_float(const HctPoolFloatParams *p, const HctTensor *inputs, int32_t num_inputs,
                          HctTensor *outputs, int32_t num_outputs, int32_t dtype, int is_max)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &n));
    const int32_t window[4] = {p->stride_h, p->stride_w, p->filter_h, p->filter_w};
    PoolShape s;
    HCT_TRY(pool_shape(&inputs[0], &outputs[0], window, &s));
    if (p->pad_h < 0 || p->pad_w < 0)
    {
        return HCT_E_PARAM;
    }
    HCT_TRY(hct_check_float_activation(p->activation_min, p->activation_max));
    HCT_POOL_LOOP(s, p, {
        int32_t count = 0;
        HctXsum total;
        hct_xsum_init(&total);
        float max = -3.40282347e38f;
        for (int32_t fy = fy0; fy < fy1; ++fy)
        {
            for (int32_t fx = fx0; fx < fx1; ++fx)
            {
                const float v = hct_load_f32(&inputs[0], at4(s.in_h, s.in_w, s.depth, b, iy0 + fy, ix0 + fx, ch));
                hct_xsum_add(&total, (double)v);
                if (max < v)
                {
                    max = v;
                }
                ++count;
            }
        }
        if (count == 0)
        {
            return HCT_E_PARAM;
        }
        if (is_max)
        {
            hct_store_rounded(&outputs[0], oi, (double)max, p->activation_min, p->activation_max);
        }
        else
        {
            HCT_TRY(hct_store_xsum(&outputs[0], oi, &total, count, p->activation_min, p->activation_max));
        }
    })
    return HCT_OK;
}

int32_t hct_ref_avg_pool_s8(const HctPoolParams *params, const HctTensor *inputs, int32_t num_inputs,
                            HctTensor *outputs, int32_t num_outputs)
{
    return pool_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8, 0);
}

int32_t hct_ref_avg_pool_s16(const HctPoolParams *params, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    return pool_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16, 0);
}

int32_t hct_ref_max_pool_s8(const HctPoolParams *params, const HctTensor *inputs, int32_t num_inputs,
                            HctTensor *outputs, int32_t num_outputs)
{
    return pool_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8, 1);
}

int32_t hct_ref_max_pool_s16(const HctPoolParams *params, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    return pool_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16, 1);
}

int32_t hct_ref_avg_pool_f32(const HctPoolFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    return pool_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32, 0);
}

int32_t hct_ref_avg_pool_f16(const HctPoolFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    return pool_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16, 0);
}

int32_t hct_ref_max_pool_f32(const HctPoolFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    return pool_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32, 1);
}

int32_t hct_ref_max_pool_f16(const HctPoolFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    return pool_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16, 1);
}
