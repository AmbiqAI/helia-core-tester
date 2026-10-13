/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variant: ns-cmsis-nn's arm_transpose_conv_nhwc_f16 on Helium, bit for bit. Input NHWC, filter OHWI
 * (I = input depth), bias [O] or empty, output NHWC (its height and width given by the output tensor).
 *
 * Every output element starts at its bias (or +0) in float16. The kernel then walks the input in
 * (batch, y, x, channel) order, skips elements equal to zero, and for each filter tap (ky, kx) that lands
 * inside the output adds value * weight to the output channels with one float16 fused multiply-add
 * (vfmaq_m over a gather of the eight channels' weights; the gather needs the output-channel stride
 * kh * kw * ic <= 65535 / 7, which the generator checks). The activation clamp is applied to finite
 * values; NaN passes (the kernel's vmaxnm/vminnm clamp maps it to a bound; nonfinite lanes are masked).
 */
#include <math.h>
#include <stdlib.h>

#include "conv_common.h"

/* a * b + c rounded once to float16 (exactly, through the exact accumulator). */
static int32_t tc_fma16(float a, float b, float c, float *out)
{
    HctXsum x;
    hct_xsum_init(&x);
    hct_xsum_add_product(&x, a, b);
    hct_xsum_add(&x, (double)c);
    double v = 0.0;
    HCT_TRY(hct_xsum_round(&x, 1, HCT_FLOAT16, &v));
    *out = (float)v;
    return HCT_OK;
}

int32_t hct_ref_transpose_conv_f16_mve(const HctConvFloatParams *p, const HctTensor *inputs, int32_t num_inputs,
                                       HctTensor *outputs, int32_t num_outputs)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    int64_t n = 0;
    int64_t out_count = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], HCT_FLOAT16, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], HCT_FLOAT16, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT16, &out_count));
    const HctTensor *in = &inputs[0];
    const HctTensor *flt = &inputs[1];
    HctTensor *out = &outputs[0];
    if (in->rank != 4 || flt->rank != 4 || out->rank != 4)
    {
        return HCT_E_SHAPE;
    }
    const int32_t batches = in->dims[0], in_h = in->dims[1], in_w = in->dims[2], in_c = in->dims[3];
    const int32_t out_c = flt->dims[0], kh = flt->dims[1], kw = flt->dims[2];
    const int32_t out_h = out->dims[1], out_w = out->dims[2];
    if (flt->dims[3] != in_c || out->dims[0] != batches || out->dims[3] != out_c)
    {
        return HCT_E_SHAPE;
    }
    int has_bias = 0;
    HCT_TRY(hct_check_bias(&inputs[2], HCT_FLOAT16, out_c, &has_bias));
    HCT_TRY(hct_check_window(p->stride_h, p->stride_w, p->dilation_h, p->dilation_w, p->pad_h, p->pad_w));
    HCT_TRY(hct_check_float_activation(p->activation_min, p->activation_max));
    if (out_count == 0)
    {
        return HCT_OK;
    }
    float *acc = malloc((size_t)out_count * sizeof(float));
    if (acc == NULL)
    {
        return HCT_E_SIZE;
    }
    for (int64_t i = 0; i < out_count; ++i)
    {
        acc[i] = has_bias ? hct_load_f32(&inputs[2], i % out_c) : 0.0f;
    }
    int32_t status = HCT_OK;
    for (int32_t b = 0; b < batches && status == HCT_OK; ++b)
        for (int32_t iy = 0; iy < in_h && status == HCT_OK; ++iy)
            for (int32_t ix = 0; ix < in_w && status == HCT_OK; ++ix)
                for (int32_t ic = 0; ic < in_c && status == HCT_OK; ++ic)
                {
                    const float val = hct_load_f32(in, hct_offset4(in, b, iy, ix, ic));
                    if (val == 0.0f)
                    {
                        continue;
                    }
                    for (int32_t ky = 0; ky < kh && status == HCT_OK; ++ky)
                    {
                        const int32_t oy = iy * p->stride_h - p->pad_h + ky * p->dilation_h;
                        if (oy < 0 || oy >= out_h)
                        {
                            continue;
                        }
                        for (int32_t kx = 0; kx < kw && status == HCT_OK; ++kx)
                        {
                            const int32_t ox = ix * p->stride_w - p->pad_w + kx * p->dilation_w;
                            if (ox < 0 || ox >= out_w)
                            {
                                continue;
                            }
                            for (int32_t oc = 0; oc < out_c && status == HCT_OK; ++oc)
                            {
                                float *o = &acc[hct_offset4(out, b, oy, ox, oc)];
                                status = tc_fma16(val, hct_load_f32(flt, hct_offset4(flt, oc, ky, kx, ic)), *o, o);
                            }
                        }
                    }
                }
    const float lo = hct_round_f16(p->activation_min);
    const float hi = hct_round_f16(p->activation_max);
    for (int64_t i = 0; i < out_count && status == HCT_OK; ++i)
    {
        hct_store_f32(out, i, isnan(acc[i]) ? acc[i] : hct_clamp_f32(acc[i], lo, hi));
    }
    free(acc);
    return status;
}
