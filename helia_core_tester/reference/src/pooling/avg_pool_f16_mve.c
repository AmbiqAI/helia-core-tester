/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variant: ns-cmsis-nn's arm_avg_pool_f16 on Helium, bit for bit. Input and output NHWC; the window is
 * clipped to the image. Each output channel sums its window taps in float16 from +0, rows outer and columns
 * inner, one rounding per addition; the sum is multiplied in float16 by 1 / count, computed in float32 and
 * rounded to float16. The activation clamp is applied to finite values; NaN passes (the kernel's
 * vmaxnm/vminnm clamp maps it to a bound; nonfinite lanes are masked).
 */
#include <math.h>

#include "hct_ref_internal.h"

int32_t hct_ref_avg_pool_f16_mve(const HctPoolFloatParams *p, const HctTensor *inputs, int32_t num_inputs,
                                 HctTensor *outputs, int32_t num_outputs)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], HCT_FLOAT16, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT16, &n));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    if (in->rank != 4 || out->rank != 4 || in->dims[0] != out->dims[0] || in->dims[3] != out->dims[3])
    {
        return HCT_E_SHAPE;
    }
    if (p->stride_h < 1 || p->stride_w < 1 || p->filter_h < 1 || p->filter_w < 1 || p->pad_h < 0 || p->pad_w < 0)
    {
        return HCT_E_PARAM;
    }
    HCT_TRY(hct_check_float_activation(p->activation_min, p->activation_max));
    const int32_t batches = in->dims[0], in_h = in->dims[1], in_w = in->dims[2], depth = in->dims[3];
    const int32_t out_h = out->dims[1], out_w = out->dims[2];
    const float lo = hct_round_f16(p->activation_min);
    const float hi = hct_round_f16(p->activation_max);
    for (int32_t b = 0; b < batches; ++b)
        for (int32_t oy = 0; oy < out_h; ++oy)
            for (int32_t ox = 0; ox < out_w; ++ox)
            {
                const int32_t by = oy * p->stride_h - p->pad_h;
                const int32_t bx = ox * p->stride_w - p->pad_w;
                const int32_t ky0 = by < 0 ? -by : 0;
                const int32_t kx0 = bx < 0 ? -bx : 0;
                const int32_t ky1 = p->filter_h < in_h - by ? p->filter_h : in_h - by;
                const int32_t kx1 = p->filter_w < in_w - bx ? p->filter_w : in_w - bx;
                const int32_t count = (ky1 - ky0) * (kx1 - kx0);
                const float inv = count > 0 ? hct_round_f16(1.0f / (float)count) : 0.0f;
                for (int32_t c = 0; c < depth; ++c)
                {
                    float sum = 0.0f;
                    for (int32_t ky = ky0; ky < ky1; ++ky)
                    {
                        for (int32_t kx = kx0; kx < kx1; ++kx)
                        {
                            const int64_t ii = (((int64_t)b * in_h + by + ky) * in_w + bx + kx) * depth + c;
                            sum = hct_round_f16(sum + hct_load_f32(in, ii));
                        }
                    }
                    const float avg = hct_round_f16(sum * inv);
                    const int64_t oi = (((int64_t)b * out_h + oy) * out_w + ox) * depth + c;
                    hct_store_f32(out, oi, isnan(avg) ? avg : hct_clamp_f32(avg, lo, hi));
                }
            }
    return HCT_OK;
}
