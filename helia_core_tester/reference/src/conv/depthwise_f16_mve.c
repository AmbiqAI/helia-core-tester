/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variant: ns-cmsis-nn's float16 depthwise convolution on Helium, bit for bit, for every route that
 * accumulates over the in-range taps only: the ch_mult == 1 direct kernel (lanes are channels), the generic
 * kernel (one scalar accumulator, `acc += x * w` fused by GCC -Ofast), and the conv1d k3 and 2x5
 * specialisations. The one-input-channel route (arm_nn_mat_mult_nt_n_packed_f16 over zero-padded patches) is
 * matmul_nt_n_packed_f16_mve. Input [N, H, W, C], filter [1, KH, KW, C * M] (channel c * M + m), bias [C * M] or
 * empty, output [N, OH, OW, C * M].
 *
 * Every output: an accumulator starts at its bias in float16 and takes one fused multiply-add per in-range tap,
 * row by row then column by column, the in-range window found with the kernels' own start/end formulas
 * (arm_depthwise_direct_tap_start / _end). Folding (not acc16, more than 32 in-range taps) as in hct_f16_chain (hct_f16_mve.h).
 * The specialisations never reach 32 taps, so acc16 makes no difference there, as in the kernel. acc32 is the
 * default entries from ns-cmsis-nn#881 on every route: a float32 sum from the bias, one in-range tap at a time in
 * (ky, kx) order, rounded to float16 once. The activation clamp is applied to non-NaN values (the generator masks
 * nonfinite lanes).
 */
#include <stdlib.h>

#include "../common/hct_f16_mve.h"
#include "conv_common.h"

static int32_t tap_start(int32_t base, int32_t dilation)
{
    if (dilation > 1)
    {
        const int32_t s = (-base + dilation - 1) / dilation;
        return s > 0 ? s : 0;
    }
    return base < 0 ? -base : 0;
}

static int32_t tap_end(int32_t base, int32_t dilation, int32_t kernel, int32_t size)
{
    const int32_t e = dilation > 1 ? (size - base + dilation - 1) / dilation : size - base;
    return e < kernel ? e : kernel;
}

int32_t hct_ref_depthwise_f16_mve(const HctDepthwiseF16MveParams *p, const HctTensor *inputs, int32_t num_inputs,
                                  HctTensor *outputs, int32_t num_outputs)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], HCT_FLOAT16, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], HCT_FLOAT16, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT16, &n));
    HCT_TRY(hct_check_window(p->stride_h, p->stride_w, p->dilation_h, p->dilation_w, p->pad_h, p->pad_w));
    if ((p->acc16 != 0 && p->acc16 != 1) || (p->acc32 != 0 && p->acc32 != 1) || (p->acc16 && p->acc32) ||
        !(p->activation_min <= p->activation_max))
    {
        return HCT_E_PARAM;
    }
    const HctTensor *it = &inputs[0];
    const HctTensor *ft = &inputs[1];
    HctTensor *ot = &outputs[0];
    if (it->rank != 4 || ft->rank != 4 || ot->rank != 4 || ft->dims[0] != 1)
    {
        return HCT_E_SHAPE;
    }
    const int32_t B = it->dims[0];
    const int32_t H = it->dims[1];
    const int32_t W = it->dims[2];
    const int32_t C = it->dims[3];
    const int32_t KH = ft->dims[1];
    const int32_t KW = ft->dims[2];
    const int32_t OC = ft->dims[3];
    const int32_t OH = ot->dims[1];
    const int32_t OW = ot->dims[2];
    if (B < 1 || H < 1 || W < 1 || C < 1 || KH < 1 || KW < 1 || OC < 1 || OH < 1 || OW < 1 || OC % C != 0 ||
        ot->dims[0] != B || ot->dims[3] != OC)
    {
        return HCT_E_SHAPE;
    }
    int has_bias = 0;
    HCT_TRY(hct_check_bias(&inputs[2], HCT_FLOAT16, OC, &has_bias));
    if (has_bias && inputs[2].rank != 1)
    {
        return HCT_E_SHAPE;
    }
    if ((int64_t)KH * KW > HCT_MAX_ELEMENTS)
    {
        return HCT_E_SIZE;
    }
    const int32_t M = OC / C;
    const float lo = hct_round_f16(p->activation_min);
    const float hi = hct_round_f16(p->activation_max);
    float *xs = malloc((size_t)KH * KW * sizeof(float));
    float *ws = malloc((size_t)KH * KW * sizeof(float));
    if (xs == NULL || ws == NULL)
    {
        free(xs);
        free(ws);
        return HCT_E_SIZE;
    }
    int32_t status = HCT_OK;
    for (int32_t b = 0; b < B && status == HCT_OK; ++b)
    {
        for (int32_t oy = 0; oy < OH && status == HCT_OK; ++oy)
        {
            const int32_t base_y = oy * p->stride_h - p->pad_h;
            const int32_t y0 = tap_start(base_y, p->dilation_h);
            const int32_t y1 = tap_end(base_y, p->dilation_h, KH, H);
            for (int32_t ox = 0; ox < OW && status == HCT_OK; ++ox)
            {
                const int32_t base_x = ox * p->stride_w - p->pad_w;
                const int32_t x0 = tap_start(base_x, p->dilation_w);
                const int32_t x1 = tap_end(base_x, p->dilation_w, KW, W);
                for (int32_t oc = 0; oc < OC && status == HCT_OK; ++oc)
                {
                    const int32_t ic = oc / M;
                    int32_t taps = 0;
                    for (int32_t ky = y0; ky < y1; ++ky)
                    {
                        const int32_t iy = base_y + p->dilation_h * ky;
                        for (int32_t kx = x0; kx < x1; ++kx)
                        {
                            const int32_t ix = base_x + p->dilation_w * kx;
                            xs[taps] = hct_load_f32(it, hct_offset4(it, b, iy, ix, ic));
                            ws[taps] = hct_load_f32(ft, hct_offset4(ft, 0, ky, kx, oc));
                            ++taps;
                        }
                    }
                    const float bias = has_bias ? hct_load_f32(&inputs[2], oc) : 0.0f;
                    float acc = 0.0f;
                    if (p->acc32)
                    {
                        acc = bias;
                        for (int32_t t = 0; t < taps; ++t)
                        {
                            acc += xs[t] * ws[t];
                        }
                    }
                    else
                    {
                        status = hct_f16_chain(xs, ws, taps, !p->acc16, bias, &acc);
                    }
                    hct_store_f32(ot, hct_offset4(ot, b, oy, ox, oc), hct_f16_clamp_finite(acc, lo, hi));
                }
            }
        }
    }
    free(xs);
    free(ws);
    return status;
}
