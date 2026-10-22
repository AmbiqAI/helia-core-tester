/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variant: the float16 convolution routes of ns-cmsis-nn's arm_convolve_f16 family on Helium (other than
 * arm_convolve_1_x_n_f16, see conv_1xn_f16_mve.c), bit for bit. Input NHWC [N, H, W, C], filter OHWI
 * [O, KH, KW, C] (unpacked even for an NT_N_PACKED case: packing only changes storage), bias [O] or empty, output
 * [N, OH, OW, O]. A tap is (ky, kx, ic) in that order; it is in range when its input row and column are.
 *
 * route 0 (patch GEMM / 1x1, standard filter): each output's zero-padded patch row through
 *   arm_nn_mat_mult_nt_t_f16 (_acc16) -- hct_f16_nt_t_column. A 1x1 filter's patch is the strided pixel.
 * route 1 (patch GEMM / 1x1, NT_N_PACKED filter): the patch row through arm_nn_mat_mult_nt_n_packed_f16 (_acc16).
 * route 2 (direct, OHWI rows): float16 lanes from zero over the in-range taps only, channel ic of a tap in lane
 *   ic % 8. Folding (patch length over 32, not acc16) restarts the lanes after every 32 channel vectors and adds
 *   their adjacent pairs into four float32 accumulators; an output whose in-range taps number more than 32 sums
 *   those (0+1)+(2+3), adds the bias in float32 and rounds once; any other output reduces its lanes in float16
 *   ((0+1)+(2+3))+((4+5)+(6+7)) and adds the bias in float16.
 * route 3 (direct, NT_N_PACKED): lanes are output channels: a chain of fused float16 multiply-adds from the bias
 *   over the in-range taps; folding (patch length over 32, not acc16) restarts every 32 in-range taps from zero and
 *   adds the blocks in float32, one rounding.
 * route 4 (small C, fewer than 8 input channels; either filter format): float32 from the bias, one float32 fused
 *   multiply-add per in-range tap, one rounding to float16.
 * route 5 (conv1d k3/k5, OHWI rows; batch 1, height 1, unit stride, no padding): channel ic in lane ic % 8, per
 *   eight-channel step the K taps in order. Folding (K * C over 32, not acc16) restarts the lanes every
 *   (32 / K) * 8 channels into float32 pair accumulators; finish as route 2.
 * route 6 (conv1d k3/k5, NT_N_PACKED): a chain from the bias, per channel the K taps in order; folding (K * C over
 *   32, not acc16) restarts every 32 / K channels from zero, blocks added in float32, one rounding.
 * The activation clamp applies to non-NaN values (the generator masks nonfinite lanes).
 */
#include <math.h>
#include <stdlib.h>

#include "hct_f16_mve.h"

enum
{
    ROUTE_PATCH_NT_T = 0,
    ROUTE_PATCH_PACKED = 1,
    ROUTE_DIRECT_OHWI = 2,
    ROUTE_DIRECT_PACKED = 3,
    ROUTE_SMALL_C = 4,
    ROUTE_CONV1D = 5,
    ROUTE_CONV1D_PACKED = 6
};

typedef struct
{
    const float *x; /* [N, H, W, C] */
    const float *w; /* [O, KH * KW * C] */
    const float *bias;
    int32_t H, W, C, O, KH, KW, K;
    int32_t sh, sw, ph, pw, dh, dw;
    int acc16;
} Conv;

static int tap_in_range(const Conv *cv, int32_t oy, int32_t ox, int32_t ky, int32_t kx, int32_t *iy, int32_t *ix)
{
    *iy = oy * cv->sh - cv->ph + ky * cv->dh;
    *ix = ox * cv->sw - cv->pw + kx * cv->dw;
    return *iy >= 0 && *iy < cv->H && *ix >= 0 && *ix < cv->W;
}

static const float *pixel(const Conv *cv, int32_t b, int32_t iy, int32_t ix)
{
    return cv->x + (((int64_t)b * cv->H + iy) * cv->W + ix) * cv->C;
}

/* The zero-padded patch row (arm_nn_pack_conv_patch_f16). */
static void patch_row(const Conv *cv, int32_t b, int32_t oy, int32_t ox, float *row)
{
    for (int32_t ky = 0; ky < cv->KH; ++ky)
    {
        for (int32_t kx = 0; kx < cv->KW; ++kx)
        {
            int32_t iy = 0;
            int32_t ix = 0;
            const int in = tap_in_range(cv, oy, ox, ky, kx, &iy, &ix);
            float *dst = row + ((int64_t)ky * cv->KW + kx) * cv->C;
            for (int32_t ic = 0; ic < cv->C; ++ic)
            {
                dst[ic] = in ? pixel(cv, b, iy, ix)[ic] : 0.0f;
            }
        }
    }
}

/* The in-range taps of one output, compacted in tap order: inputs into xs, filter k-indices into ks. */
static int32_t in_range_taps(const Conv *cv, int32_t b, int32_t oy, int32_t ox, float *xs, int32_t *ks)
{
    int32_t n = 0;
    for (int32_t ky = 0; ky < cv->KH; ++ky)
    {
        for (int32_t kx = 0; kx < cv->KW; ++kx)
        {
            int32_t iy = 0;
            int32_t ix = 0;
            if (!tap_in_range(cv, oy, ox, ky, kx, &iy, &ix))
            {
                continue;
            }
            const float *px = pixel(cv, b, iy, ix);
            for (int32_t ic = 0; ic < cv->C; ++ic)
            {
                xs[n] = px[ic];
                ks[n] = ((ky * cv->KW) + kx) * cv->C + ic;
                ++n;
            }
        }
    }
    return n;
}

static void zero_lanes(float lane[HCT_F16_LANES])
{
    for (int i = 0; i < HCT_F16_LANES; ++i)
    {
        lane[i] = 0.0f;
    }
}

/* Finish of the lane routes: pairs + bias in float32 (one rounding) when folded, else the float16 reduce + bias. */
static float finish_lanes(int folded, float pairs[4], const float lane[HCT_F16_LANES], int first, float bias)
{
    if (folded)
    {
        hct_f16_fold_pairs(pairs, lane, first);
        return hct_h16(bias + hct_f16_pairs_sum(pairs));
    }
    return hct_h16(bias + hct_f16_reduce8(lane));
}

/* Route 2 for output channel oc: arm_convolve_nhwc_direct_f16_body's OHWI leg. */
static int32_t direct_ohwi(const Conv *cv, int32_t b, int32_t oy, int32_t ox, int32_t oc, float *out)
{
    const float *w = cv->w + (int64_t)oc * cv->K;
    const int fold = !cv->acc16 && cv->K > HCT_F16_ACC_BLOCK;
    float lane[HCT_F16_LANES];
    float pairs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    zero_lanes(lane);
    int first = 1;
    int32_t n_vec = 0;
    int32_t n_in = 0;
    for (int32_t ky = 0; ky < cv->KH; ++ky)
    {
        for (int32_t kx = 0; kx < cv->KW; ++kx)
        {
            int32_t iy = 0;
            int32_t ix = 0;
            if (!tap_in_range(cv, oy, ox, ky, kx, &iy, &ix))
            {
                continue;
            }
            const float *px = pixel(cv, b, iy, ix);
            const float *wt = w + ((int64_t)ky * cv->KW + kx) * cv->C;
            n_in += cv->C;
            int32_t ic = 0;
            while (ic < cv->C)
            {
                int32_t ic_end = cv->C;
                if (fold)
                {
                    if (n_vec == HCT_F16_ACC_BLOCK)
                    {
                        hct_f16_fold_pairs(pairs, lane, first);
                        first = 0;
                        zero_lanes(lane);
                        n_vec = 0;
                    }
                    const int32_t n_left = (cv->C - ic + 7) / 8;
                    if (n_left > HCT_F16_ACC_BLOCK - n_vec)
                    {
                        ic_end = ic + (HCT_F16_ACC_BLOCK - n_vec) * 8;
                    }
                    n_vec += (ic_end - ic + 7) / 8;
                }
                for (; ic < ic_end; ++ic)
                {
                    HCT_TRY(hct_f16_fma(px[ic], wt[ic], lane[ic % HCT_F16_LANES], &lane[ic % HCT_F16_LANES]));
                }
            }
        }
    }
    const float bias = cv->bias ? cv->bias[oc] : 0.0f;
    *out = finish_lanes(fold && n_in > HCT_F16_ACC_BLOCK, pairs, lane, first, bias);
    return HCT_OK;
}

/* Route 5 for output (ox, oc): arm_nn_conv1d_k{3,5}_nhwc_f16 (_acc16). */
static int32_t conv1d_ohwi(const Conv *cv, int32_t b, int32_t ox, int32_t oc, float *out)
{
    const int32_t KW = cv->KW;
    const int32_t C = cv->C;
    const float *w = cv->w + (int64_t)oc * cv->K;
    const int fold = !cv->acc16 && KW * C > HCT_F16_ACC_BLOCK;
    const int32_t span = fold ? (HCT_F16_ACC_BLOCK / KW) * 8 : C;
    float lane[HCT_F16_LANES];
    float pairs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int32_t ic0 = 0; ic0 < C; ic0 += span)
    {
        const int32_t ic_end = C - ic0 > span ? ic0 + span : C;
        zero_lanes(lane);
        for (int32_t s = ic0; s < ic_end; s += 8)
        {
            for (int32_t t = 0; t < KW; ++t)
            {
                const float *px = pixel(cv, b, 0, ox + t);
                for (int32_t ic = s; ic < s + 8 && ic < ic_end; ++ic)
                {
                    float *l = &lane[ic % HCT_F16_LANES];
                    HCT_TRY(hct_f16_fma(px[ic], w[(int64_t)t * C + ic], *l, l));
                }
            }
        }
        if (fold)
        {
            hct_f16_fold_pairs(pairs, lane, ic0 == 0);
        }
    }
    const float bias = cv->bias ? cv->bias[oc] : 0.0f;
    *out = fold ? hct_h16(bias + hct_f16_pairs_sum(pairs)) : hct_h16(bias + hct_f16_reduce8(lane));
    return HCT_OK;
}

/* Route 6 for output (ox, oc): arm_nn_conv1d_k{3,5}_packed_f16 (_acc16). */
static int32_t conv1d_packed(const Conv *cv, int32_t b, int32_t ox, int32_t oc, float *out)
{
    const int32_t KW = cv->KW;
    const int32_t C = cv->C;
    const float *w = cv->w + (int64_t)oc * cv->K;
    const int fold = !cv->acc16 && KW * C > HCT_F16_ACC_BLOCK;
    const int32_t span = fold ? HCT_F16_ACC_BLOCK / KW : C;
    float acc32 = 0.0f;
    float part = 0.0f;
    for (int32_t ic0 = 0; ic0 < C; ic0 += span)
    {
        const int32_t ic_end = C - ic0 > span ? ic0 + span : C;
        part = ic0 == 0 && cv->bias ? cv->bias[oc] : 0.0f;
        for (int32_t ic = ic0; ic < ic_end; ++ic)
        {
            for (int32_t t = 0; t < KW; ++t)
            {
                HCT_TRY(hct_f16_fma(w[(int64_t)t * C + ic], pixel(cv, b, 0, ox + t)[ic], part, &part));
            }
        }
        if (fold)
        {
            acc32 = ic0 == 0 ? part : acc32 + part;
        }
    }
    *out = fold ? hct_h16(acc32) : part;
    return HCT_OK;
}

static int32_t check_shapes(const HctConvF16MveParams *p, const HctTensor *it, const HctTensor *ft,
                            const HctTensor *bt, int64_t n_bias, const HctTensor *ot)
{
    if (it->rank != 4 || ft->rank != 4 || ot->rank != 4)
    {
        return HCT_E_SHAPE;
    }
    const int32_t N = it->dims[0];
    const int32_t H = it->dims[1];
    const int32_t W = it->dims[2];
    const int32_t C = it->dims[3];
    const int32_t O = ft->dims[0];
    const int32_t KH = ft->dims[1];
    const int32_t KW = ft->dims[2];
    if (N < 1 || H < 1 || W < 1 || C < 1 || O < 1 || KH < 1 || KW < 1 || ft->dims[3] != C || ot->dims[0] != N ||
        ot->dims[1] < 1 || ot->dims[2] < 1 || ot->dims[3] != O ||
        (n_bias != 0 && (bt->rank != 1 || bt->dims[0] != O)))
    {
        return HCT_E_SHAPE;
    }
    if (p->route == ROUTE_SMALL_C && C >= HCT_F16_LANES)
    {
        return HCT_E_SHAPE;
    }
    if (p->route == ROUTE_CONV1D || p->route == ROUTE_CONV1D_PACKED)
    {
        if (N != 1 || H != 1 || KH != 1 || (KW != 3 && KW != 5) || ot->dims[1] != 1 || ot->dims[2] != W - KW + 1)
        {
            return HCT_E_SHAPE;
        }
        if (p->stride_h != 1 || p->stride_w != 1 || p->pad_h != 0 || p->pad_w != 0 || p->dilation_h != 1 ||
            p->dilation_w != 1)
        {
            return HCT_E_PARAM;
        }
    }
    return HCT_OK;
}

int32_t hct_ref_conv_f16_mve(const HctConvF16MveParams *p, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    int64_t n_in = 0;
    int64_t n_filter = 0;
    int64_t n_bias = 0;
    int64_t n_out = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], HCT_FLOAT16, &n_in));
    HCT_TRY(hct_check_tensor(&inputs[1], HCT_FLOAT16, &n_filter));
    HCT_TRY(hct_check_tensor(&inputs[2], HCT_FLOAT16, &n_bias));
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT16, &n_out));
    if (p->route < ROUTE_PATCH_NT_T || p->route > ROUTE_CONV1D_PACKED || p->stride_h < 1 || p->stride_w < 1 ||
        p->dilation_h < 1 || p->dilation_w < 1 || p->pad_h < 0 || p->pad_w < 0 || (p->acc16 != 0 && p->acc16 != 1) ||
        !(p->activation_min <= p->activation_max))
    {
        return HCT_E_PARAM;
    }
    HCT_TRY(check_shapes(p, &inputs[0], &inputs[1], &inputs[2], n_bias, &outputs[0]));
    const HctTensor *ft = &inputs[1];
    HctTensor *ot = &outputs[0];
    Conv cv = {.H = inputs[0].dims[1],
               .W = inputs[0].dims[2],
               .C = inputs[0].dims[3],
               .O = ft->dims[0],
               .KH = ft->dims[1],
               .KW = ft->dims[2],
               .sh = p->stride_h,
               .sw = p->stride_w,
               .ph = p->pad_h,
               .pw = p->pad_w,
               .dh = p->dilation_h,
               .dw = p->dilation_w,
               .acc16 = p->acc16};
    if ((int64_t)cv.KH * cv.KW * cv.C > HCT_MAX_ELEMENTS)
    {
        return HCT_E_SIZE;
    }
    cv.K = cv.KH * cv.KW * cv.C;
    const int32_t N = inputs[0].dims[0];
    const int32_t OH = ot->dims[1];
    const int32_t OW = ot->dims[2];
    const float lo = hct_h16(p->activation_min);
    const float hi = hct_h16(p->activation_max);

    float *x = malloc((size_t)n_in * sizeof(float));
    float *w = malloc((size_t)n_filter * sizeof(float));
    float *bias = malloc((size_t)(n_bias > 0 ? n_bias : 1) * sizeof(float));
    float *row = malloc((size_t)cv.K * sizeof(float));
    int32_t *ks = malloc((size_t)cv.K * sizeof(int32_t));
    float *ws = malloc((size_t)cv.K * sizeof(float));
    int32_t status = HCT_OK;
    if (x == NULL || w == NULL || bias == NULL || row == NULL || ks == NULL || ws == NULL)
    {
        status = HCT_E_SIZE;
        goto done;
    }
    for (int64_t i = 0; i < n_in; ++i)
    {
        x[i] = hct_load_f32(&inputs[0], i);
    }
    for (int64_t i = 0; i < n_filter; ++i)
    {
        w[i] = hct_load_f32(ft, i);
    }
    for (int64_t i = 0; i < n_bias; ++i)
    {
        bias[i] = hct_load_f32(&inputs[2], i);
    }
    cv.x = x;
    cv.w = w;
    cv.bias = n_bias != 0 ? bias : NULL;

    for (int32_t b = 0; b < N && status == HCT_OK; ++b)
    {
        for (int32_t oy = 0; oy < OH && status == HCT_OK; ++oy)
        {
            for (int32_t ox = 0; ox < OW && status == HCT_OK; ++ox)
            {
                int32_t n_taps = 0;
                if (p->route == ROUTE_PATCH_NT_T || p->route == ROUTE_PATCH_PACKED)
                {
                    patch_row(&cv, b, oy, ox, row);
                }
                else if (p->route == ROUTE_DIRECT_PACKED || p->route == ROUTE_SMALL_C)
                {
                    n_taps = in_range_taps(&cv, b, oy, ox, row, ks);
                }
                for (int32_t oc = 0; oc < cv.O && status == HCT_OK; ++oc)
                {
                    const float bv = cv.bias ? cv.bias[oc] : 0.0f;
                    const float *w_oc = w + (int64_t)oc * cv.K;
                    float acc = 0.0f;
                    switch (p->route)
                    {
                    case ROUTE_PATCH_NT_T:
                        status = hct_f16_nt_t_column(row, w_oc, cv.K, oc, cv.O, bv, cv.acc16, &acc);
                        break;
                    case ROUTE_PATCH_PACKED:
                        status = hct_f16_nt_n_packed_column(
                            row, w_oc, cv.K, bv, cv.acc16 ? HCT_F16_PACKED_ACC16 : HCT_F16_PACKED_FOLD, &acc);
                        break;
                    case ROUTE_DIRECT_OHWI:
                        status = direct_ohwi(&cv, b, oy, ox, oc, &acc);
                        break;
                    case ROUTE_DIRECT_PACKED:
                        for (int32_t i = 0; i < n_taps; ++i)
                        {
                            ws[i] = w_oc[ks[i]];
                        }
                        status = hct_f16_chain(row, ws, n_taps, !cv.acc16 && cv.K > HCT_F16_ACC_BLOCK, bv, &acc);
                        break;
                    case ROUTE_SMALL_C:
                    {
                        float acc32 = bv;
                        for (int32_t i = 0; i < n_taps; ++i)
                        {
                            acc32 = fmaf(row[i], w_oc[ks[i]], acc32);
                        }
                        acc = hct_h16(acc32);
                        break;
                    }
                    case ROUTE_CONV1D:
                        status = conv1d_ohwi(&cv, b, ox, oc, &acc);
                        break;
                    default:
                        status = conv1d_packed(&cv, b, ox, oc, &acc);
                        break;
                    }
                    hct_store_f32(ot, (((int64_t)b * OH + oy) * OW + ox) * cv.O + oc,
                                  hct_f16_clamp_finite(acc, lo, hi));
                }
            }
        }
    }
done:
    free(x);
    free(w);
    free(bias);
    free(row);
    free(ks);
    free(ws);
    return status;
}
