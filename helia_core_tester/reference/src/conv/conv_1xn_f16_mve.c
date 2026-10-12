/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variant: ns-cmsis-nn's arm_convolve_1_x_n_f16 / _acc16 (and the nhwc, ohwi and packed entries, and the
 * routers' 1xN branch) on Helium, bit for bit. Input [N, 1, W, C], filter OHWI [O, 1, KW, C] (unpacked; packing
 * only changes storage), bias [O] or empty, output [N, 1, OW, O]. K = KW * C; tap k of output x is
 * input[x * stride - pad + k / C][k % C], zero outside the row (the kernel packs zeros and multiplies them).
 *
 * Each output x is one of three routes, chosen as the kernel chooses them:
 *   packed filter: every row through arm_nn_mat_mult_nt_n_packed_f16 -- "chain" arithmetic for every column.
 *   x in the left/right padded region, or the no-pad rows when output_c >= 4 and
 *     ((output_c % 8 >= 4 and K >= 80) or K >= 224): arm_nn_mat_mult_nt_t_f16 -- K < 32: the gather-covered
 *     columns (blocks of 16, 8, 4) chain, the rest "lanes16"; K >= 32: "fold" (K > 32 and not acc16) or "lanes16".
 *   otherwise the no-pad rows: the strided kernel -- gather-covered columns chain (none when folding with fewer
 *     than 4 output channels), the rest "fold" (K > 32, not acc16) or "lanes16".
 * Arithmetic of one output:
 *   chain:   starts at the bias in float16, one fused multiply-add per tap in k order. Folding (K > 32, not acc16):
 *            the first 32 taps from the bias, every further 32 from zero, each block's float16 result widened and
 *            added in float32, one rounding to float16 at the end.
 *   lanes16: tap k into float16 lane k % 8 by fused multiply-add from zero, lanes reduced
 *            ((0+1)+(2+3))+((4+5)+(6+7)) in float16, then the bias added in float16.
 *   fold:    every 256 taps fresh float16 lanes as in lanes16; each block's lanes 2j, 2j+1 add in float32 into
 *            pair accumulator j; those sum (0+1)+(2+3); the bias is added in float32; one rounding.
 * The activation clamp is applied to non-NaN values (the generator masks nonfinite lanes).
 */
#include <math.h>
#include <stdlib.h>

#include "hct_ref_internal.h"

#define LANES 8
#define ACC_BLOCK 32
#define FOLD_SPAN (ACC_BLOCK * LANES)
#define GATHER_MAX_8 (65535 / 7)
#define GATHER_MAX_16 (65535 / 15)
#define GATHER_MAX_4 (65535 / 3)

enum
{
    ARITH_CHAIN = 0,
    ARITH_LANES16 = 1,
    ARITH_FOLD = 2
};

static float h(float v)
{
    return hct_round_f16(v);
}

static int32_t fma16(float a, float b, float c, float *out)
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

static int32_t chain_block(const float *lhs, const float *rhs, int32_t k0, int32_t n, float start, float *out)
{
    float acc = start;
    for (int32_t k = k0; k < k0 + n; ++k)
    {
        HCT_TRY(fma16(rhs[k], lhs[k], acc, &acc));
    }
    *out = acc;
    return HCT_OK;
}

static int32_t chain(const float *lhs, const float *rhs, int32_t K, int fold, float bias, float *out)
{
    const int32_t first_n = (fold && K > ACC_BLOCK) ? ACC_BLOCK : K;
    float acc = 0.0f;
    HCT_TRY(chain_block(lhs, rhs, 0, first_n, bias, &acc));
    if (first_n == K)
    {
        *out = acc;
        return HCT_OK;
    }
    float acc32 = acc;
    for (int32_t k0 = first_n; k0 < K; k0 += ACC_BLOCK)
    {
        const int32_t n = K - k0 < ACC_BLOCK ? K - k0 : ACC_BLOCK;
        float part = 0.0f;
        HCT_TRY(chain_block(lhs, rhs, k0, n, 0.0f, &part));
        acc32 = acc32 + part;
    }
    *out = h(acc32);
    return HCT_OK;
}

static int32_t lanes_block(const float *lhs, const float *rhs, int32_t k0, int32_t n, float lane[LANES])
{
    for (int i = 0; i < LANES; ++i)
    {
        lane[i] = 0.0f;
    }
    for (int32_t k = 0; k < n; ++k)
    {
        HCT_TRY(fma16(lhs[k0 + k], rhs[k0 + k], lane[k % LANES], &lane[k % LANES]));
    }
    return HCT_OK;
}

static int32_t lanes16(const float *lhs, const float *rhs, int32_t K, float bias, float *out)
{
    float l[LANES];
    HCT_TRY(lanes_block(lhs, rhs, 0, K, l));
    const float a = h(h(l[0] + l[1]) + h(l[2] + l[3]));
    const float b = h(h(l[4] + l[5]) + h(l[6] + l[7]));
    *out = h(bias + h(a + b));
    return HCT_OK;
}

static int32_t fold_lanes(const float *lhs, const float *rhs, int32_t K, float bias, float *out)
{
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int32_t k0 = 0; k0 < K; k0 += FOLD_SPAN)
    {
        const int32_t n = K - k0 < FOLD_SPAN ? K - k0 : FOLD_SPAN;
        float l[LANES];
        HCT_TRY(lanes_block(lhs, rhs, k0, n, l));
        for (int j = 0; j < 4; ++j)
        {
            const float pair = l[2 * j] + l[2 * j + 1];
            acc[j] = k0 == 0 ? pair : acc[j] + pair;
        }
    }
    const float total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    *out = h(bias + total);
    return HCT_OK;
}

/* Columns [0, result) the 16/8/4-lane gather kernels cover; 0 when K exceeds the eight-lane gather stride. */
static int32_t gather_columns(int32_t K, int32_t N)
{
    if (K > GATHER_MAX_8)
    {
        return 0;
    }
    int32_t c = 0;
    if (K <= GATHER_MAX_16)
    {
        c = N - N % 16;
    }
    c += (N - c) - (N - c) % 8;
    if (K <= GATHER_MAX_4)
    {
        c += (N - c) - (N - c) % 4;
    }
    return c;
}

static float clamp_finite(float v, float lo, float hi)
{
    return isnan(v) ? v : (v < lo ? lo : (v > hi ? hi : v));
}

/* arm_convolve_1_x_n_find_regions: [first_valid, first_valid + no_pad) are the outputs with every tap in range. */
static void find_regions(int32_t out_w, int32_t in_w, int32_t kw, int32_t stride, int32_t pad, int32_t *first,
                         int32_t *no_pad)
{
    int32_t f = 0;
    while (f < out_w)
    {
        const int64_t base = (int64_t)f * stride - pad;
        if (base >= 0 && base + kw <= in_w)
        {
            break;
        }
        ++f;
    }
    int32_t l = out_w - 1;
    while (l >= f)
    {
        const int64_t base = (int64_t)l * stride - pad;
        if (base >= 0 && base + kw <= in_w)
        {
            break;
        }
        --l;
    }
    *first = f;
    *no_pad = l - f + 1 > 0 ? l - f + 1 : 0;
}

int32_t hct_ref_conv_1xn_f16_mve(const HctConv1xNF16MveParams *p, const HctTensor *inputs, int32_t num_inputs,
                                 HctTensor *outputs, int32_t num_outputs)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    int64_t n = 0;
    int64_t n_bias = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], HCT_FLOAT16, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], HCT_FLOAT16, &n));
    HCT_TRY(hct_check_tensor(&inputs[2], HCT_FLOAT16, &n_bias));
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT16, &n));
    if (p->stride_w < 1 || p->pad_w < 0 || (p->acc16 != 0 && p->acc16 != 1) || (p->packed != 0 && p->packed != 1) ||
        !(p->activation_min <= p->activation_max))
    {
        return HCT_E_PARAM;
    }
    const HctTensor *it = &inputs[0];
    const HctTensor *ft = &inputs[1];
    HctTensor *ot = &outputs[0];
    if (it->rank != 4 || ft->rank != 4 || ot->rank != 4)
    {
        return HCT_E_SHAPE;
    }
    const int32_t B = it->dims[0];
    const int32_t W = it->dims[2];
    const int32_t C = it->dims[3];
    const int32_t O = ft->dims[0];
    const int32_t KW = ft->dims[2];
    const int32_t OW = ot->dims[2];
    const int has_bias = n_bias != 0;
    /* The 1xN shape: heights 1, a filter wider than 1, filter depth equal to the input depth. */
    if (it->dims[1] != 1 || ft->dims[1] != 1 || ot->dims[1] != 1 || KW < 2 || ft->dims[3] != C || C < 1 || O < 1 ||
        W < 1 || OW < 1 || B < 1 || ot->dims[0] != B || ot->dims[3] != O ||
        (has_bias && (inputs[2].rank != 1 || inputs[2].dims[0] != O)))
    {
        return HCT_E_SHAPE;
    }
    if ((int64_t)KW * C > HCT_MAX_ELEMENTS)
    {
        return HCT_E_SIZE;
    }
    const int32_t K = KW * C;
    const int fold = !p->acc16 && K > ACC_BLOCK;
    const float lo = h(p->activation_min);
    const float hi = h(p->activation_max);
    int32_t first = 0;
    int32_t no_pad = 0;
    find_regions(OW, W, KW, p->stride_w, p->pad_w, &first, &no_pad);
    const int in_place = O >= 4 && ((O % 8 >= 4 && K >= 80) || K >= 224);

    /* Per route, the columns the gather kernels cover and the arithmetic of the rest. */
    const int32_t matmul_gather = K < 32 ? gather_columns(K, O) : 0;
    const int matmul_rest = fold ? ARITH_FOLD : ARITH_LANES16;
    const int32_t strided_gather = (fold && O < 4) ? 0 : gather_columns(K, O);

    float *lhs = malloc((size_t)K * sizeof(float));
    float *rhs = malloc((size_t)K * sizeof(float));
    if (lhs == NULL || rhs == NULL)
    {
        free(lhs);
        free(rhs);
        return HCT_E_SIZE;
    }
    int32_t status = HCT_OK;
    for (int32_t b = 0; b < B && status == HCT_OK; ++b)
    {
        for (int32_t x = 0; x < OW && status == HCT_OK; ++x)
        {
            const int64_t base = (int64_t)x * p->stride_w - p->pad_w;
            for (int32_t k = 0; k < K; ++k)
            {
                const int64_t ix = base + k / C;
                lhs[k] = (ix >= 0 && ix < W) ? hct_load_f32(it, ((int64_t)b * W + ix) * C + k % C) : 0.0f;
            }
            const int padded = x < first || x >= first + no_pad;
            int32_t gather = 0;
            int rest = matmul_rest;
            if (p->packed)
            {
                gather = O;
            }
            else if (padded || in_place)
            {
                gather = matmul_gather;
            }
            else
            {
                gather = strided_gather;
            }
            for (int32_t c = 0; c < O && status == HCT_OK; ++c)
            {
                for (int32_t k = 0; k < K; ++k)
                {
                    rhs[k] = hct_load_f32(ft, (int64_t)c * K + k);
                }
                const float bias = has_bias ? hct_load_f32(&inputs[2], c) : 0.0f;
                const int arith = c < gather ? ARITH_CHAIN : rest;
                float acc = 0.0f;
                if (arith == ARITH_CHAIN)
                {
                    status = chain(lhs, rhs, K, fold, bias, &acc);
                }
                else if (arith == ARITH_FOLD)
                {
                    status = fold_lanes(lhs, rhs, K, bias, &acc);
                }
                else
                {
                    status = lanes16(lhs, rhs, K, bias, &acc);
                }
                hct_store_f32(ot, (((int64_t)b * OW) + x) * O + c, clamp_finite(acc, lo, hi));
            }
        }
    }
    free(lhs);
    free(rhs);
    return status;
}
