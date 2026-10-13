/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared float16 Helium arithmetic of the kernel-faithful variants (see hct_f16_mve.h).
 */
#include <math.h>

#include "hct_f16_mve.h"

int32_t hct_f16_fma(float a, float b, float c, float *out)
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

int32_t hct_f16_lanes(const float *a, const float *b, int32_t k0, int32_t n, float lane[HCT_F16_LANES])
{
    for (int i = 0; i < HCT_F16_LANES; ++i)
    {
        lane[i] = 0.0f;
    }
    for (int32_t k = 0; k < n; ++k)
    {
        HCT_TRY(hct_f16_fma(a[k0 + k], b[k0 + k], lane[k % HCT_F16_LANES], &lane[k % HCT_F16_LANES]));
    }
    return HCT_OK;
}

float hct_f16_reduce8(const float l[HCT_F16_LANES])
{
    const float a = hct_h16(hct_h16(l[0] + l[1]) + hct_h16(l[2] + l[3]));
    const float b = hct_h16(hct_h16(l[4] + l[5]) + hct_h16(l[6] + l[7]));
    return hct_h16(a + b);
}

void hct_f16_fold_pairs(float pairs[4], const float l[HCT_F16_LANES], int first)
{
    for (int j = 0; j < 4; ++j)
    {
        const float pair = l[2 * j] + l[2 * j + 1];
        pairs[j] = first ? pair : pairs[j] + pair;
    }
}

float hct_f16_pairs_sum(const float pairs[4])
{
    return (pairs[0] + pairs[1]) + (pairs[2] + pairs[3]);
}

int32_t hct_f16_dot_lanes16(const float *a, const float *b, int32_t K, float bias, float *out)
{
    float l[HCT_F16_LANES];
    HCT_TRY(hct_f16_lanes(a, b, 0, K, l));
    *out = hct_h16(bias + hct_f16_reduce8(l));
    return HCT_OK;
}

int32_t hct_f16_dot_fold(const float *a, const float *b, int32_t K, float bias, float *out)
{
    float pairs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int32_t k0 = 0; k0 < K; k0 += HCT_F16_FOLD_SPAN)
    {
        const int32_t n = K - k0 < HCT_F16_FOLD_SPAN ? K - k0 : HCT_F16_FOLD_SPAN;
        float l[HCT_F16_LANES];
        HCT_TRY(hct_f16_lanes(a, b, k0, n, l));
        hct_f16_fold_pairs(pairs, l, k0 == 0);
    }
    *out = hct_h16(bias + hct_f16_pairs_sum(pairs));
    return HCT_OK;
}

static int32_t chain_block(const float *a, const float *b, int32_t k0, int32_t n, float start, float *out)
{
    float acc = start;
    for (int32_t k = k0; k < k0 + n; ++k)
    {
        HCT_TRY(hct_f16_fma(b[k], a[k], acc, &acc));
    }
    *out = acc;
    return HCT_OK;
}

int32_t hct_f16_chain(const float *a, const float *b, int32_t K, int fold, float start, float *out)
{
    const int32_t first_n = (fold && K > HCT_F16_ACC_BLOCK) ? HCT_F16_ACC_BLOCK : K;
    float acc = 0.0f;
    HCT_TRY(chain_block(a, b, 0, first_n, start, &acc));
    if (first_n == K)
    {
        *out = acc;
        return HCT_OK;
    }
    float acc32 = acc;
    for (int32_t k0 = first_n; k0 < K; k0 += HCT_F16_ACC_BLOCK)
    {
        const int32_t n = K - k0 < HCT_F16_ACC_BLOCK ? K - k0 : HCT_F16_ACC_BLOCK;
        float part = 0.0f;
        HCT_TRY(chain_block(a, b, k0, n, 0.0f, &part));
        acc32 = acc32 + part;
    }
    *out = hct_h16(acc32);
    return HCT_OK;
}

int32_t hct_f16_gather_columns(int32_t K, int32_t N)
{
    if (K > HCT_F16_GATHER_MAX_8)
    {
        return 0;
    }
    int32_t c = 0;
    if (K <= HCT_F16_GATHER_MAX_16)
    {
        c = N - N % 16;
    }
    c += (N - c) - (N - c) % 8;
    if (K <= HCT_F16_GATHER_MAX_4)
    {
        c += (N - c) - (N - c) % 4;
    }
    return c;
}

int32_t hct_f16_nt_t_column(const float *lhs, const float *rhs, int32_t K, int32_t c, int32_t N, float bias,
                            int acc16, float *out)
{
    if (K < HCT_F16_ACC_BLOCK && c < hct_f16_gather_columns(K, N))
    {
        return hct_f16_chain(lhs, rhs, K, 0, bias, out);
    }
    if (!acc16 && K > HCT_F16_ACC_BLOCK)
    {
        return hct_f16_dot_fold(lhs, rhs, K, bias, out);
    }
    return hct_f16_dot_lanes16(lhs, rhs, K, bias, out);
}

int32_t hct_f16_nt_n_packed_column(const float *lhs, const float *rhs, int32_t K, float bias, int32_t policy,
                                   float *out)
{
    if (policy == HCT_F16_PACKED_PRECISE)
    {
        /* From zero, folding every four taps (the first block sets the lanes, a K below four adds into zero); the
         * bias widens and adds into the float32 lane; one rounding. */
        float acc32 = 0.0f;
        for (int32_t k0 = 0; k0 < K; k0 += 4)
        {
            const int32_t n = K - k0 < 4 ? K - k0 : 4;
            float part = 0.0f;
            HCT_TRY(chain_block(lhs, rhs, k0, n, 0.0f, &part));
            acc32 = (k0 + n == 4) ? part : acc32 + part;
        }
        *out = hct_h16(acc32 + bias);
        return HCT_OK;
    }
    return hct_f16_chain(lhs, rhs, K, policy == HCT_F16_PACKED_FOLD, bias, out);
}

float hct_f16_clamp_finite(float v, float lo, float hi)
{
    return isnan(v) ? v : (v < lo ? lo : (v > hi ? hi : v));
}
