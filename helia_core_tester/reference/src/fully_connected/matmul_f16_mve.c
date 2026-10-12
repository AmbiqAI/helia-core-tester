/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variant: ns-cmsis-nn's arm_nn_mat_mult_nt_t_f16 / _acc16 on Helium, bit for bit, so a float16
 * golden matches the kernel within one rounding instead of within its accumulation error. lhs [B, M, K],
 * rhs [B, N, K] (row c of rhs is output column c), bias [N] or empty, output [B, M, N].
 *
 * routes 2 and 3 are arm_batch_matmul_f16's scalar adjoint loop (strides past the gather limit):
 * acc += a * b in float16 from zero, fused (2) or with the product rounded first (3).
 * route 1 is arm_batch_matmul_f16's adjoint path: every output is one row dot product in float16 lanes
 * (lane k % 8), reduced in float16, with no bias and no fold at any K.
 *
 * route 0, per output row, depends on K and N:
 *   K >= 32: columns in groups of 4 dot contiguous rows; element k sits in lane k % 8 of a float16
 *     accumulator (fused multiply-add, one float16 rounding each). Without folding (acc16, or K == 32)
 *     the lanes reduce in float16 as ((0+1)+(2+3))+((4+5)+(6+7)) and the bias is added in float16.
 *     With folding (K > 32), every block of 256 elements gets fresh lanes, whose adjacent pairs widen to
 *     float32 and add into four float32 accumulators; those sum as (0+1)+(2+3), the bias is added in
 *     float32 and the total rounds to float16 once.
 *   K < 32: columns in blocks of 16, 8, then 4 (as the gather stride limits allow) start from the bias
 *     and take one fused multiply-add per k, in k order, rounding to float16 each time.
 *   Remaining columns: one row dot product as above (fold or float16 reduce plus bias).
 * The activation clamp is applied to finite values; NaN passes (nonfinite lanes are masked by the
 * generator, and the kernel's NaN clamping differs between its paths).
 */
#include <math.h>
#include <stdlib.h>

#include "hct_ref_internal.h"

#define LANES 8
#define FOLD_SPAN (32 * LANES) /* ARM_NN_F16_ACC_BLOCK taps per lane */
#define GATHER_MAX_8 (65535 / 7)
#define GATHER_MAX_16 (65535 / 15)
#define GATHER_MAX_4 (65535 / 3)

static float h(float v)
{
    return hct_round_f16(v);
}

/* One float16 fused multiply-add: a * b + c rounded once (exactly, through the exact accumulator). */
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

/* Elements [k0, k0 + n) of a row dot product into fresh float16 lanes (lane = (k - k0) % 8). */
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

static float reduce16(const float l[LANES])
{
    const float a = h(h(l[0] + l[1]) + h(l[2] + l[3]));
    const float b = h(h(l[4] + l[5]) + h(l[6] + l[7]));
    return h(a + b);
}

/* Row dot product with float16 lanes throughout, reduced in float16, plus the bias in float16. */
static int32_t dot_lanes16(const float *lhs, const float *rhs, int32_t K, int has_bias, float bias, float *out)
{
    float lane[LANES];
    HCT_TRY(lanes_block(lhs, rhs, 0, K, lane));
    *out = h((has_bias ? bias : 0.0f) + reduce16(lane));
    return HCT_OK;
}

/* Row dot product folded into float32 every FOLD_SPAN elements, bias in float32, one rounding. */
static int32_t dot_fold(const float *lhs, const float *rhs, int32_t K, int has_bias, float bias, float *out)
{
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int first = 1;
    for (int32_t k0 = 0; k0 < K; k0 += FOLD_SPAN)
    {
        const int32_t n = K - k0 < FOLD_SPAN ? K - k0 : FOLD_SPAN;
        float lane[LANES];
        HCT_TRY(lanes_block(lhs, rhs, k0, n, lane));
        for (int j = 0; j < 4; ++j)
        {
            const float pair = lane[2 * j] + lane[2 * j + 1];
            acc[j] = first ? pair : acc[j] + pair;
        }
        first = 0;
    }
    const float total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    *out = h((has_bias ? bias : 0.0f) + total);
    return HCT_OK;
}

static float clamp_finite(float v, float lo, float hi)
{
    return isnan(v) ? v : (v < lo ? lo : (v > hi ? hi : v));
}

int32_t hct_ref_matmul_nt_t_f16_mve(const HctMatMulF16MveParams *p, const HctTensor *inputs, int32_t num_inputs,
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
    if (p->route < 0 || p->route > 3 || (p->acc16 != 0 && p->acc16 != 1) ||
        !(p->activation_min <= p->activation_max) || (p->route != 0 && n_bias != 0))
    {
        return HCT_E_PARAM;
    }
    const HctTensor *lt = &inputs[0];
    const HctTensor *rt = &inputs[1];
    HctTensor *ot = &outputs[0];
    if (lt->rank != 3 || rt->rank != 3 || ot->rank != 3)
    {
        return HCT_E_SHAPE;
    }
    const int32_t B = lt->dims[0];
    const int32_t M = lt->dims[1];
    const int32_t K = lt->dims[2];
    const int32_t N = rt->dims[1];
    const int has_bias = n_bias != 0;
    if (rt->dims[0] != B || rt->dims[2] != K || ot->dims[0] != B || ot->dims[1] != M || ot->dims[2] != N || K < 1 ||
        (has_bias && (inputs[2].rank != 1 || inputs[2].dims[0] != N)))
    {
        return HCT_E_SHAPE;
    }
    const float lo = h(p->activation_min);
    const float hi = h(p->activation_max);
    const int fold = p->route == 0 && !p->acc16 && K > 32;
    float *lhs = malloc((size_t)K * sizeof(float));
    float *rhs_row = malloc((size_t)K * sizeof(float));
    if (lhs == NULL || rhs_row == NULL)
    {
        free(lhs);
        free(rhs_row);
        return HCT_E_SIZE;
    }
    int32_t status = HCT_OK;
    for (int32_t b = 0; b < B && status == HCT_OK; ++b)
    {
        for (int32_t r = 0; r < M && status == HCT_OK; ++r)
        {
            for (int32_t k = 0; k < K; ++k)
            {
                lhs[k] = hct_load_f32(lt, ((int64_t)b * M + r) * K + k);
            }
            int32_t c = 0;
            /* Columns the gather blocks cover when K < 32. For K >= 32 the grouped and the single-row dot
             * products round identically, so only the fold decides. */
            int32_t blocked = 0;
            if (p->route == 0 && K < 32 && K <= GATHER_MAX_8)
            {
                int32_t cc = 0;
                if (K <= GATHER_MAX_16)
                {
                    cc = N - N % 16;
                }
                cc += (N - cc) - (N - cc) % 8;
                if (K <= GATHER_MAX_4)
                {
                    cc += (N - cc) - (N - cc) % 4;
                }
                blocked = cc;
            }
            for (; c < N && status == HCT_OK; ++c)
            {
                for (int32_t k = 0; k < K; ++k)
                {
                    rhs_row[k] = hct_load_f32(rt, ((int64_t)b * N + c) * K + k);
                }
                const float bias = has_bias ? hct_load_f32(&inputs[2], c) : 0.0f;
                float acc = 0.0f;
                if (c < blocked && K < 32)
                {
                    /* Gather lanes: the column's lane starts at its bias, one fma per k. */
                    acc = has_bias ? bias : 0.0f;
                    for (int32_t k = 0; k < K && status == HCT_OK; ++k)
                    {
                        status = fma16(rhs_row[k], lhs[k], acc, &acc);
                    }
                }
                else if (p->route >= 2)
                {
                    /* The scalar loop: acc += a * b in float16, fused (route 2) or rounded twice (route 3). */
                    for (int32_t k = 0; k < K && status == HCT_OK; ++k)
                    {
                        if (p->route == 2)
                        {
                            status = fma16(lhs[k], rhs_row[k], acc, &acc);
                        }
                        else
                        {
                            acc = h(acc + h(lhs[k] * rhs_row[k]));
                        }
                    }
                }
                else if (fold)
                {
                    status = dot_fold(lhs, rhs_row, K, has_bias, bias, &acc);
                }
                else
                {
                    status = dot_lanes16(lhs, rhs_row, K, has_bias, bias, &acc);
                }
                hct_store_f32(ot, ((int64_t)b * M + r) * N + c, clamp_finite(acc, lo, hi));
            }
        }
    }
    free(lhs);
    free(rhs_row);
    return status;
}
