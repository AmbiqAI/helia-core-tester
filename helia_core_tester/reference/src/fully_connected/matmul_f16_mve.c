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
#include <stdlib.h>

#include "hct_f16_mve.h"

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
    const float lo = hct_h16(p->activation_min);
    const float hi = hct_h16(p->activation_max);
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
            for (int32_t c = 0; c < N && status == HCT_OK; ++c)
            {
                for (int32_t k = 0; k < K; ++k)
                {
                    rhs_row[k] = hct_load_f32(rt, ((int64_t)b * N + c) * K + k);
                }
                const float bias = has_bias ? hct_load_f32(&inputs[2], c) : 0.0f;
                float acc = 0.0f;
                if (p->route == 0)
                {
                    /* For K >= 32 the grouped and the single-row dot products round identically, so only the
                     * gather blocks (K < 32) and the fold decide. */
                    status = hct_f16_nt_t_column(lhs, rhs_row, K, c, N, bias, p->acc16, &acc);
                }
                else if (p->route >= 2)
                {
                    /* The scalar loop: acc += a * b in float16, fused (route 2) or rounded twice (route 3). */
                    for (int32_t k = 0; k < K && status == HCT_OK; ++k)
                    {
                        if (p->route == 2)
                        {
                            status = hct_f16_fma(lhs[k], rhs_row[k], acc, &acc);
                        }
                        else
                        {
                            acc = hct_h16(acc + hct_h16(lhs[k] * rhs_row[k]));
                        }
                    }
                }
                else
                {
                    status = hct_f16_dot_lanes16(lhs, rhs_row, K, 0.0f, &acc);
                }
                hct_store_f32(ot, ((int64_t)b * M + r) * N + c, hct_f16_clamp_finite(acc, lo, hi));
            }
        }
    }
    free(lhs);
    free(rhs_row);
    return status;
}
