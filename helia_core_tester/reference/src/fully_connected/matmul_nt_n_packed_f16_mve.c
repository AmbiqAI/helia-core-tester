/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variant: ns-cmsis-nn's arm_nn_mat_mult_nt_n_packed_f16 / _acc16 / _precise on Helium, bit for bit.
 * lhs [B, M, K], rhs [B, N, K] (row c is output column c, i.e. the weights before NT_N packing; packing only
 * changes storage), bias [N] or empty, output [B, M, N].
 *
 * Lanes are output columns, so every output is one chain of fused float16 multiply-adds in k order:
 *   policy 0 (the plain entry): from the bias; past ARM_NN_F16_ACC_BLOCK (32) taps the first 32 run from the bias,
 *     every further 32 from zero, each block's float16 result widens and adds in float32, one rounding at the end.
 *   policy 1 (_acc16): from the bias, float16 throughout.
 *   policy 2 (_precise): from zero, blocks of 4 taps folded into float32, the bias added in float32, one rounding.
 * The four-row and single-row blocks of the kernel round identically. The clamp applies to non-NaN values.
 */
#include <stdlib.h>

#include "hct_f16_mve.h"

int32_t hct_ref_matmul_nt_n_packed_f16_mve(const HctMatMulNtNPackedF16MveParams *p, const HctTensor *inputs,
                                           int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
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
    if (p->policy < HCT_F16_PACKED_FOLD || p->policy > HCT_F16_PACKED_PRECISE ||
        !(p->activation_min <= p->activation_max))
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
                    rhs[k] = hct_load_f32(rt, ((int64_t)b * N + c) * K + k);
                }
                const float bias = has_bias ? hct_load_f32(&inputs[2], c) : 0.0f;
                float acc = 0.0f;
                status = hct_f16_nt_n_packed_column(lhs, rhs, K, bias, p->policy, &acc);
                hct_store_f32(ot, ((int64_t)b * M + r) * N + c, hct_f16_clamp_finite(acc, lo, hi));
            }
        }
    }
    free(lhs);
    free(rhs);
    return status;
}
