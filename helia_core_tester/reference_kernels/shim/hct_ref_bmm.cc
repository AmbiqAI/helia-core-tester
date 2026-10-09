/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * BatchMatMul entries of the hct_ref shim (see hct_ref.h), called as TFLM's micro
 * EvalInt8/EvalInt16/EvalFloat call reference_ops::BatchMatMul: RHS first, both
 * operands with the accumulation depth innermost.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "hct_ref.h"
#include "hct_ref_internal.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/batch_matmul.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace {

using namespace hct;

// lhs [..., M, K], rhs [..., N, K], out [..., M, N]; batch dims broadcast (rank <= 5).
int32_t check_bmm_shapes(const HctShape *lhs, const HctShape *rhs, const HctShape *out)
{
    HCT_TRY(check_shape(lhs, 0));
    HCT_TRY(check_shape(rhs, 0));
    HCT_TRY(check_shape(out, 0));
    if (lhs->rank < 2 || rhs->rank < 2 || lhs->rank > 5 || rhs->rank > 5)
    {
        return HCT_REF_E_UNSUPPORTED;
    }
    const int32_t rank = std::max(lhs->rank, rhs->rank);
    if (out->rank != rank)
    {
        return HCT_REF_E_DIMS;
    }
    const int32_t m = lhs->dims[lhs->rank - 2];
    const int32_t k = lhs->dims[lhs->rank - 1];
    const int32_t n = rhs->dims[rhs->rank - 2];
    if (rhs->dims[rhs->rank - 1] != k || out->dims[rank - 2] != m || out->dims[rank - 1] != n)
    {
        return HCT_REF_E_DIMS;
    }
    for (int32_t i = 0; i < rank - 2; ++i)
    {
        const int32_t il = i - (rank - lhs->rank);
        const int32_t ir = i - (rank - rhs->rank);
        const int32_t dl = il >= 0 ? lhs->dims[il] : 1;
        const int32_t dr = ir >= 0 ? rhs->dims[ir] : 1;
        if ((dl != dr && dl != 1 && dr != 1) || out->dims[i] != std::max(dl, dr))
        {
            return HCT_REF_E_DIMS;
        }
    }
    return HCT_REF_OK;
}

// BatchMatMulEval hands the LHS over with its row/column dims swapped (SwapRowColumnDims)
// while its data stays [M, K]; the RHS goes as [N, K], data and shape alike.
HctShape swap_lhs(const HctShape *lhs_shape)
{
    HctShape swapped = *lhs_shape;
    swapped.dims[swapped.rank - 2] = lhs_shape->dims[lhs_shape->rank - 1];
    swapped.dims[swapped.rank - 1] = lhs_shape->dims[lhs_shape->rank - 2];
    return swapped;
}

template <typename T, typename AccumT>
int32_t bmm(const HctBmmParams *params,
            const HctShape *lhs_shape,
            const T *lhs,
            const HctShape *rhs_shape,
            const T *rhs,
            const HctShape *output_shape,
            T *output)
{
    if (params == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_bmm_shapes(lhs_shape, rhs_shape, output_shape));
    HCT_TRY(check_buffers(lhs, rhs, output));
    HCT_TRY(check_activation<T>(params->act));
    if (sizeof(T) == 2)
    {
        if (params->lhs_offset != 0 || params->rhs_offset != 0 || params->output_offset != 0)
        {
            return HCT_REF_E_PARAM;
        }
    }
    else if (params->lhs_offset < -127 || params->lhs_offset > 128 || params->rhs_offset < -127 ||
             params->rhs_offset > 128 || params->output_offset < -128 || params->output_offset > 127)
    {
        return HCT_REF_E_PARAM;
    }
    if (params->output_multiplier < 0 || params->output_shift < -31 || params->output_shift > 30)
    {
        return HCT_REF_E_PARAM;
    }
    tflite::FullyConnectedParams p = {};
    p.input_offset = params->lhs_offset;
    p.weights_offset = params->rhs_offset;
    p.output_offset = params->output_offset;
    p.output_multiplier = params->output_multiplier;
    p.output_shift = params->output_shift;
    p.quantized_activation_min = params->act.min;
    p.quantized_activation_max = params->act.max;
    const HctShape lhs_swapped = swap_lhs(lhs_shape);
    tflite::reference_ops::BatchMatMul<T, AccumT>(p, to_runtime(rhs_shape), rhs, to_runtime(&lhs_swapped), lhs,
                                                  to_runtime(output_shape), output);
    return HCT_REF_OK;
}

} // namespace

extern "C" {

int32_t hct_ref_bmm_s8(const HctBmmParams *params,
                       const HctShape *lhs_shape,
                       const int8_t *lhs,
                       const HctShape *rhs_shape,
                       const int8_t *rhs,
                       const HctShape *output_shape,
                       int8_t *output)
{
    return bmm<int8_t, int32_t>(params, lhs_shape, lhs, rhs_shape, rhs, output_shape, output);
}

int32_t hct_ref_bmm_s16(const HctBmmParams *params,
                        const HctShape *lhs_shape,
                        const int16_t *lhs,
                        const HctShape *rhs_shape,
                        const int16_t *rhs,
                        const HctShape *output_shape,
                        int16_t *output)
{
    return bmm<int16_t, int64_t>(params, lhs_shape, lhs, rhs_shape, rhs, output_shape, output);
}

int32_t hct_ref_bmm_f32(const HctBmmParams *params,
                        const HctShape *lhs_shape,
                        const float *lhs,
                        const HctShape *rhs_shape,
                        const float *rhs,
                        const HctShape *output_shape,
                        float *output)
{
    if (params == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_bmm_shapes(lhs_shape, rhs_shape, output_shape));
    HCT_TRY(check_buffers(lhs, rhs, output));
    HCT_TRY(check_activation<float>(params->act));
    if (params->lhs_offset != 0 || params->rhs_offset != 0 || params->output_offset != 0 ||
        params->output_multiplier != 0 || params->output_shift != 0)
    {
        return HCT_REF_E_PARAM;
    }
    const HctShape lhs_swapped = swap_lhs(lhs_shape);
    tflite::reference_ops::BatchMatMul<float, float, float>(to_runtime(rhs_shape), rhs, to_runtime(&lhs_swapped), lhs,
                                                            to_runtime(output_shape), output);
    // TFLM's BATCH_MATMUL has no fused activation; a finite bound clamps as the CMSIS kernel does.
    if (std::isfinite(params->act.fmin) || std::isfinite(params->act.fmax))
    {
        int64_t count = 1;
        for (int32_t i = 0; i < output_shape->rank; ++i)
        {
            count *= output_shape->dims[i];
        }
        for (int64_t i = 0; i < count; ++i)
        {
            output[i] = tflite::ActivationFunctionWithMinMax(output[i], params->act.fmin, params->act.fmax);
        }
    }
    return HCT_REF_OK;
}

} // extern "C"
