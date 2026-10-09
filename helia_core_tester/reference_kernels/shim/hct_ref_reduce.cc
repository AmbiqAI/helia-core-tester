/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Reduction entries of the hct_ref shim (see hct_ref.h): integer Mean as TFLM's
 * micro EvalIntegerMean runs it.
 */
#include <algorithm>
#include <cstdint>
#include <limits>

#include "hct_ref.h"
#include "hct_ref_internal.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"

namespace {

using namespace hct;

template <typename T>
int32_t mean(const HctMeanParams *params,
             const HctShape *input_shape,
             const T *input,
             const int32_t *axes,
             int32_t num_axes,
             const HctShape *output_shape,
             T *output)
{
    if (params == nullptr || axes == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_shape(input_shape, 0));
    HCT_TRY(check_shape(output_shape, 0));
    HCT_TRY(check_buffers(input, output));
    if (num_axes < 1 || num_axes > input_shape->rank)
    {
        return HCT_REF_E_PARAM;
    }
    bool reduced[HCT_REF_MAX_RANK] = {};
    for (int32_t i = 0; i < num_axes; ++i)
    {
        const int32_t a = axes[i] < 0 ? axes[i] + input_shape->rank : axes[i];
        if (a < 0 || a >= input_shape->rank)
        {
            return HCT_REF_E_PARAM;
        }
        reduced[a] = true;
    }
    // The output shape must be the reduction of the input shape.
    int32_t expected_rank = 0;
    int32_t expected[HCT_REF_MAX_RANK] = {};
    for (int32_t i = 0; i < input_shape->rank; ++i)
    {
        if (!reduced[i])
        {
            expected[expected_rank++] = input_shape->dims[i];
        }
        else if (params->keep_dims)
        {
            expected[expected_rank++] = 1;
        }
    }
    if (expected_rank == 0)
    {
        expected[expected_rank++] = 1;
    }
    if (output_shape->rank != expected_rank)
    {
        return HCT_REF_E_DIMS;
    }
    for (int32_t i = 0; i < expected_rank; ++i)
    {
        if (output_shape->dims[i] != expected[i])
        {
            return HCT_REF_E_DIMS;
        }
    }
    if (sizeof(T) == 2 && (params->input_zero_point != 0 || params->output_zero_point != 0))
    {
        return HCT_REF_E_PARAM;
    }
    if (params->input_zero_point < std::numeric_limits<T>::min() ||
        params->input_zero_point > std::numeric_limits<T>::max() ||
        params->output_zero_point < std::numeric_limits<T>::min() ||
        params->output_zero_point > std::numeric_limits<T>::max() || params->multiplier < 0)
    {
        return HCT_REF_E_PARAM;
    }
    int temp_index[HCT_REF_MAX_RANK];
    int resolved_axis[HCT_REF_MAX_RANK];
    int axis_buf[HCT_REF_MAX_RANK];
    for (int32_t i = 0; i < num_axes; ++i)
    {
        axis_buf[i] = axes[i];
    }
    const int64_t n_out = flat_size(output_shape);
    int32_t *temp_sum = new int32_t[static_cast<size_t>(n_out)];
    const bool ok = tflite::reference_ops::QuantizedMeanOrSum<T, int32_t>(
        input, params->input_zero_point, input_shape->dims, input_shape->rank, output, params->multiplier,
        params->shift, params->output_zero_point, output_shape->dims, output_shape->rank, axis_buf, num_axes,
        params->keep_dims != 0, temp_index, resolved_axis, temp_sum, /*compute_sum=*/false);
    delete[] temp_sum;
    return ok ? HCT_REF_OK : HCT_REF_E_PARAM;
}

} // namespace

extern "C" {

int32_t hct_ref_mean_fold(int32_t multiplier, int32_t shift, int64_t count, HctQuant *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (multiplier < 0 || count < 1 || shift < -31 || shift > 30)
    {
        return HCT_REF_E_PARAM;
    }
    // QuantizedMeanOrSum's 1/count fold (kernels/internal/reference/reduce.h).
    int fold = 63 - tflite::CountLeadingZeros(static_cast<uint64_t>(count));
    fold = std::min(fold, 32);
    fold = std::min(fold, 31 + shift);
    out->multiplier = static_cast<int32_t>((static_cast<int64_t>(multiplier) << fold) / count);
    out->shift = shift - fold;
    return HCT_REF_OK;
}

int32_t hct_ref_mean_s8(const HctMeanParams *params,
                        const HctShape *input_shape,
                        const int8_t *input,
                        const int32_t *axes,
                        int32_t num_axes,
                        const HctShape *output_shape,
                        int8_t *output)
{
    return mean<int8_t>(params, input_shape, input, axes, num_axes, output_shape, output);
}

int32_t hct_ref_mean_s16(const HctMeanParams *params,
                         const HctShape *input_shape,
                         const int16_t *input,
                         const int32_t *axes,
                         int32_t num_axes,
                         const HctShape *output_shape,
                         int16_t *output)
{
    return mean<int16_t>(params, input_shape, input, axes, num_axes, output_shape, output);
}

} // extern "C"
