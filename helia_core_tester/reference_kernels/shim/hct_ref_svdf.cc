/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Integer SVDF entries of the hct_ref shim (see hct_ref.h): a port of TFLM's
 * EvalIntegerSvdfReference (micro/kernels/svdf_common.cc) over raw pointers,
 * with the activation-state zero point fixed at 0 as the CMSIS-NN kernels assume.
 */
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "hct_ref.h"
#include "hct_ref_internal.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace {

using namespace hct;

template <typename T>
int32_t svdf(const HctSvdfParams *p,
             const int8_t *input,
             const int8_t *weights_feature,
             const T *weights_time,
             const int32_t *bias,
             T *state,
             int8_t *output)
{
    if (p == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_buffers(input, weights_feature, weights_time));
    HCT_TRY(check_buffers(state, output));
    const int64_t n_batch = p->batch, n_input = p->input_size, n_filter = p->num_filters, n_memory = p->memory_size;
    if (n_batch < 1 || n_input < 1 || n_filter < 1 || n_memory < 1 || p->rank < 1 || n_filter % p->rank != 0 ||
        n_batch * n_filter * (n_memory + n_input) > kMaxElements)
    {
        return HCT_REF_E_DIMS;
    }
    if (p->input_zero_point < -128 || p->input_zero_point > 127 || p->output_zero_point < -128 ||
        p->output_zero_point > 127 || p->scale1_multiplier < 0 || p->scale2_multiplier < 0 || p->scale1_shift < -31 ||
        p->scale1_shift > 30 || p->scale2_shift < -31 || p->scale2_shift > 30)
    {
        return HCT_REF_E_PARAM;
    }
    const int64_t n_unit = n_filter / p->rank;

    // Left shift the activation state by one element; the newest column is rewritten below.
    std::copy(state + 1, state + n_batch * n_filter * n_memory, state);

    // Feature matmul into the newest state column.
    const int32_t state_max = std::numeric_limits<T>::max();
    const int32_t state_min = std::numeric_limits<T>::min();
    for (int64_t b = 0; b < n_batch; ++b)
    {
        T *result = state + b * n_filter * n_memory + (n_memory - 1);
        for (int64_t r = 0; r < n_filter; ++r)
        {
            int32_t dot = 0;
            const int8_t *w = weights_feature + r * n_input;
            const int8_t *x = input + b * n_input;
            for (int64_t c = 0; c < n_input; ++c)
            {
                dot += w[c] * (x[c] - p->input_zero_point);
            }
            dot = tflite::MultiplyByQuantizedMultiplier(dot, p->scale1_multiplier, p->scale1_shift);
            *result = static_cast<T>(std::min(std::max(state_min, dot), state_max));
            result += n_memory;
        }
    }

    // Time: per filter, dot(weights_time, state); then reduce over rank, add bias, rescale.
    std::vector<int32_t> scratch(static_cast<size_t>(n_batch * n_filter));
    for (int64_t b = 0; b < n_batch; ++b)
    {
        for (int64_t f = 0; f < n_filter; ++f)
        {
            int32_t acc = 0;
            const T *wt = weights_time + f * n_memory;
            const T *st = state + (b * n_filter + f) * n_memory;
            for (int64_t m = 0; m < n_memory; ++m)
            {
                acc += wt[m] * st[m];
            }
            scratch[static_cast<size_t>(b * n_filter + f)] = acc;
        }
    }
    for (int64_t b = 0; b < n_batch; ++b)
    {
        for (int64_t u = 0; u < n_unit; ++u)
        {
            int32_t acc = bias != nullptr ? bias[u] : 0;
            for (int64_t r = 0; r < p->rank; ++r)
            {
                acc += scratch[static_cast<size_t>(b * n_filter + u * p->rank + r)];
            }
            int32_t out = tflite::MultiplyByQuantizedMultiplier(acc, p->scale2_multiplier, p->scale2_shift) +
                          p->output_zero_point;
            output[b * n_unit + u] = static_cast<int8_t>(std::min(std::max(-128, out), 127));
        }
    }
    return HCT_REF_OK;
}

} // namespace

extern "C" {

int32_t hct_ref_svdf_s8(const HctSvdfParams *params,
                        const int8_t *input,
                        const int8_t *weights_feature,
                        const int8_t *weights_time,
                        const int32_t *bias,
                        int8_t *state,
                        int8_t *output)
{
    return svdf<int8_t>(params, input, weights_feature, weights_time, bias, state, output);
}

int32_t hct_ref_svdf_s8_state_s16(const HctSvdfParams *params,
                                  const int8_t *input,
                                  const int8_t *weights_feature,
                                  const int16_t *weights_time,
                                  const int32_t *bias,
                                  int16_t *state,
                                  int8_t *output)
{
    return svdf<int16_t>(params, input, weights_feature, weights_time, bias, state, output);
}

} // extern "C"
