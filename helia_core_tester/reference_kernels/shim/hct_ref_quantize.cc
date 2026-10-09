/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Quantize entries of the hct_ref shim (see hct_ref.h): TFLM's AffineQuantize.
 */
#include <cmath>
#include <cstdint>
#include <limits>

#include "hct_ref.h"
#include "hct_ref_internal.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace {

using namespace hct;

template <typename T> int32_t quantize(float scale, int32_t zero_point, const HctShape *shape, const float *input, T *output)
{
    HCT_TRY(check_shape(shape, 0));
    HCT_TRY(check_buffers(input, output));
    if (!std::isfinite(scale) || scale <= 0.0f || zero_point < std::numeric_limits<T>::min() ||
        zero_point > std::numeric_limits<T>::max())
    {
        return HCT_REF_E_PARAM;
    }
    const int64_t n = flat_size(shape);
    for (int64_t i = 0; i < n; ++i)
    {
        if (!std::isfinite(input[i]))
        {
            return HCT_REF_E_PARAM;
        }
    }
    tflite::QuantizationParams p;
    p.zero_point = zero_point;
    p.scale = scale;
    const RuntimeShape s = to_runtime(shape);
    tflite::reference_ops::AffineQuantize(p, s, input, s, output);
    return HCT_REF_OK;
}

} // namespace

extern "C" {

int32_t hct_ref_quantize_f32_s8(float scale, int32_t zero_point, const HctShape *shape, const float *input, int8_t *output)
{
    return quantize<int8_t>(scale, zero_point, shape, input, output);
}

int32_t hct_ref_quantize_f32_s16(float scale,
                                 int32_t zero_point,
                                 const HctShape *shape,
                                 const float *input,
                                 int16_t *output)
{
    return quantize<int16_t>(scale, zero_point, shape, input, output);
}

} // extern "C"
