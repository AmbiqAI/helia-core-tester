/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Argument validation shared by the hct_ref shim sources.
 */
#ifndef HCT_REF_INTERNAL_H
#define HCT_REF_INTERNAL_H

#include <cmath>
#include <cstdint>
#include <limits>

#include "hct_ref.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"

namespace hct {

using tflite::RuntimeShape;

#define HCT_TRY(expr)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        const int32_t hct_status_ = (expr);                                                                            \
        if (hct_status_ != HCT_REF_OK)                                                                                 \
        {                                                                                                              \
            return hct_status_;                                                                                        \
        }                                                                                                              \
    } while (0)

// Largest element count any entry accepts; keeps index arithmetic in int.
constexpr int64_t kMaxElements = int64_t{1} << 28;

inline int32_t check_shape(const HctShape *shape, int32_t rank)
{
    if (shape == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (shape->rank < 1 || shape->rank > HCT_REF_MAX_RANK || (rank > 0 && shape->rank != rank))
    {
        return HCT_REF_E_DIMS;
    }
    int64_t count = 1;
    for (int32_t i = 0; i < shape->rank; ++i)
    {
        if (shape->dims[i] <= 0)
        {
            return HCT_REF_E_DIMS;
        }
        count *= shape->dims[i];
        if (count > kMaxElements)
        {
            return HCT_REF_E_DIMS;
        }
    }
    return HCT_REF_OK;
}

inline int64_t flat_size(const HctShape *shape)
{
    int64_t count = 1;
    for (int32_t i = 0; i < shape->rank; ++i)
    {
        count *= shape->dims[i];
    }
    return count;
}

inline RuntimeShape to_runtime(const HctShape *shape)
{
    return RuntimeShape(shape->rank, shape->dims);
}

template <typename T> int32_t check_activation(const HctActivation &act)
{
    if (act.min > act.max || act.min < std::numeric_limits<T>::min() || act.max > std::numeric_limits<T>::max())
    {
        return HCT_REF_E_PARAM;
    }
    return HCT_REF_OK;
}

template <> inline int32_t check_activation<float>(const HctActivation &act)
{
    if (std::isnan(act.fmin) || std::isnan(act.fmax) || act.fmin > act.fmax)
    {
        return HCT_REF_E_PARAM;
    }
    return HCT_REF_OK;
}

// Zero points of 8-bit activations lie in [-128, 127]; 16-bit activations are
// symmetric (zero point 0).
template <typename T> int32_t check_offsets(int32_t input_offset, int32_t output_offset)
{
    if (sizeof(T) == 1)
    {
        if (input_offset < -127 || input_offset > 128 || output_offset < -128 || output_offset > 127)
        {
            return HCT_REF_E_PARAM;
        }
    }
    else if (input_offset != 0 || output_offset != 0)
    {
        return HCT_REF_E_PARAM;
    }
    return HCT_REF_OK;
}

template <> inline int32_t check_offsets<float>(int32_t input_offset, int32_t output_offset)
{
    return (input_offset == 0 && output_offset == 0) ? HCT_REF_OK : HCT_REF_E_PARAM;
}

inline int32_t check_buffers(const void *a, const void *b, const void *c)
{
    return (a != nullptr && b != nullptr && c != nullptr) ? HCT_REF_OK : HCT_REF_E_NULL;
}

inline int32_t check_buffers(const void *a, const void *b)
{
    return (a != nullptr && b != nullptr) ? HCT_REF_OK : HCT_REF_E_NULL;
}


} // namespace hct

#endif /* HCT_REF_INTERNAL_H */
