/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Checks shared by the convolution-family entries.
 */
#ifndef HCT_CONV_COMMON_H
#define HCT_CONV_COMMON_H

#include "hct_ref_internal.h"

static inline int32_t hct_check_window(int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
                                       int32_t pad_h, int32_t pad_w)
{
    return (stride_h >= 1 && stride_w >= 1 && dilation_h >= 1 && dilation_w >= 1 && pad_h >= 0 && pad_w >= 0)
               ? HCT_OK
               : HCT_E_PARAM;
}

/* Per-output-channel requantization: count must match, multipliers non-negative,
 * shifts in [min_shift, max_shift]. */
static inline int32_t hct_check_per_channel(const HctTensor *multiplier, const HctTensor *shift, int64_t channels,
                                            int32_t min_shift, int32_t max_shift)
{
    int64_t n_mult = 0;
    int64_t n_shift = 0;
    HCT_TRY(hct_check_tensor(multiplier, HCT_INT32, &n_mult));
    HCT_TRY(hct_check_tensor(shift, HCT_INT32, &n_shift));
    if (n_mult != channels || n_shift != channels)
    {
        return HCT_E_SHAPE;
    }
    for (int64_t c = 0; c < channels; ++c)
    {
        const int32_t m = ((const int32_t *)multiplier->data)[c];
        const int32_t s = ((const int32_t *)shift->data)[c];
        if (m < 0 || s < min_shift || s > max_shift)
        {
            return HCT_E_PARAM;
        }
    }
    return HCT_OK;
}

/* A bias is either absent (zero elements) or one value per output channel. */
static inline int32_t hct_check_bias(const HctTensor *bias, int32_t dtype, int64_t channels, int *present)
{
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(bias, dtype, &n));
    if (n != 0 && n != channels)
    {
        return HCT_E_SHAPE;
    }
    *present = n != 0;
    return HCT_OK;
}

static inline int64_t hct_offset4(const HctTensor *t, int32_t a, int32_t b, int32_t c, int32_t d)
{
    return (((int64_t)a * t->dims[1] + b) * t->dims[2] + c) * t->dims[3] + d;
}

#endif /* HCT_CONV_COMMON_H */
