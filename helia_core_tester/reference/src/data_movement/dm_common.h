/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Helpers shared by the dtype-generic data-movement entries: element copies by size,
 * row-major strides, and the checks every entry repeats.
 */
#ifndef HCT_DM_COMMON_H
#define HCT_DM_COMMON_H

#include <string.h>

#include "hct_ref_internal.h"

static inline void hct_copy_elem(HctTensor *dst, int64_t di, const HctTensor *src, int64_t si)
{
    memcpy(hct_elem(dst, di), hct_elem(src, si), (size_t)hct_dtype_size(src->dtype));
}

/* Row-major element strides of t. */
static inline void hct_strides(const HctTensor *t, int64_t strides[HCT_MAX_RANK])
{
    int64_t s = 1;
    for (int32_t d = t->rank - 1; d >= 0; --d)
    {
        strides[d] = s;
        s *= t->dims[d];
    }
}

/* Splits a row-major linear index into coordinates of t. */
static inline void hct_unravel(const HctTensor *t, int64_t index, int32_t coord[HCT_MAX_RANK])
{
    for (int32_t d = t->rank - 1; d >= 0; --d)
    {
        const int32_t n = t->dims[d] > 0 ? t->dims[d] : 1;
        coord[d] = (int32_t)(index % n);
        index /= n;
    }
}

/* Checks a data tensor declared `any` and an output that must share its dtype. */
static inline int32_t hct_check_any_pair(const HctTensor *in, int64_t *n_in, HctTensor *out, int64_t *n_out)
{
    HCT_TRY(hct_check_tensor_any(in, n_in));
    HCT_TRY(hct_check_tensor_any(out, n_out));
    return in->dtype == out->dtype ? HCT_OK : HCT_E_DTYPE;
}

/* An int32 vector of exactly `n` elements; its data in *v. */
static inline int32_t hct_check_i32_vector(const HctTensor *t, int64_t n, const int32_t **v)
{
    int64_t count = 0;
    HCT_TRY(hct_check_tensor(t, HCT_INT32, &count));
    if (count != n || (t->rank != 1 && !(t->rank == 0 && n == 1)))
    {
        return HCT_E_SHAPE;
    }
    *v = (const int32_t *)t->data;
    return HCT_OK;
}

/* A one-element value tensor of dtype `dtype`. */
static inline int32_t hct_check_scalar_value(const HctTensor *t, int32_t dtype)
{
    int64_t count = 0;
    HCT_TRY(hct_check_tensor_any(t, &count));
    if (t->dtype != dtype)
    {
        return HCT_E_DTYPE;
    }
    return count == 1 ? HCT_OK : HCT_E_SHAPE;
}

#endif /* HCT_DM_COMMON_H */
