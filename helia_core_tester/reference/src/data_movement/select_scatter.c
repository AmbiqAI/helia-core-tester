/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * ScatterND (zero-filled output, duplicate indices summed in the element type as TFLite's
 * ScatterNd does: integers wrap, binary16 rounds each step), SelectV2 (numpy broadcasting of
 * condition, x and y), Where (row-major coordinates of the non-zero elements; -0.0 is zero,
 * NaN is not) and Fill.
 */
#include <string.h>

#include "dm_common.h"

static void accumulate(HctTensor *out, int64_t di, const HctTensor *upd, int64_t si)
{
    switch (out->dtype)
    {
    case HCT_INT8:
        ((int8_t *)out->data)[di] =
            (int8_t)(uint8_t)((uint32_t)((int8_t *)out->data)[di] + (uint32_t)((const int8_t *)upd->data)[si]);
        break;
    case HCT_INT16:
        ((int16_t *)out->data)[di] =
            (int16_t)(uint16_t)((uint32_t)((int16_t *)out->data)[di] + (uint32_t)((const int16_t *)upd->data)[si]);
        break;
    case HCT_INT32:
        ((int32_t *)out->data)[di] =
            (int32_t)((uint32_t)((int32_t *)out->data)[di] + (uint32_t)((const int32_t *)upd->data)[si]);
        break;
    case HCT_INT64:
        ((int64_t *)out->data)[di] =
            (int64_t)((uint64_t)((int64_t *)out->data)[di] + (uint64_t)((const int64_t *)upd->data)[si]);
        break;
    case HCT_FLOAT32:
        ((float *)out->data)[di] += ((const float *)upd->data)[si];
        break;
    default: /* HCT_FLOAT16: one binary16 addition */
        hct_store_f32(out, di, hct_load_f32(out, di) + hct_load_f32(upd, si));
        break;
    }
}

int32_t hct_ref_scatter_nd(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                           HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    const HctTensor *idx = &inputs[0];
    const HctTensor *upd = &inputs[1];
    HctTensor *out = &outputs[0];
    int64_t n_idx = 0;
    int64_t n_upd = 0;
    int64_t n_out = 0;
    HCT_TRY(hct_check_tensor(idx, HCT_INT32, &n_idx));
    HCT_TRY(hct_check_any_pair(upd, &n_upd, out, &n_out));
    if (out->dtype == HCT_BOOL)
    {
        return HCT_E_DTYPE;
    }
    if (idx->rank < 1)
    {
        return HCT_E_SHAPE;
    }
    const int32_t depth = idx->dims[idx->rank - 1];
    if (depth < 1 || depth > out->rank || upd->rank != idx->rank - 1 + out->rank - depth)
    {
        return HCT_E_SHAPE;
    }
    int32_t d = 0;
    for (int32_t i = 0; i < idx->rank - 1; ++i)
    {
        if (upd->dims[d++] != idx->dims[i])
        {
            return HCT_E_SHAPE;
        }
    }
    int64_t slice = 1;
    for (int32_t i = depth; i < out->rank; ++i)
    {
        if (upd->dims[d++] != out->dims[i])
        {
            return HCT_E_SHAPE;
        }
        slice *= out->dims[i];
    }
    int64_t out_strides[HCT_MAX_RANK];
    hct_strides(out, out_strides);
    const int32_t *ix = (const int32_t *)idx->data;
    const int64_t n_lookups = n_idx / depth;
    for (int64_t k = 0; k < n_lookups; ++k)
    {
        for (int32_t j = 0; j < depth; ++j)
        {
            if (ix[k * depth + j] < 0 || ix[k * depth + j] >= out->dims[j])
            {
                return HCT_E_PARAM;
            }
        }
    }
    if (n_out > 0)
    {
        memset(out->data, 0, (size_t)(n_out * hct_dtype_size(out->dtype)));
    }
    for (int64_t k = 0; k < n_lookups; ++k)
    {
        int64_t base = 0;
        for (int32_t j = 0; j < depth; ++j)
        {
            base += (int64_t)ix[k * depth + j] * out_strides[j];
        }
        for (int64_t i = 0; i < slice; ++i)
        {
            accumulate(out, base + i, upd, k * slice + i);
        }
    }
    return HCT_OK;
}

/* Offset of t's element that numpy broadcasting maps to output coordinate `coord`. */
static int64_t broadcast_offset(const HctTensor *t, const int64_t strides[HCT_MAX_RANK], const HctTensor *out,
                                const int32_t coord[HCT_MAX_RANK])
{
    const int32_t lead = out->rank - t->rank;
    int64_t off = 0;
    for (int32_t d = 0; d < t->rank; ++d)
    {
        off += (t->dims[d] == 1 ? 0 : coord[lead + d]) * strides[d];
    }
    return off;
}

static int32_t check_broadcastable(const HctTensor *t, const HctTensor *out)
{
    if (t->rank > out->rank)
    {
        return HCT_E_SHAPE;
    }
    const int32_t lead = out->rank - t->rank;
    for (int32_t d = 0; d < t->rank; ++d)
    {
        if (t->dims[d] != 1 && t->dims[d] != out->dims[lead + d])
        {
            return HCT_E_SHAPE;
        }
    }
    return HCT_OK;
}

int32_t hct_ref_select_v2(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                          int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    const HctTensor *cond = &inputs[0];
    const HctTensor *x = &inputs[1];
    const HctTensor *y = &inputs[2];
    HctTensor *out = &outputs[0];
    int64_t n_c = 0;
    int64_t n_x = 0;
    int64_t n_y = 0;
    int64_t n_out = 0;
    HCT_TRY(hct_check_tensor(cond, HCT_BOOL, &n_c));
    HCT_TRY(hct_check_any_pair(x, &n_x, out, &n_out));
    HCT_TRY(hct_check_tensor_any(y, &n_y));
    if (y->dtype != x->dtype)
    {
        return HCT_E_DTYPE;
    }
    /* The output shape is the broadcast of all three; each must broadcast to it, and every
     * output dim must come from one of them. */
    HCT_TRY(check_broadcastable(cond, out));
    HCT_TRY(check_broadcastable(x, out));
    HCT_TRY(check_broadcastable(y, out));
    for (int32_t d = 0; d < out->rank; ++d)
    {
        const HctTensor *ts[3] = {cond, x, y};
        int found = out->dims[d] == 1;
        for (int k = 0; k < 3 && !found; ++k)
        {
            const int32_t td = d - (out->rank - ts[k]->rank);
            found = td >= 0 && ts[k]->dims[td] == out->dims[d];
        }
        if (!found)
        {
            return HCT_E_SHAPE;
        }
    }
    int64_t sc[HCT_MAX_RANK];
    int64_t sx[HCT_MAX_RANK];
    int64_t sy[HCT_MAX_RANK];
    hct_strides(cond, sc);
    hct_strides(x, sx);
    hct_strides(y, sy);
    int32_t coord[HCT_MAX_RANK];
    for (int64_t i = 0; i < n_out; ++i)
    {
        hct_unravel(out, i, coord);
        const uint8_t c = ((const uint8_t *)cond->data)[broadcast_offset(cond, sc, out, coord)];
        if (c)
        {
            hct_copy_elem(out, i, x, broadcast_offset(x, sx, out, coord));
        }
        else
        {
            hct_copy_elem(out, i, y, broadcast_offset(y, sy, out, coord));
        }
    }
    return HCT_OK;
}

static int nonzero(const HctTensor *t, int64_t i)
{
    switch (t->dtype)
    {
    case HCT_INT8:
        return ((const int8_t *)t->data)[i] != 0;
    case HCT_BOOL:
        return ((const uint8_t *)t->data)[i] != 0;
    case HCT_INT16:
        return ((const int16_t *)t->data)[i] != 0;
    case HCT_INT32:
        return ((const int32_t *)t->data)[i] != 0;
    case HCT_INT64:
        return ((const int64_t *)t->data)[i] != 0;
    case HCT_FLOAT32:
    {
        const float v = ((const float *)t->data)[i];
        return v != 0.0f; /* NaN != 0 */
    }
    default: /* HCT_FLOAT16: any bit other than the sign */
        return (((const uint16_t *)t->data)[i] & 0x7FFFu) != 0;
    }
}

int32_t hct_ref_where(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                      int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 2));
    const HctTensor *cond = &inputs[0];
    int64_t n_c = 0;
    int64_t n_coords = 0;
    int64_t n_count = 0;
    HCT_TRY(hct_check_tensor_any(cond, &n_c));
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_INT64, &n_coords));
    HCT_TRY(hct_check_tensor(&outputs[1], HCT_INT32, &n_count));
    /* coords is [capacity, rank]; count holds how many rows are filled. */
    if (outputs[0].rank != 2 || outputs[0].dims[1] != cond->rank || n_count != 1)
    {
        return HCT_E_SHAPE;
    }
    int64_t found = 0;
    for (int64_t i = 0; i < n_c; ++i)
    {
        found += nonzero(cond, i);
    }
    if (found > outputs[0].dims[0])
    {
        return HCT_E_SHAPE;
    }
    if (n_coords > 0)
    {
        memset(outputs[0].data, 0, (size_t)n_coords * sizeof(int64_t));
    }
    int64_t row = 0;
    int32_t coord[HCT_MAX_RANK];
    for (int64_t i = 0; i < n_c; ++i)
    {
        if (!nonzero(cond, i))
        {
            continue;
        }
        hct_unravel(cond, i, coord);
        for (int32_t d = 0; d < cond->rank; ++d)
        {
            ((int64_t *)outputs[0].data)[row * cond->rank + d] = coord[d];
        }
        ++row;
    }
    ((int32_t *)outputs[1].data)[0] = (int32_t)found;
    return HCT_OK;
}

int32_t hct_ref_fill(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                     int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    int64_t n_out = 0;
    HCT_TRY(hct_check_tensor_any(&outputs[0], &n_out));
    HCT_TRY(hct_check_scalar_value(&inputs[0], outputs[0].dtype));
    for (int64_t i = 0; i < n_out; ++i)
    {
        hct_copy_elem(&outputs[0], i, &inputs[0], 0);
    }
    return HCT_OK;
}
