/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Concatenation / Split, Gather / GatherND and Pad / MirrorPad (TFLite reference semantics),
 * for any element type. A variable number of operands travels as one buffer: Concatenation
 * takes its inputs back to back with each one's extent along the axis, Split returns its
 * slices back to back. Pack and Unpack are these with unit extents and a reshape.
 */
#include "dm_common.h"

static int32_t normalize_axis(int32_t axis, int32_t rank, int32_t *out)
{
    if (rank < 1 || axis < -rank || axis >= rank)
    {
        return HCT_E_PARAM;
    }
    *out = axis < 0 ? axis + rank : axis;
    return HCT_OK;
}

/* outer / inner element counts around `axis` of t. */
static void around_axis(const HctTensor *t, int32_t axis, int64_t *outer, int64_t *inner)
{
    *outer = 1;
    *inner = 1;
    for (int32_t d = 0; d < t->rank; ++d)
    {
        if (d < axis)
        {
            *outer *= t->dims[d];
        }
        else if (d > axis)
        {
            *inner *= t->dims[d];
        }
    }
}

/* The packed side (Concatenation's data, Split's output) against the shaped side. */
static int32_t concat_split(const HctAxisParams *p, const HctTensor *packed, const HctTensor *extents_t,
                            const HctTensor *shaped, int is_concat, HctTensor *dst)
{
    int64_t n_packed = 0;
    int64_t n_shaped = 0;
    HCT_TRY(hct_check_tensor_any(packed, &n_packed));
    HCT_TRY(hct_check_tensor_any(shaped, &n_shaped));
    if (packed->dtype != shaped->dtype)
    {
        return HCT_E_DTYPE;
    }
    int32_t axis = 0;
    HCT_TRY(normalize_axis(p->axis, shaped->rank, &axis));
    int64_t n_ext = 0;
    HCT_TRY(hct_check_tensor(extents_t, HCT_INT32, &n_ext));
    if (extents_t->rank != 1 || n_ext < 1)
    {
        return HCT_E_SHAPE;
    }
    const int32_t *ext = (const int32_t *)extents_t->data;
    int64_t total = 0;
    for (int64_t k = 0; k < n_ext; ++k)
    {
        if (ext[k] < 0)
        {
            return HCT_E_PARAM;
        }
        total += ext[k];
    }
    if (total != shaped->dims[axis] || n_packed != n_shaped)
    {
        return HCT_E_SHAPE;
    }
    int64_t outer = 0;
    int64_t inner = 0;
    around_axis(shaped, axis, &outer, &inner);
    int64_t base = 0;
    int64_t offset = 0;
    for (int64_t k = 0; k < n_ext; ++k)
    {
        for (int64_t o = 0; o < outer; ++o)
        {
            for (int64_t j = 0; j < ext[k]; ++j)
            {
                for (int64_t i = 0; i < inner; ++i)
                {
                    const int64_t si = (o * total + offset + j) * inner + i;
                    const int64_t pi = base + (o * ext[k] + j) * inner + i;
                    if (is_concat)
                    {
                        hct_copy_elem(dst, si, packed, pi);
                    }
                    else
                    {
                        hct_copy_elem(dst, pi, shaped, si);
                    }
                }
            }
        }
        base += outer * ext[k] * inner;
        offset += ext[k];
    }
    return HCT_OK;
}

int32_t hct_ref_concatenation(const HctAxisParams *params, const HctTensor *inputs, int32_t num_inputs,
                              HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    return concat_split(params, &inputs[0], &inputs[1], &outputs[0], 1, &outputs[0]);
}

int32_t hct_ref_split(const HctAxisParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                      int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    return concat_split(params, &outputs[0], &inputs[1], &inputs[0], 0, &outputs[0]);
}

int32_t hct_ref_gather(const HctAxisParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                       int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    const HctTensor *idx = &inputs[1];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t n_idx = 0;
    HCT_TRY(hct_check_any_pair(in, &n_in, out, &n_out));
    HCT_TRY(hct_check_tensor(idx, HCT_INT32, &n_idx));
    int32_t axis = 0;
    HCT_TRY(normalize_axis(params->axis, in->rank, &axis));
    /* Output: in[:axis] + indices.shape + in[axis+1:]. */
    if (out->rank != in->rank - 1 + idx->rank)
    {
        return HCT_E_SHAPE;
    }
    int32_t d = 0;
    for (int32_t i = 0; i < axis; ++i)
    {
        if (out->dims[d++] != in->dims[i])
        {
            return HCT_E_SHAPE;
        }
    }
    for (int32_t i = 0; i < idx->rank; ++i)
    {
        if (out->dims[d++] != idx->dims[i])
        {
            return HCT_E_SHAPE;
        }
    }
    for (int32_t i = axis + 1; i < in->rank; ++i)
    {
        if (out->dims[d++] != in->dims[i])
        {
            return HCT_E_SHAPE;
        }
    }
    const int32_t *ix = (const int32_t *)idx->data;
    for (int64_t k = 0; k < n_idx; ++k)
    {
        if (ix[k] < 0 || ix[k] >= in->dims[axis])
        {
            return HCT_E_PARAM; /* TFLite's Gather rejects an index outside the axis */
        }
    }
    int64_t outer = 0;
    int64_t inner = 0;
    around_axis(in, axis, &outer, &inner);
    for (int64_t o = 0; o < outer; ++o)
    {
        for (int64_t k = 0; k < n_idx; ++k)
        {
            for (int64_t i = 0; i < inner; ++i)
            {
                hct_copy_elem(out, (o * n_idx + k) * inner + i, in, (o * in->dims[axis] + ix[k]) * inner + i);
            }
        }
    }
    return HCT_OK;
}

int32_t hct_ref_gather_nd(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                          HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    const HctTensor *idx = &inputs[1];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t n_idx = 0;
    HCT_TRY(hct_check_any_pair(in, &n_in, out, &n_out));
    HCT_TRY(hct_check_tensor(idx, HCT_INT32, &n_idx));
    if (idx->rank < 1 || in->rank < 1)
    {
        return HCT_E_SHAPE;
    }
    const int32_t depth = idx->dims[idx->rank - 1];
    if (depth < 1 || depth > in->rank)
    {
        return HCT_E_SHAPE;
    }
    /* Output: indices.shape[:-1] + in.shape[depth:]. */
    if (out->rank != idx->rank - 1 + in->rank - depth)
    {
        return HCT_E_SHAPE;
    }
    int32_t d = 0;
    for (int32_t i = 0; i < idx->rank - 1; ++i)
    {
        if (out->dims[d++] != idx->dims[i])
        {
            return HCT_E_SHAPE;
        }
    }
    int64_t slice = 1;
    for (int32_t i = depth; i < in->rank; ++i)
    {
        if (out->dims[d++] != in->dims[i])
        {
            return HCT_E_SHAPE;
        }
        slice *= in->dims[i];
    }
    int64_t in_strides[HCT_MAX_RANK];
    hct_strides(in, in_strides);
    const int32_t *ix = (const int32_t *)idx->data;
    const int64_t n_lookups = n_idx / depth;
    for (int64_t k = 0; k < n_lookups; ++k)
    {
        int64_t base = 0;
        for (int32_t j = 0; j < depth; ++j)
        {
            const int32_t v = ix[k * depth + j];
            if (v < 0 || v >= in->dims[j])
            {
                return HCT_E_PARAM;
            }
            base += (int64_t)v * in_strides[j];
        }
        for (int64_t i = 0; i < slice; ++i)
        {
            hct_copy_elem(out, k * slice + i, in, base + i);
        }
    }
    return HCT_OK;
}

/* Source coordinate of output coordinate o along a padded dim, or -1 for the pad value. */
static int64_t padded_source(int64_t o, int64_t before, int64_t dim, int32_t mode)
{
    int64_t c = o - before;
    if (c >= 0 && c < dim)
    {
        return c;
    }
    if (mode == 0)
    {
        return -1;
    }
    if (c < 0)
    {
        return mode == 1 ? -c : -c - 1;
    }
    return mode == 1 ? 2 * dim - 2 - c : 2 * dim - 1 - c;
}

int32_t hct_ref_pad(const HctPadParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                    int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t n_pad = 0;
    HCT_TRY(hct_check_any_pair(in, &n_in, out, &n_out));
    HCT_TRY(hct_check_tensor(&inputs[1], HCT_INT32, &n_pad));
    HCT_TRY(hct_check_scalar_value(&inputs[2], in->dtype));
    if (params->mode < 0 || params->mode > 2)
    {
        return HCT_E_PARAM;
    }
    if (inputs[1].rank != 2 || inputs[1].dims[0] != in->rank || inputs[1].dims[1] != 2 || out->rank != in->rank)
    {
        return HCT_E_SHAPE;
    }
    const int32_t *pad = (const int32_t *)inputs[1].data;
    for (int32_t d = 0; d < in->rank; ++d)
    {
        const int32_t b = pad[2 * d];
        const int32_t a = pad[2 * d + 1];
        /* REFLECT may reach dim - 1 past an edge, SYMMETRIC dim, CONSTANT any distance. */
        const int32_t reach = params->mode == 1 ? in->dims[d] - 1 : in->dims[d];
        if (b < 0 || a < 0 || (params->mode != 0 && (b > reach || a > reach)))
        {
            return HCT_E_PARAM;
        }
        if (out->dims[d] != (int64_t)in->dims[d] + b + a)
        {
            return HCT_E_SHAPE;
        }
    }
    int64_t in_strides[HCT_MAX_RANK];
    hct_strides(in, in_strides);
    int32_t coord[HCT_MAX_RANK];
    for (int64_t i = 0; i < n_out; ++i)
    {
        hct_unravel(out, i, coord);
        int64_t src = 0;
        for (int32_t d = 0; d < in->rank && src >= 0; ++d)
        {
            const int64_t c = padded_source(coord[d], pad[2 * d], in->dims[d], params->mode);
            src = c < 0 ? -1 : src + c * in_strides[d];
        }
        if (src < 0)
        {
            hct_copy_elem(out, i, &inputs[2], 0);
        }
        else
        {
            hct_copy_elem(out, i, in, src);
        }
    }
    return HCT_OK;
}
