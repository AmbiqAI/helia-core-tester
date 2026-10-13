/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Entries whose output element is one input element chosen by an index map: Transpose,
 * StridedSlice (TFLite's strided_slice_logic: masks, negative indices, clamping, shrink),
 * Tile, BroadcastTo (numpy rules), ReverseSequence and DynamicUpdateSlice (TFLite clamps the
 * start so the update fits). Any element type.
 */
#include "dm_common.h"

int32_t hct_ref_transpose(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                          int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    HCT_TRY(hct_check_any_pair(in, &n_in, out, &n_out));
    const int32_t *perm = NULL;
    HCT_TRY(hct_check_i32_vector(&inputs[1], in->rank, &perm));
    if (out->rank != in->rank)
    {
        return HCT_E_SHAPE;
    }
    int32_t seen = 0;
    for (int32_t d = 0; d < in->rank; ++d)
    {
        if (perm[d] < 0 || perm[d] >= in->rank || (seen & (1 << perm[d])))
        {
            return HCT_E_PARAM;
        }
        seen |= 1 << perm[d];
        if (out->dims[d] != in->dims[perm[d]])
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
        for (int32_t d = 0; d < out->rank; ++d)
        {
            src += coord[d] * in_strides[perm[d]];
        }
        hct_copy_elem(out, i, in, src);
    }
    return HCT_OK;
}

static int64_t clamp64(int64_t v, int64_t lo, int64_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

int32_t hct_ref_strided_slice(const HctStridedSliceParams *params, const HctTensor *inputs, int32_t num_inputs,
                              HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 4, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    HCT_TRY(hct_check_any_pair(in, &n_in, out, &n_out));
    const int32_t rank = in->rank;
    const int32_t *begin = NULL;
    const int32_t *end = NULL;
    const int32_t *strides = NULL;
    HCT_TRY(hct_check_i32_vector(&inputs[1], rank, &begin));
    HCT_TRY(hct_check_i32_vector(&inputs[2], rank, &end));
    HCT_TRY(hct_check_i32_vector(&inputs[3], rank, &strides));
    const int32_t valid = rank >= 31 ? -1 : (int32_t)((1u << rank) - 1u);
    if (params->ellipsis_mask != 0 || params->new_axis_mask != 0 || (params->begin_mask & ~valid) ||
        (params->end_mask & ~valid) || (params->shrink_axis_mask & ~valid))
    {
        return HCT_E_PARAM;
    }
    int64_t start[HCT_MAX_RANK];
    int64_t count[HCT_MAX_RANK];
    int32_t out_d = 0;
    for (int32_t d = 0; d < rank; ++d)
    {
        const int64_t size = in->dims[d];
        const int64_t s = strides[d];
        if (s == 0)
        {
            return HCT_E_PARAM;
        }
        const int shrink = (params->shrink_axis_mask >> d) & 1;
        int64_t b = begin[d];
        if ((params->begin_mask >> d) & 1)
        {
            b = s > 0 ? INT32_MIN : INT32_MAX;
        }
        if (b < 0)
        {
            b += size;
        }
        b = s > 0 ? clamp64(b, 0, size) : clamp64(b, -1, size - 1);
        int64_t e = 0;
        if (shrink)
        {
            e = b + 1;
            if (size == 0 || b < 0 || b >= size)
            {
                return HCT_E_PARAM; /* a shrunk axis must name an element */
            }
        }
        else
        {
            e = end[d];
            if ((params->end_mask >> d) & 1)
            {
                e = s > 0 ? INT32_MAX : INT32_MIN;
            }
            if (e < 0)
            {
                e += size;
            }
            e = s > 0 ? clamp64(e, 0, size) : clamp64(e, -1, size - 1);
        }
        const int64_t span = s > 0 ? e - b : b - e;
        const int64_t step = s > 0 ? s : -s;
        start[d] = b;
        count[d] = span <= 0 ? 0 : (span + step - 1) / step;
        if (shrink)
        {
            count[d] = 1;
            continue;
        }
        if (out_d >= out->rank || out->dims[out_d] != count[d])
        {
            return HCT_E_SHAPE;
        }
        ++out_d;
    }
    if (out_d != out->rank)
    {
        return HCT_E_SHAPE;
    }
    int64_t in_strides[HCT_MAX_RANK];
    hct_strides(in, in_strides);
    for (int64_t i = 0; i < n_out; ++i)
    {
        int64_t rem = i;
        int64_t src = 0;
        for (int32_t d = rank - 1; d >= 0; --d)
        {
            const int64_t c = rem % count[d];
            rem /= count[d];
            src += (start[d] + c * strides[d]) * in_strides[d];
        }
        hct_copy_elem(out, i, in, src);
    }
    return HCT_OK;
}

int32_t hct_ref_tile(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                     int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    HCT_TRY(hct_check_any_pair(in, &n_in, out, &n_out));
    const int32_t *mult = NULL;
    HCT_TRY(hct_check_i32_vector(&inputs[1], in->rank, &mult));
    if (out->rank != in->rank)
    {
        return HCT_E_SHAPE;
    }
    for (int32_t d = 0; d < in->rank; ++d)
    {
        if (mult[d] < 0)
        {
            return HCT_E_PARAM;
        }
        if (out->dims[d] != (int64_t)in->dims[d] * mult[d])
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
        for (int32_t d = 0; d < in->rank; ++d)
        {
            src += (coord[d] % in->dims[d]) * in_strides[d];
        }
        hct_copy_elem(out, i, in, src);
    }
    return HCT_OK;
}

int32_t hct_ref_broadcast_to(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    HCT_TRY(hct_check_any_pair(in, &n_in, out, &n_out));
    if (out->rank < in->rank)
    {
        return HCT_E_SHAPE;
    }
    const int32_t lead = out->rank - in->rank;
    for (int32_t d = 0; d < in->rank; ++d)
    {
        if (in->dims[d] != 1 && in->dims[d] != out->dims[lead + d])
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
        for (int32_t d = 0; d < in->rank; ++d)
        {
            src += (in->dims[d] == 1 ? 0 : coord[lead + d]) * in_strides[d];
        }
        hct_copy_elem(out, i, in, src);
    }
    return HCT_OK;
}

int32_t hct_ref_reverse_sequence(const HctReverseSequenceParams *params, const HctTensor *inputs, int32_t num_inputs,
                                 HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    HCT_TRY(hct_check_any_pair(in, &n_in, out, &n_out));
    HCT_TRY(hct_check_same_shape(in, out));
    const int32_t sd = params->seq_dim;
    const int32_t bd = params->batch_dim;
    if (sd < 0 || sd >= in->rank || bd < 0 || bd >= in->rank || sd == bd)
    {
        return HCT_E_PARAM;
    }
    const int32_t *len = NULL;
    HCT_TRY(hct_check_i32_vector(&inputs[1], in->dims[bd], &len));
    for (int32_t b = 0; b < in->dims[bd]; ++b)
    {
        if (len[b] < 0 || len[b] > in->dims[sd])
        {
            return HCT_E_PARAM;
        }
    }
    int64_t in_strides[HCT_MAX_RANK];
    hct_strides(in, in_strides);
    int32_t coord[HCT_MAX_RANK];
    for (int64_t i = 0; i < n_out; ++i)
    {
        hct_unravel(out, i, coord);
        const int32_t l = len[coord[bd]];
        if (coord[sd] < l)
        {
            coord[sd] = l - 1 - coord[sd];
        }
        int64_t src = 0;
        for (int32_t d = 0; d < in->rank; ++d)
        {
            src += coord[d] * in_strides[d];
        }
        hct_copy_elem(out, i, in, src);
    }
    return HCT_OK;
}

int32_t hct_ref_dynamic_update_slice(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                                     HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    const HctTensor *op = &inputs[0];
    const HctTensor *upd = &inputs[1];
    HctTensor *out = &outputs[0];
    int64_t n_op = 0;
    int64_t n_upd = 0;
    int64_t n_out = 0;
    HCT_TRY(hct_check_any_pair(op, &n_op, out, &n_out));
    HCT_TRY(hct_check_tensor_any(upd, &n_upd));
    if (upd->dtype != op->dtype)
    {
        return HCT_E_DTYPE;
    }
    HCT_TRY(hct_check_same_shape(op, out));
    if (upd->rank != op->rank)
    {
        return HCT_E_SHAPE;
    }
    const int32_t *start_in = NULL;
    HCT_TRY(hct_check_i32_vector(&inputs[2], op->rank, &start_in));
    int64_t start[HCT_MAX_RANK];
    for (int32_t d = 0; d < op->rank; ++d)
    {
        if (upd->dims[d] > op->dims[d])
        {
            return HCT_E_SHAPE;
        }
        start[d] = clamp64(start_in[d], 0, (int64_t)op->dims[d] - upd->dims[d]);
    }
    for (int64_t i = 0; i < n_op; ++i)
    {
        hct_copy_elem(out, i, op, i);
    }
    int64_t out_strides[HCT_MAX_RANK];
    hct_strides(out, out_strides);
    int32_t coord[HCT_MAX_RANK];
    for (int64_t i = 0; i < n_upd; ++i)
    {
        hct_unravel(upd, i, coord);
        int64_t dst = 0;
        for (int32_t d = 0; d < op->rank; ++d)
        {
            dst += (start[d] + coord[d]) * out_strides[d];
        }
        hct_copy_elem(out, dst, upd, i);
    }
    return HCT_OK;
}
