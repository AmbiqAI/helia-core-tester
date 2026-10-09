/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 */
#include "hct_ref_internal.h"

int32_t hct_ref_abi_version(void)
{
    return HCT_REF_ABI_VERSION;
}

int32_t hct_check_tensor(const HctTensor *t, int32_t dtype, int64_t *count)
{
    if (t == NULL || count == NULL)
    {
        return HCT_E_NULL;
    }
    if (t->dtype != dtype)
    {
        return HCT_E_DTYPE;
    }
    if (t->rank < 0 || t->rank > HCT_MAX_RANK)
    {
        return HCT_E_SHAPE;
    }
    int64_t n = 1;
    for (int32_t i = 0; i < t->rank; ++i)
    {
        if (t->dims[i] < 0)
        {
            return HCT_E_SHAPE;
        }
        n *= t->dims[i];
        if (n > HCT_MAX_ELEMENTS)
        {
            return HCT_E_SIZE;
        }
    }
    if (n > 0 && t->data == NULL)
    {
        return HCT_E_NULL;
    }
    *count = n;
    return HCT_OK;
}

int32_t hct_check_io(
    const HctTensor *inputs, int32_t num_inputs, int32_t want_inputs, const HctTensor *outputs, int32_t num_outputs,
    int32_t want_outputs)
{
    if (inputs == NULL || outputs == NULL)
    {
        return HCT_E_NULL;
    }
    if (num_inputs != want_inputs || num_outputs != want_outputs)
    {
        return HCT_E_COUNT;
    }
    return HCT_OK;
}

int32_t hct_broadcast2_init(HctBroadcast2 *bc, const HctTensor *a, const HctTensor *b, const HctTensor *out)
{
    if (bc == NULL || a == NULL || b == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    const int32_t rank = a->rank > b->rank ? a->rank : b->rank;
    if (out->rank != rank || rank > HCT_MAX_RANK)
    {
        return HCT_E_SHAPE;
    }
    bc->rank = rank;
    int64_t a_stride = 1;
    int64_t b_stride = 1;
    bc->count = 1;
    /* Walk from the innermost dim; a missing leading dim of an operand is 1. */
    for (int32_t i = rank - 1; i >= 0; --i)
    {
        const int32_t ai = i - (rank - a->rank);
        const int32_t bi = i - (rank - b->rank);
        const int32_t da = ai >= 0 ? a->dims[ai] : 1;
        const int32_t db = bi >= 0 ? b->dims[bi] : 1;
        if (da != db && da != 1 && db != 1)
        {
            return HCT_E_SHAPE;
        }
        const int32_t d = (da == 1) ? db : da;
        if (out->dims[i] != d)
        {
            return HCT_E_SHAPE;
        }
        bc->dims[i] = d;
        bc->a_strides[i] = (da == 1) ? 0 : a_stride;
        bc->b_strides[i] = (db == 1) ? 0 : b_stride;
        a_stride *= da;
        b_stride *= db;
        bc->count *= d;
        if (bc->count > HCT_MAX_ELEMENTS)
        {
            return HCT_E_SIZE;
        }
    }
    return HCT_OK;
}

void hct_broadcast2_offsets(const HctBroadcast2 *bc, int64_t index, int64_t *a_offset, int64_t *b_offset)
{
    int64_t ao = 0;
    int64_t bo = 0;
    for (int32_t i = bc->rank - 1; i >= 0; --i)
    {
        const int64_t coord = index % bc->dims[i];
        index /= bc->dims[i];
        ao += coord * bc->a_strides[i];
        bo += coord * bc->b_strides[i];
    }
    *a_offset = ao;
    *b_offset = bo;
}
