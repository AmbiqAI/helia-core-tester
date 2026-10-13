/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * BatchMatMul: TFLite's reference_ops::BatchMatMul. out[..., i, j] = sum_k op(lhs)[..., i, k] *
 * op(rhs)[..., k, j], op transposing the last two dims of a stored operand when adj_x / adj_y
 * is set; batch dims (all but the last two, up to three) broadcast numpy-style. Quantized:
 * operands offset by their zero points, an int32 accumulator for int8 and int64 for int16,
 * then one per-tensor MultiplyByQuantizedMultiplier. Float: exact sums rounded once.
 */
#include "hct_ref_internal.h"

typedef struct
{
    int32_t rows, cols, depth;
    int32_t batch[3];
    int64_t lhs_stride[3], rhs_stride[3];
    int64_t batches;
} BmmShape;

static int32_t bmm_shape(const HctTensor *lhs, const HctTensor *rhs, const HctTensor *out, int32_t adj_x,
                         int32_t adj_y, BmmShape *s)
{
    const int32_t lr = lhs->rank;
    const int32_t rr = rhs->rank;
    if (lr < 2 || rr < 2 || lr > 5 || rr > 5 || out->rank != (lr > rr ? lr : rr))
    {
        return HCT_E_SHAPE;
    }
    const int32_t l_rows = lhs->dims[lr - 2];
    const int32_t l_cols = lhs->dims[lr - 1];
    const int32_t r_rows = rhs->dims[rr - 2];
    const int32_t r_cols = rhs->dims[rr - 1];
    s->rows = adj_x ? l_cols : l_rows;
    s->depth = adj_x ? l_rows : l_cols;
    const int32_t r_depth = adj_y ? r_cols : r_rows;
    s->cols = adj_y ? r_rows : r_cols;
    if (s->depth != r_depth || out->dims[out->rank - 2] != s->rows || out->dims[out->rank - 1] != s->cols)
    {
        return HCT_E_SHAPE;
    }
    /* Batch dims, extended on the left to three, with element strides 0 where a dim broadcasts. */
    int64_t l_stride = (int64_t)l_rows * l_cols;
    int64_t r_stride = (int64_t)r_rows * r_cols;
    s->batches = 1;
    for (int32_t i = 2; i >= 0; --i)
    {
        const int32_t li = lr - 3 - (2 - i);
        const int32_t ri = rr - 3 - (2 - i);
        const int32_t oi = out->rank - 3 - (2 - i);
        const int32_t ld = li >= 0 ? lhs->dims[li] : 1;
        const int32_t rd = ri >= 0 ? rhs->dims[ri] : 1;
        if (ld != rd && ld != 1 && rd != 1)
        {
            return HCT_E_SHAPE;
        }
        s->batch[i] = ld > rd ? ld : rd;
        if ((oi >= 0 ? out->dims[oi] : 1) != s->batch[i])
        {
            return HCT_E_SHAPE;
        }
        s->lhs_stride[i] = ld == 1 ? 0 : l_stride;
        s->rhs_stride[i] = rd == 1 ? 0 : r_stride;
        l_stride *= ld;
        r_stride *= rd;
        s->batches *= s->batch[i];
    }
    return HCT_OK;
}

/* Element offsets of op(lhs)[b, i, k] and op(rhs)[b, k, j]. */
static int64_t lhs_at(const BmmShape *s, int64_t base, int32_t adj_x, int32_t i, int32_t k)
{
    return base + (adj_x ? (int64_t)k * s->rows + i : (int64_t)i * s->depth + k);
}

static int64_t rhs_at(const BmmShape *s, int64_t base, int32_t adj_y, int32_t k, int32_t j)
{
    return base + (adj_y ? (int64_t)j * s->depth + k : (int64_t)k * s->cols + j);
}

#define HCT_BMM_LOOP(s, adj_x, adj_y, BODY)                                                                            \
    for (int32_t b0 = 0; b0 < (s).batch[0]; ++b0)                                                                     \
        for (int32_t b1 = 0; b1 < (s).batch[1]; ++b1)                                                                 \
            for (int32_t b2 = 0; b2 < (s).batch[2]; ++b2)                                                             \
            {                                                                                                          \
                const int64_t lb = b0 * (s).lhs_stride[0] + b1 * (s).lhs_stride[1] + b2 * (s).lhs_stride[2];          \
                const int64_t rb = b0 * (s).rhs_stride[0] + b1 * (s).rhs_stride[1] + b2 * (s).rhs_stride[2];          \
                const int64_t ob = (((int64_t)b0 * (s).batch[1] + b1) * (s).batch[2] + b2) * (s).rows * (s).cols;     \
                for (int32_t i = 0; i < (s).rows; ++i)                                                                 \
                    for (int32_t j = 0; j < (s).cols; ++j)                                                             \
                    {                                                                                                  \
                        BODY;                                                                                          \
                    }                                                                                                  \
            }

static int32_t bmm_quantized(const HctBatchMatMulParams *p, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], dtype, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &n));
    if ((p->adj_x != 0 && p->adj_x != 1) || (p->adj_y != 0 && p->adj_y != 1))
    {
        return HCT_E_PARAM;
    }
    BmmShape s;
    HCT_TRY(bmm_shape(&inputs[0], &inputs[1], &outputs[0], p->adj_x, p->adj_y, &s));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (p->multiplier < 0 || p->shift < -31 || p->shift > (dtype == HCT_INT16 ? 7 : 30) ||
        p->activation_min > p->activation_max || p->activation_min < qmin || p->activation_max > qmax ||
        (dtype == HCT_INT16 ? (p->lhs_offset != 0 || p->rhs_offset != 0 || p->output_offset != 0)
                            : (p->lhs_offset < -127 || p->lhs_offset > 128 || p->rhs_offset < -127 ||
                               p->rhs_offset > 128 || p->output_offset < qmin || p->output_offset > qmax)))
    {
        return HCT_E_PARAM;
    }
    HCT_BMM_LOOP(s, p->adj_x, p->adj_y, {
        int64_t acc = 0;
        for (int32_t k = 0; k < s.depth; ++k)
        {
            acc += (int64_t)(hct_load_i32(&inputs[0], lhs_at(&s, lb, p->adj_x, i, k)) + p->lhs_offset) *
                   (hct_load_i32(&inputs[1], rhs_at(&s, rb, p->adj_y, k, j)) + p->rhs_offset);
        }
        int64_t scaled = 0;
        if (dtype == HCT_INT16)
        {
            if (acc < -((int64_t)1 << 47) || acc >= ((int64_t)1 << 47))
            {
                return HCT_E_PARAM;
            }
            scaled = hct_multiply_by_quantized_multiplier_64(acc, p->multiplier, p->shift);
            if (scaled < INT32_MIN || scaled > INT32_MAX)
            {
                return HCT_E_PARAM;
            }
        }
        else
        {
            if (acc < INT32_MIN || acc > INT32_MAX)
            {
                return HCT_E_PARAM;
            }
            scaled = hct_multiply_by_quantized_multiplier((int32_t)acc, p->multiplier, p->shift);
        }
        int64_t v = scaled + p->output_offset;
        v = v < p->activation_min ? p->activation_min : (v > p->activation_max ? p->activation_max : v);
        hct_store_i32(&outputs[0], ob + (int64_t)i * s.cols + j, (int32_t)v);
    })
    return HCT_OK;
}

int32_t hct_ref_batch_matmul_s8(const HctBatchMatMulParams *params, const HctTensor *inputs, int32_t num_inputs,
                                HctTensor *outputs, int32_t num_outputs)
{
    return bmm_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_batch_matmul_s16(const HctBatchMatMulParams *params, const HctTensor *inputs, int32_t num_inputs,
                                 HctTensor *outputs, int32_t num_outputs)
{
    return bmm_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

static int32_t bmm_float(const HctBatchMatMulFloatParams *p, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 2, outputs, num_outputs, 1));
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], dtype, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &n));
    if ((p->adj_x != 0 && p->adj_x != 1) || (p->adj_y != 0 && p->adj_y != 1))
    {
        return HCT_E_PARAM;
    }
    BmmShape s;
    HCT_TRY(bmm_shape(&inputs[0], &inputs[1], &outputs[0], p->adj_x, p->adj_y, &s));
    HCT_TRY(hct_check_float_activation(p->activation_min, p->activation_max));
    HCT_BMM_LOOP(s, p->adj_x, p->adj_y, {
        HctXsum total;
        hct_xsum_init(&total);
        for (int32_t k = 0; k < s.depth; ++k)
        {
            hct_xsum_add_product(&total, hct_load_f32(&inputs[0], lhs_at(&s, lb, p->adj_x, i, k)),
                                 hct_load_f32(&inputs[1], rhs_at(&s, rb, p->adj_y, k, j)));
        }
        HCT_TRY(hct_store_xsum(&outputs[0], ob + (int64_t)i * s.cols + j, &total, 1, p->activation_min,
                               p->activation_max));
    })
    return HCT_OK;
}

int32_t hct_ref_batch_matmul_f32(const HctBatchMatMulFloatParams *params, const HctTensor *inputs,
                                 int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return bmm_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_batch_matmul_f16(const HctBatchMatMulFloatParams *params, const HctTensor *inputs,
                                 int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return bmm_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
