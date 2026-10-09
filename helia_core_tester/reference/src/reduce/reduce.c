/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Reductions over the axes set in axis_mask (bit d = input dim d). The output holds the kept
 * dims in order, whether or not reduced dims stay as 1s.
 *   mean_s8/s16:          TFLite's QuantizedMeanOrSum (mean_prepare folds 1/count into the
 *                         multiplier as TFLM does, by truncating division after a shift).
 *   mean_f32/f16, reduce_sum_f32/f16: the exact sum (or mean) rounded once.
 *   reduce_max/min_s8/s16: selection on raw codes (input and output share a quantization).
 *   reduce_max/min_f32/f16: ns-cmsis-nn#498: a NaN anywhere gives the canonical quiet NaN,
 *                         a tie keeps the first value (so +0 and -0 keep their order), an
 *                         empty domain gives -inf / +inf, an empty mask copies the bits.
 *   arg_max/arg_min:      the first index of the extremum; a NaN never wins, all-NaN is 0.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hct_ref_internal.h"

typedef struct
{
    int64_t in_count, out_count, domain;
    int32_t rank;
    int64_t out_stride[HCT_MAX_RANK]; /* 0 on a reduced dim */
    int32_t dims[HCT_MAX_RANK];
} ReduceMap;

static int32_t reduce_map(const HctTensor *in, int64_t in_count, const HctTensor *out, int64_t out_count,
                          int32_t axis_mask, ReduceMap *m)
{
    if (in->rank < 1 || axis_mask < 0 || (axis_mask >> in->rank) != 0)
    {
        return HCT_E_PARAM;
    }
    m->rank = in->rank;
    m->in_count = in_count;
    m->out_count = 1;
    m->domain = 1;
    int64_t stride = 1;
    for (int32_t d = in->rank - 1; d >= 0; --d)
    {
        m->dims[d] = in->dims[d];
        if (axis_mask & (1 << d))
        {
            m->out_stride[d] = 0;
            m->domain *= in->dims[d];
        }
        else
        {
            m->out_stride[d] = stride;
            stride *= in->dims[d];
            m->out_count *= in->dims[d];
        }
    }
    /* The output is the kept dims in order: equal counts and the kept extents in order. */
    if (m->out_count != out_count)
    {
        return HCT_E_SHAPE;
    }
    int32_t oi = out->rank - 1;
    for (int32_t d = in->rank - 1; d >= 0; --d)
    {
        if (axis_mask & (1 << d))
        {
            if (oi >= 0 && out->rank == in->rank)
            {
                if (out->dims[oi] != 1)
                {
                    return HCT_E_SHAPE;
                }
                --oi;
            }
            continue;
        }
        if (oi < 0 || out->dims[oi] != in->dims[d])
        {
            return HCT_E_SHAPE;
        }
        --oi;
    }
    return HCT_OK;
}

/* Output slot of each input element, in row-major input order. */
static void reduce_slots(const ReduceMap *m, int64_t *slot)
{
    int32_t coord[HCT_MAX_RANK] = {0};
    for (int64_t i = 0; i < m->in_count; ++i)
    {
        int64_t o = 0;
        for (int32_t d = 0; d < m->rank; ++d)
        {
            o += coord[d] * m->out_stride[d];
        }
        slot[i] = o;
        for (int32_t d = m->rank - 1; d >= 0; --d)
        {
            if (++coord[d] < m->dims[d])
            {
                break;
            }
            coord[d] = 0;
        }
    }
}

static int32_t setup(const HctAxisMaskParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                     int32_t num_outputs, int32_t in_dtype, int32_t out_dtype, ReduceMap *m, int64_t **slot)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    int64_t in_count = 0;
    int64_t out_count = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], in_dtype, &in_count));
    HCT_TRY(hct_check_tensor(&outputs[0], out_dtype, &out_count));
    HCT_TRY(reduce_map(&inputs[0], in_count, &outputs[0], out_count, p->axis_mask, m));
    *slot = malloc((size_t)(in_count > 0 ? in_count : 1) * sizeof(int64_t));
    if (*slot == NULL)
    {
        return HCT_E_SIZE;
    }
    reduce_slots(m, *slot);
    return HCT_OK;
}

/* ------------------------------------------------------------------ mean */

int32_t hct_ref_mean_prepare(const HctMeanQuant *in, HctMeanParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    if (!(isfinite(in->input_scale) && in->input_scale > 0.0f && isfinite(in->output_scale) &&
          in->output_scale > 0.0f) ||
        in->count < 1)
    {
        return HCT_E_PARAM;
    }
    int32_t multiplier = 0;
    int32_t shift = 0;
    HCT_TRY(hct_quantize_multiplier_impl((double)in->input_scale / (double)in->output_scale, &multiplier, &shift));
    /* 63 - CountLeadingZeros(count): the index of its top bit. */
    int32_t fold = 0;
    while (fold < 62 && ((int64_t)1 << (fold + 1)) <= in->count)
    {
        ++fold;
    }
    fold = fold < 32 ? fold : 32;
    fold = fold < 31 + shift ? fold : 31 + shift;
    if (fold < 0)
    {
        return HCT_E_PARAM;
    }
    out->multiplier = (int32_t)(((int64_t)multiplier << fold) / in->count);
    out->shift = shift - fold;
    return out->shift >= -31 ? HCT_OK : HCT_E_PARAM;
}

static int32_t mean_quantized(const HctMeanQuantizedParams *p, const HctTensor *inputs, int32_t num_inputs,
                              HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (p->input_zero_point < qmin || p->input_zero_point > qmax || p->output_zero_point < qmin ||
        p->output_zero_point > qmax || p->multiplier < 0 || p->shift < -31 || p->shift > 30)
    {
        return HCT_E_PARAM;
    }
    const HctAxisMaskParams mask = {p->axis_mask};
    ReduceMap m;
    int64_t *slot = NULL;
    HCT_TRY(setup(&mask, inputs, num_inputs, outputs, num_outputs, dtype, dtype, &m, &slot));
    int64_t *sum = calloc((size_t)(m.out_count > 0 ? m.out_count : 1), sizeof(int64_t));
    if (sum == NULL)
    {
        free(slot);
        return HCT_E_SIZE;
    }
    for (int64_t i = 0; i < m.in_count; ++i)
    {
        sum[slot[i]] += hct_load_i32(&inputs[0], i);
    }
    int32_t status = HCT_OK;
    for (int64_t o = 0; o < m.out_count && m.domain > 0; ++o)
    {
        /* TFLite keeps the sum, and the zero-point-shifted sum, in int32. */
        const int64_t shifted = sum[o] - (int64_t)p->input_zero_point * m.domain;
        if (sum[o] < INT32_MIN || sum[o] > INT32_MAX || shifted < INT32_MIN || shifted > INT32_MAX)
        {
            status = HCT_E_PARAM;
            break;
        }
        int64_t v = (int64_t)hct_multiply_by_quantized_multiplier((int32_t)shifted, p->multiplier, p->shift) +
                    p->output_zero_point;
        hct_store_i32(&outputs[0], o, (int32_t)(v < qmin ? qmin : (v > qmax ? qmax : v)));
    }
    if (m.domain == 0)
    {
        /* TFLite leaves the zero-filled output when a reduced dim is empty. */
        for (int64_t o = 0; o < m.out_count; ++o)
        {
            hct_store_i32(&outputs[0], o, 0);
        }
    }
    free(sum);
    free(slot);
    return status;
}

int32_t hct_ref_mean_s8(const HctMeanQuantizedParams *params, const HctTensor *inputs, int32_t num_inputs,
                        HctTensor *outputs, int32_t num_outputs)
{
    return mean_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_mean_s16(const HctMeanQuantizedParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return mean_quantized(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

static int32_t sum_float(const HctAxisMaskParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                         int32_t num_outputs, int32_t dtype, int is_mean)
{
    ReduceMap m;
    int64_t *slot = NULL;
    HCT_TRY(setup(p, inputs, num_inputs, outputs, num_outputs, dtype, dtype, &m, &slot));
    HctXsum *sum = calloc((size_t)(m.out_count > 0 ? m.out_count : 1), sizeof(HctXsum));
    if (sum == NULL)
    {
        free(slot);
        return HCT_E_SIZE;
    }
    for (int64_t i = 0; i < m.in_count; ++i)
    {
        hct_xsum_add(&sum[slot[i]], (double)hct_load_f32(&inputs[0], i));
    }
    int32_t status = HCT_OK;
    for (int64_t o = 0; o < m.out_count && status == HCT_OK; ++o)
    {
        if (is_mean && m.domain == 0)
        {
            hct_store_rounded(&outputs[0], o, NAN, -INFINITY, INFINITY); /* 0 / 0 */
        }
        else
        {
            status = hct_store_xsum(&outputs[0], o, &sum[o], is_mean ? m.domain : 1, -INFINITY, INFINITY);
        }
    }
    free(sum);
    free(slot);
    return status;
}

int32_t hct_ref_mean_f32(const HctAxisMaskParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return sum_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32, 1);
}

int32_t hct_ref_mean_f16(const HctAxisMaskParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return sum_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16, 1);
}

int32_t hct_ref_reduce_sum_f32(const HctAxisMaskParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    return sum_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32, 0);
}

int32_t hct_ref_reduce_sum_f16(const HctAxisMaskParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    return sum_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16, 0);
}

/* --------------------------------------------------------------- extrema */

static int32_t extrema_int(const HctAxisMaskParams *p, const HctTensor *inputs, int32_t num_inputs,
                           HctTensor *outputs, int32_t num_outputs, int32_t dtype, int is_max)
{
    ReduceMap m;
    int64_t *slot = NULL;
    HCT_TRY(setup(p, inputs, num_inputs, outputs, num_outputs, dtype, dtype, &m, &slot));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    int32_t *best = malloc((size_t)(m.out_count > 0 ? m.out_count : 1) * sizeof(int32_t));
    if (best == NULL)
    {
        free(slot);
        return HCT_E_SIZE;
    }
    for (int64_t o = 0; o < m.out_count; ++o)
    {
        best[o] = is_max ? qmin : qmax;
    }
    for (int64_t i = 0; i < m.in_count; ++i)
    {
        const int32_t v = hct_load_i32(&inputs[0], i);
        if (is_max ? v > best[slot[i]] : v < best[slot[i]])
        {
            best[slot[i]] = v;
        }
    }
    for (int64_t o = 0; o < m.out_count; ++o)
    {
        hct_store_i32(&outputs[0], o, best[o]);
    }
    free(best);
    free(slot);
    return HCT_OK;
}

/* Bits of element i (binary32 or binary16) and an integer key ordered as the value is,
 * with both zeros keyed 0 so they tie. */
static uint32_t float_bits(const HctTensor *t, int64_t i)
{
    if (t->dtype == HCT_FLOAT16)
    {
        return ((const uint16_t *)t->data)[i];
    }
    uint32_t b = 0;
    memcpy(&b, &((const float *)t->data)[i], sizeof(b));
    return b;
}

static void float_store_bits(HctTensor *t, int64_t i, uint32_t b)
{
    if (t->dtype == HCT_FLOAT16)
    {
        ((uint16_t *)t->data)[i] = (uint16_t)b;
    }
    else
    {
        memcpy(&((float *)t->data)[i], &b, sizeof(b));
    }
}

static int is_nan_bits(uint32_t b, int half)
{
    return half ? (b & 0x7FFFu) > 0x7C00u : (b & 0x7FFFFFFFu) > 0x7F800000u;
}

static int64_t order_key(uint32_t b, int half)
{
    const uint32_t sign = half ? 0x8000u : 0x80000000u;
    const int64_t mag = (int64_t)(b & (sign - 1u));
    return (b & sign) ? -mag : mag;
}

static int32_t extrema_float(const HctAxisMaskParams *p, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs, int32_t dtype, int is_max)
{
    ReduceMap m;
    int64_t *slot = NULL;
    HCT_TRY(setup(p, inputs, num_inputs, outputs, num_outputs, dtype, dtype, &m, &slot));
    const int half = dtype == HCT_FLOAT16;
    if (p->axis_mask == 0)
    {
        for (int64_t i = 0; i < m.in_count; ++i)
        {
            float_store_bits(&outputs[0], i, float_bits(&inputs[0], i));
        }
        free(slot);
        return HCT_OK;
    }
    uint32_t *best = malloc((size_t)(m.out_count > 0 ? m.out_count : 1) * sizeof(uint32_t));
    unsigned char *state = calloc((size_t)(m.out_count > 0 ? m.out_count : 1), 1); /* 0 empty, 1 value, 2 NaN */
    if (best == NULL || state == NULL)
    {
        free(best);
        free(state);
        free(slot);
        return HCT_E_SIZE;
    }
    for (int64_t i = 0; i < m.in_count; ++i)
    {
        const int64_t o = slot[i];
        const uint32_t b = float_bits(&inputs[0], i);
        if (state[o] == 2)
        {
            continue;
        }
        if (is_nan_bits(b, half))
        {
            state[o] = 2;
            continue;
        }
        if (state[o] == 0 || (is_max ? order_key(b, half) > order_key(best[o], half)
                                     : order_key(b, half) < order_key(best[o], half)))
        {
            best[o] = b;
            state[o] = 1;
        }
    }
    const uint32_t qnan = half ? 0x7E00u : 0x7FC00000u;
    const uint32_t inf = half ? 0x7C00u : 0x7F800000u;
    const uint32_t sign = half ? 0x8000u : 0x80000000u;
    for (int64_t o = 0; o < m.out_count; ++o)
    {
        float_store_bits(&outputs[0], o, state[o] == 2 ? qnan : (state[o] == 1 ? best[o] : (is_max ? sign | inf : inf)));
    }
    free(best);
    free(state);
    free(slot);
    return HCT_OK;
}

#define HCT_EXTREMA_ENTRY(NAME, IMPL, DTYPE, IS_MAX)                                                                   \
    int32_t hct_ref_##NAME(const HctAxisMaskParams *params, const HctTensor *inputs, int32_t num_inputs,                \
                           HctTensor *outputs, int32_t num_outputs)                                                    \
    {                                                                                                                  \
        return IMPL(params, inputs, num_inputs, outputs, num_outputs, DTYPE, IS_MAX);                                 \
    }

HCT_EXTREMA_ENTRY(reduce_max_s8, extrema_int, HCT_INT8, 1)
HCT_EXTREMA_ENTRY(reduce_max_s16, extrema_int, HCT_INT16, 1)
HCT_EXTREMA_ENTRY(reduce_min_s8, extrema_int, HCT_INT8, 0)
HCT_EXTREMA_ENTRY(reduce_min_s16, extrema_int, HCT_INT16, 0)
HCT_EXTREMA_ENTRY(reduce_max_f32, extrema_float, HCT_FLOAT32, 1)
HCT_EXTREMA_ENTRY(reduce_max_f16, extrema_float, HCT_FLOAT16, 1)
HCT_EXTREMA_ENTRY(reduce_min_f32, extrema_float, HCT_FLOAT32, 0)
HCT_EXTREMA_ENTRY(reduce_min_f16, extrema_float, HCT_FLOAT16, 0)

/* ------------------------------------------------------------------- arg */

static int32_t arg_extrema(const HctAxisParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                           int32_t num_outputs, int32_t dtype, int is_max)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    if (inputs != NULL && num_inputs >= 1 && (p->axis < 0 || p->axis >= inputs[0].rank))
    {
        return HCT_E_PARAM;
    }
    const HctAxisMaskParams mask = {1 << (p->axis >= 0 && p->axis < 30 ? p->axis : 0)};
    ReduceMap m;
    int64_t *slot = NULL;
    HCT_TRY(setup(&mask, inputs, num_inputs, outputs, num_outputs, dtype, HCT_INT32, &m, &slot));
    if (m.domain == 0)
    {
        free(slot);
        return HCT_E_SHAPE; /* an empty axis has no index */
    }
    const int is_float = dtype == HCT_FLOAT32 || dtype == HCT_FLOAT16;
    const int half = dtype == HCT_FLOAT16;
    int64_t *best_key = malloc((size_t)(m.out_count > 0 ? m.out_count : 1) * sizeof(int64_t));
    int32_t *best_idx = calloc((size_t)(m.out_count > 0 ? m.out_count : 1), sizeof(int32_t));
    unsigned char *seen = calloc((size_t)(m.out_count > 0 ? m.out_count : 1), 1);
    int32_t *pos = calloc((size_t)(m.out_count > 0 ? m.out_count : 1), sizeof(int32_t));
    if (best_key == NULL || best_idx == NULL || seen == NULL || pos == NULL)
    {
        free(best_key);
        free(best_idx);
        free(seen);
        free(pos);
        free(slot);
        return HCT_E_SIZE;
    }
    for (int64_t i = 0; i < m.in_count; ++i)
    {
        const int64_t o = slot[i];
        const int32_t index = pos[o]++;
        int64_t key = 0;
        if (is_float)
        {
            const uint32_t b = float_bits(&inputs[0], i);
            if (is_nan_bits(b, half))
            {
                continue;
            }
            key = order_key(b, half);
        }
        else
        {
            key = hct_load_i32(&inputs[0], i);
        }
        if (!seen[o] || (is_max ? key > best_key[o] : key < best_key[o]))
        {
            seen[o] = 1;
            best_key[o] = key;
            best_idx[o] = index;
        }
    }
    for (int64_t o = 0; o < m.out_count; ++o)
    {
        ((int32_t *)outputs[0].data)[o] = best_idx[o];
    }
    free(best_key);
    free(best_idx);
    free(seen);
    free(pos);
    free(slot);
    return HCT_OK;
}

#define HCT_ARG_ENTRY(NAME, DTYPE, IS_MAX)                                                                             \
    int32_t hct_ref_##NAME(const HctAxisParams *params, const HctTensor *inputs, int32_t num_inputs,                    \
                           HctTensor *outputs, int32_t num_outputs)                                                    \
    {                                                                                                                  \
        return arg_extrema(params, inputs, num_inputs, outputs, num_outputs, DTYPE, IS_MAX);                          \
    }

HCT_ARG_ENTRY(arg_max_s8, HCT_INT8, 1)
HCT_ARG_ENTRY(arg_max_s16, HCT_INT16, 1)
HCT_ARG_ENTRY(arg_max_f32, HCT_FLOAT32, 1)
HCT_ARG_ENTRY(arg_max_f16, HCT_FLOAT16, 1)
HCT_ARG_ENTRY(arg_min_s8, HCT_INT8, 0)
HCT_ARG_ENTRY(arg_min_s16, HCT_INT16, 0)
HCT_ARG_ENTRY(arg_min_f32, HCT_FLOAT32, 0)
HCT_ARG_ENTRY(arg_min_f16, HCT_FLOAT16, 0)
