/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variants: ns-cmsis-nn's arm_reduce_sum_f16 and arm_nn_mean_f16 on Helium, bit for bit. The output keeps
 * the input's rank with every reduced dim 1. Each output sums its reduced elements, widened exactly, in float32
 * from +0, taking them in row-major order:
 *   route 0: one running float32 sum (the generic, spatial and middle-block paths);
 *   route 1: the flatten path's eight lanes: element j goes to lane j % 8; lanes 2i and 2i + 1 are added, and
 *            the four pair sums are added left to right ((p0 + p1) + p2) + p3.
 * reduce_sum rounds the sum to float16 once. mean multiplies it in float32 by 1 / count, itself a float32
 * division (GCC -Ofast hoists the divide out of the loop as a reciprocal), and rounds the product to float16.
 */
#include <math.h>

#include "hct_ref_internal.h"

static int32_t sum_f32acc(const HctReduceF16AccParams *p, const HctTensor *inputs, int32_t num_inputs,
                          HctTensor *outputs, int32_t num_outputs, int is_mean)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    int64_t in_count = 0;
    int64_t out_count = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], HCT_FLOAT16, &in_count));
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT16, &out_count));
    const HctTensor *in = &inputs[0];
    const int32_t rank = in->rank;
    if (rank < 1 || p->axis_mask < 0 || (p->axis_mask >> rank) != 0 || (p->route != 0 && p->route != 1))
    {
        return HCT_E_PARAM;
    }
    if (outputs[0].rank != rank)
    {
        return HCT_E_SHAPE;
    }
    int64_t stride[HCT_MAX_RANK];
    int64_t domain = 1;
    int64_t s = 1;
    for (int32_t d = rank - 1; d >= 0; --d)
    {
        const int reduced = (p->axis_mask >> d) & 1;
        if (outputs[0].dims[d] != (reduced ? 1 : in->dims[d]))
        {
            return HCT_E_SHAPE;
        }
        stride[d] = s;
        s *= in->dims[d];
        domain *= reduced ? in->dims[d] : 1;
    }
    if (out_count == 0)
    {
        return HCT_OK;
    }
    if (domain == 0)
    {
        return HCT_E_SHAPE;
    }
    const float rcp = 1.0f / (float)domain;
    for (int64_t o = 0; o < out_count; ++o)
    {
        int64_t base = 0;
        int64_t rem = o;
        for (int32_t d = rank - 1; d >= 0; --d)
        {
            const int32_t od = outputs[0].dims[d];
            base += (rem % od) * stride[d];
            rem /= od;
        }
        float lane[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float sum = 0.0f;
        for (int64_t j = 0; j < domain; ++j)
        {
            int64_t off = base;
            int64_t r = j;
            for (int32_t d = rank - 1; d >= 0; --d)
            {
                if ((p->axis_mask >> d) & 1)
                {
                    off += (r % in->dims[d]) * stride[d];
                    r /= in->dims[d];
                }
            }
            const float v = hct_load_f32(in, off);
            if (p->route == 1)
            {
                lane[j % 8] = lane[j % 8] + v;
            }
            else
            {
                sum = sum + v;
            }
        }
        if (p->route == 1)
        {
            sum = (((lane[0] + lane[1]) + (lane[2] + lane[3])) + (lane[4] + lane[5])) + (lane[6] + lane[7]);
        }
        hct_store_f32(&outputs[0], o, is_mean ? sum * rcp : sum);
    }
    return HCT_OK;
}

int32_t hct_ref_reduce_sum_f16_f32acc(const HctReduceF16AccParams *params, const HctTensor *inputs,
                                      int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return sum_f32acc(params, inputs, num_inputs, outputs, num_outputs, 0);
}

int32_t hct_ref_mean_f16_f32acc(const HctReduceF16AccParams *params, const HctTensor *inputs, int32_t num_inputs,
                                HctTensor *outputs, int32_t num_outputs)
{
    return sum_f32acc(params, inputs, num_inputs, outputs, num_outputs, 1);
}
