/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Inference batch normalization folded to a per-channel affine map over the last dim:
 * output = input * scale[c] + bias[c], rounded once (a fused multiply-add). For binary16
 * the binary64 sum of the exact product and the bias is exact, so one rounding to binary16
 * is the fused result.
 */
#include <math.h>

#include "hct_ref_internal.h"

static int32_t batch_norm_float(const HctNoParams *p, const HctTensor *inputs, int32_t num_inputs,
                                HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    int64_t count = 0;
    int64_t n_out = 0;
    int64_t n_scale = 0;
    int64_t n_bias = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], dtype, &count));
    HCT_TRY(hct_check_tensor(&inputs[1], dtype, &n_scale));
    HCT_TRY(hct_check_tensor(&inputs[2], dtype, &n_bias));
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &n_out));
    if (inputs[0].rank < 1 || inputs[1].rank != 1 || inputs[2].rank != 1 || outputs[0].rank != inputs[0].rank)
    {
        return HCT_E_SHAPE;
    }
    for (int32_t d = 0; d < inputs[0].rank; ++d)
    {
        if (inputs[0].dims[d] != outputs[0].dims[d])
        {
            return HCT_E_SHAPE;
        }
    }
    const int64_t channels = inputs[0].dims[inputs[0].rank - 1];
    if (channels < 1 || n_scale != channels || n_bias != channels)
    {
        return HCT_E_SHAPE;
    }
    for (int64_t i = 0; i < count; ++i)
    {
        const int64_t c = i % channels;
        const float x = hct_load_f32(&inputs[0], i);
        const float s = hct_load_f32(&inputs[1], c);
        const float b = hct_load_f32(&inputs[2], c);
        const double v = dtype == HCT_FLOAT16 ? (double)x * (double)s + (double)b : (double)fmaf(x, s, b);
        hct_store_rounded(&outputs[0], i, v, -INFINITY, INFINITY);
    }
    return HCT_OK;
}

int32_t hct_ref_batch_norm_f32(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    return batch_norm_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_batch_norm_f16(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    return batch_norm_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
