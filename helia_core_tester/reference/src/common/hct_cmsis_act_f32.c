/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * ns-cmsis-nn's float32 sigmoid and tanh per target (see hct_cmsis_act_f32.h).
 */
#include "hct_cmsis_act_f32.h"

#include <math.h>
#include <string.h>

#include "hct_ref_internal.h"

static float bits_f(uint32_t b)
{
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

static float madd(int fused, float x, float y, float z)
{
    return fused ? fmaf(x, y, z) : (float)(x * y) + z;
}

/* arm_nn_softmax_exp_lut_f32. */
static float exp_lut(int fused, float x)
{
    x = x < 80.0f ? x : 80.0f;
    x = x > -80.0f ? x : -80.0f;
    const float t = x * 1.44269504088896341f;
    int32_t n = (int32_t)t;
    n = t < (float)n ? n - 1 : n;
    const float f = t - (float)n;
    const float idx_f = f * 256.0f;
    int32_t idx = (int32_t)idx_f;
    idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
    const float frac = idx_f - (float)idx;
    const float y0 = bits_f(hct_cmsis_exp2_lut_f32_bits[idx]);
    const float y1 = bits_f(hct_cmsis_exp2_lut_f32_bits[idx + 1]);
    n = n < -126 ? -126 : (n > 127 ? 127 : n);
    return madd(fused, y1 - y0, frac, y0) * bits_f((uint32_t)(n + 127) << 23);
}

int32_t hct_cmsis_sigmoid_f32(float x, int32_t target, float *out)
{
    if (out == NULL)
    {
        return HCT_E_NULL;
    }
    if (target < 0 || target > 2)
    {
        return HCT_E_PARAM;
    }
    const int fused = target != 2;
    if (x >= 0.0f)
    {
        *out = 1.0f / (1.0f + exp_lut(fused, -x));
        return HCT_OK;
    }
    const float e = exp_lut(fused, x);
    *out = e / (1.0f + e);
    return HCT_OK;
}

int32_t hct_cmsis_tanh_f32(float x, int32_t target, float *out)
{
    if (out == NULL)
    {
        return HCT_E_NULL;
    }
    if (target < 0 || target > 2)
    {
        return HCT_E_PARAM;
    }
    if (isnan(x))
    {
        *out = x;
        return HCT_OK;
    }
    const float ax = fabsf(x);
    if (!(ax < 6.0f))
    {
        *out = copysignf(1.0f, x);
        return HCT_OK;
    }
    /* t >= 0, so truncation (scalar) and vcvtm's floor (Helium) give the same index. */
    const float t = ax * 64.0f;
    int32_t idx = (int32_t)t;
    idx = idx > 383 ? 383 : idx;
    const float frac = t - (float)idx;
    const float y0 = bits_f(hct_cmsis_tanh_lut_f32_bits[idx]);
    const float y1 = bits_f(hct_cmsis_tanh_lut_f32_bits[idx + 1]);
    *out = copysignf(madd(target != 2, y1 - y0, frac, y0), x);
    return HCT_OK;
}
