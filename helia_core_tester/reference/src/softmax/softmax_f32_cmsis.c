/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variant: ns-cmsis-nn's arm_softmax_f32 on one target, bit for bit. Rows are the input's last dim.
 *
 * rows == 1 and cols == 2 runs arm_nn_softmax_1x2_f32 on every target: the smaller input minus the larger, the
 * LUT exp (arm_nn_softmax_exp_lut_f32, its interpolation fused except on soft float), sum = e0 + e1 and both
 * outputs divided by it.
 * target 0 (Helium): the row maximum is vmaxnm over the row and the lowest finite float (a NaN never wins);
 *   each x - max goes through arm_nn_vexpq_poly_mve_f32: m = trunc(x * log2e) (saturating, NaN -> 0),
 *   r = x - m ln2 fused, Estrin's polynomial with fused multiply-adds, m << 23 added to the bits with
 *   saturation, zero when m < -126. Each group of four exps (zero past the row) reduces as ((0+1)+2)+3 and is
 *   added to the running sum; every exp is multiplied by 1 / sum.
 * target 1 (scalar with an FPU) and 2 (soft float): the maximum by `>` from the first element, the LUT exp
 *   of x - max (fused interpolation on 1, none on 2), a running sum, every exp times 1 / sum.
 * A row whose sum is not positive makes the kernel return ARM_CMSIS_NN_ARG_ERROR; only nonfinite input reaches
 * it, and this variant writes NaN for that row so a nonfinite probe sees it as affected.
 */
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hct_ref_internal.h"

#define LANES 4

/* arm_nn_exp_poly_coeffs_f32 (binary32 bits); tests pin them to the checkout's arm_nntables_flt.c. */
static const uint32_t poly_bits[8] = {0x3f800000u, 0x3d2aa390u, 0x3f00000au, 0x3ab91bcbu,
                                      0x3f800001u, 0x3c0897a6u, 0x3e2aaa6au, 0x394d4a8bu};

static float bits_f32(uint32_t u)
{
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static uint32_t f32_bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

/* VCVT.S32.F32: toward zero, saturating, NaN -> 0. */
static int32_t cvt_s32(float f)
{
    if (isnan(f))
    {
        return 0;
    }
    if (f >= 2147483648.0f)
    {
        return INT32_MAX;
    }
    if (f <= -2147483648.0f)
    {
        return INT32_MIN;
    }
    return (int32_t)f;
}

static int32_t sat32(int64_t v)
{
    return v > INT32_MAX ? INT32_MAX : (v < INT32_MIN ? INT32_MIN : (int32_t)v);
}

/* VMAXNM: a NaN operand loses to a number; +0 beats -0. */
static float maxnm(float a, float b)
{
    if (isnan(a))
    {
        return b;
    }
    if (isnan(b))
    {
        return a;
    }
    if (a == b)
    {
        return signbit(a) ? b : a;
    }
    return a > b ? a : b;
}

/* One lane of arm_nn_vexpq_poly_mve_f32. */
static float exp_poly(float x)
{
    const float log2e = bits_f32(0x3fb8aa3bu);
    const float ln2 = bits_f32(0x3f317218u);
    float c[8];
    for (int i = 0; i < 8; ++i)
    {
        c[i] = bits_f32(poly_bits[i]);
    }
    const int32_t m = cvt_s32(x * log2e);
    const float r = fmaf(-(float)m, ln2, x);
    const float a = fmaf(c[4], r, c[0]);
    const float b = fmaf(c[6], r, c[2]);
    const float cc = fmaf(c[5], r, c[1]);
    const float d = fmaf(c[7], r, c[3]);
    const float x2 = r * r;
    const float x4 = x2 * x2;
    float y = fmaf(fmaf(d, x2, cc), x4, fmaf(b, x2, a));
    const int32_t shifted = sat32((int64_t)m * (1LL << 23));
    y = bits_f32((uint32_t)sat32((int64_t)(int32_t)f32_bits(y) + shifted));
    return m < -126 ? 0.0f : y;
}

/* arm_nn_softmax_exp_lut_f32; the interpolation fused unless on soft float. */
static float exp_lut(int fused, float x)
{
    x = x < 80.0f ? x : 80.0f;
    x = x > -80.0f ? x : -80.0f;
    const float t = x * 1.44269504088896341f;
    int32_t n = (int32_t)t;
    n = t < (float)n ? n - 1 : n;
    const float idx_f = (t - (float)n) * 256.0f;
    int32_t idx = (int32_t)idx_f;
    idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
    const float frac = idx_f - (float)idx;
    const float y0 = bits_f32(hct_cmsis_exp2_lut_f32_bits[idx]);
    const float y1 = bits_f32(hct_cmsis_exp2_lut_f32_bits[idx + 1]);
    const float y = fused ? fmaf(y1 - y0, frac, y0) : y0 + (float)((y1 - y0) * frac);
    n = n < -126 ? -126 : (n > 127 ? 127 : n);
    return y * bits_f32((uint32_t)(n + 127) << 23);
}

static void softmax_1x2(int fused, const float in[2], float out[2])
{
    float e0 = 1.0f;
    float e1 = 1.0f;
    if (in[0] >= in[1])
    {
        e1 = exp_lut(fused, in[1] - in[0]);
    }
    else
    {
        e0 = exp_lut(fused, in[0] - in[1]);
    }
    const float sum = e0 + e1;
    out[0] = e0 / sum;
    out[1] = e1 / sum;
}

int32_t hct_ref_softmax_f32_cmsis(const HctCmsisTargetParams *params, const HctTensor *inputs, int32_t num_inputs,
                                  HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    int64_t count = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], HCT_FLOAT32, &count));
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT32, &count));
    HCT_TRY(hct_check_same_shape(&inputs[0], &outputs[0]));
    if (params->target < 0 || params->target > 2)
    {
        return HCT_E_PARAM;
    }
    if (inputs[0].rank < 1)
    {
        return HCT_E_SHAPE;
    }
    const int64_t cols = inputs[0].dims[inputs[0].rank - 1];
    if (count == 0)
    {
        return HCT_OK;
    }
    const int32_t target = params->target;
    const int fused = target != 2;
    const float *in = (const float *)inputs[0].data;
    float *out = (float *)outputs[0].data;
    const int64_t rows = count / cols;
    if (rows == 1 && cols == 2)
    {
        softmax_1x2(fused, in, out);
        return HCT_OK;
    }
    for (int64_t r = 0; r < rows; ++r)
    {
        const float *x = in + r * cols;
        float *y = out + r * cols;
        float sum = 0.0f;
        if (target == 0)
        {
            float max = -FLT_MAX;
            for (int64_t c = 0; c < cols; ++c)
            {
                max = maxnm(max, x[c]);
            }
            for (int64_t c0 = 0; c0 < cols; c0 += LANES)
            {
                float lane[LANES] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int64_t k = 0; k < LANES && c0 + k < cols; ++k)
                {
                    lane[k] = exp_poly(x[c0 + k] - max);
                    y[c0 + k] = lane[k];
                }
                sum += ((lane[0] + lane[1]) + lane[2]) + lane[3];
            }
        }
        else
        {
            float max = x[0];
            for (int64_t c = 1; c < cols; ++c)
            {
                if (x[c] > max)
                {
                    max = x[c];
                }
            }
            for (int64_t c = 0; c < cols; ++c)
            {
                y[c] = exp_lut(fused, x[c] - max);
                sum += y[c];
            }
        }
        if (sum <= 0.0f)
        {
            for (int64_t c = 0; c < cols; ++c)
            {
                y[c] = NAN;
            }
            continue;
        }
        const float inv = 1.0f / sum;
        for (int64_t c = 0; c < cols; ++c)
        {
            y[c] *= inv;
        }
    }
    return HCT_OK;
}
