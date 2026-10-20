/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * arm_nn_activation_f32/f16 (no TFLite counterpart as one op).
 *   nn_activation_f32/f16: each activation's exact result (binary64) rounded once to the output.
 *   tanh_lut_f16, tanh_lut_mve_f16: CMSIS-NN's float16 tanh, a 257-entry table over [0, 4] with
 *   linear interpolation; named variants, as the table's error is the kernel's contract. The
 *   scalar form rounds each binary16 step; the MVE form fuses the interpolation (one rounding).
 *   nn_activation_f32_cmsis: arm_nn_activation_f32 on one target (0 Helium, 1 scalar with an FPU as GCC -Ofast
 *   contracts it, 2 soft float). Helium runs tanh, hardswish, relu and relu6 as vectors (tanh flushes a
 *   subnormal input to a signed zero, maxnm/minnm give +0 for -0) and sigmoid, leaky relu and none as the
 *   scalar loop; the scalar targets run arm_nn_apply_activation_type_f32. Sigmoid and tanh are the shared
 *   hct_cmsis_sigmoid_f32 / hct_cmsis_tanh_f32; hardswish is x * clamp(x / 6 + 1/2, 0, 1) with the x / 6 + 1/2
 *   fused except on soft float.
 *   nn_activation_f16_mve: arm_nn_activation_f16 on Helium. Tanh is tanh_lut_mve_f16; hardswish one float16
 *   fused multiply-add x * h(1/6) + 1/2, maxnm 0, minnm 1, times x; sigmoid the scalar float16 leg: exp from
 *   the float16 exp2 table with float32 intermediates (the index from t rounded to float16, the interpolation
 *   fused), rounded to float16, then 1 / (1 + e) or e / (1 + e) in float32 rounded to float16.
 */
#include <math.h>
#include <string.h>

#include "hct_cmsis_act_f32.h"
#include "hct_ref_internal.h"

/* binary16 of tanh(4 i / 256) for i = 0..256 (arm_nn_tanh_lut_f16's values). */
const uint16_t hct_cmsis_tanh_lut_f16[257] = {
    0x0000u, 0x2400u, 0x27FFu, 0x29FFu, 0x2BFDu, 0x2CFDu, 0x2DFCu, 0x2EF9u, 0x2FF5u, 0x3078u, 0x30F6u, 0x3172u,
    0x31EEu, 0x3269u, 0x32E4u, 0x335Eu, 0x33D6u, 0x3427u, 0x3463u, 0x349Du, 0x34D8u, 0x3512u, 0x354Bu, 0x3584u,
    0x35BCu, 0x35F3u, 0x362Au, 0x3660u, 0x3696u, 0x36CBu, 0x36FFu, 0x3732u, 0x3765u, 0x3797u, 0x37C8u, 0x37F9u,
    0x3814u, 0x382Cu, 0x3843u, 0x3859u, 0x3870u, 0x3886u, 0x389Bu, 0x38B1u, 0x38C5u, 0x38DAu, 0x38EEu, 0x3902u,
    0x3915u, 0x3928u, 0x393Au, 0x394Cu, 0x395Eu, 0x3970u, 0x3981u, 0x3991u, 0x39A2u, 0x39B2u, 0x39C1u, 0x39D0u,
    0x39DFu, 0x39EEu, 0x39FCu, 0x3A0Au, 0x3A18u, 0x3A25u, 0x3A32u, 0x3A3Fu, 0x3A4Bu, 0x3A57u, 0x3A63u, 0x3A6Eu,
    0x3A79u, 0x3A84u, 0x3A8Fu, 0x3A99u, 0x3AA3u, 0x3AADu, 0x3AB7u, 0x3AC0u, 0x3AC9u, 0x3AD2u, 0x3ADBu, 0x3AE3u,
    0x3AEBu, 0x3AF3u, 0x3AFBu, 0x3B03u, 0x3B0Au, 0x3B11u, 0x3B18u, 0x3B1Fu, 0x3B25u, 0x3B2Cu, 0x3B32u, 0x3B38u,
    0x3B3Eu, 0x3B43u, 0x3B49u, 0x3B4Eu, 0x3B54u, 0x3B59u, 0x3B5Eu, 0x3B62u, 0x3B67u, 0x3B6Cu, 0x3B70u, 0x3B74u,
    0x3B78u, 0x3B7Du, 0x3B80u, 0x3B84u, 0x3B88u, 0x3B8Cu, 0x3B8Fu, 0x3B92u, 0x3B96u, 0x3B99u, 0x3B9Cu, 0x3B9Fu,
    0x3BA2u, 0x3BA5u, 0x3BA7u, 0x3BAAu, 0x3BADu, 0x3BAFu, 0x3BB2u, 0x3BB4u, 0x3BB6u, 0x3BB9u, 0x3BBBu, 0x3BBDu,
    0x3BBFu, 0x3BC1u, 0x3BC3u, 0x3BC5u, 0x3BC6u, 0x3BC8u, 0x3BCAu, 0x3BCBu, 0x3BCDu, 0x3BCFu, 0x3BD0u, 0x3BD2u,
    0x3BD3u, 0x3BD4u, 0x3BD6u, 0x3BD7u, 0x3BD8u, 0x3BD9u, 0x3BDBu, 0x3BDCu, 0x3BDDu, 0x3BDEu, 0x3BDFu, 0x3BE0u,
    0x3BE1u, 0x3BE2u, 0x3BE3u, 0x3BE4u, 0x3BE5u, 0x3BE5u, 0x3BE6u, 0x3BE7u, 0x3BE8u, 0x3BE9u, 0x3BE9u, 0x3BEAu,
    0x3BEBu, 0x3BEBu, 0x3BECu, 0x3BEDu, 0x3BEDu, 0x3BEEu, 0x3BEEu, 0x3BEFu, 0x3BEFu, 0x3BF0u, 0x3BF0u, 0x3BF1u,
    0x3BF1u, 0x3BF2u, 0x3BF2u, 0x3BF3u, 0x3BF3u, 0x3BF3u, 0x3BF4u, 0x3BF4u, 0x3BF5u, 0x3BF5u, 0x3BF5u, 0x3BF6u,
    0x3BF6u, 0x3BF6u, 0x3BF6u, 0x3BF7u, 0x3BF7u, 0x3BF7u, 0x3BF8u, 0x3BF8u, 0x3BF8u, 0x3BF8u, 0x3BF9u, 0x3BF9u,
    0x3BF9u, 0x3BF9u, 0x3BF9u, 0x3BFAu, 0x3BFAu, 0x3BFAu, 0x3BFAu, 0x3BFAu, 0x3BFBu, 0x3BFBu, 0x3BFBu, 0x3BFBu,
    0x3BFBu, 0x3BFBu, 0x3BFBu, 0x3BFCu, 0x3BFCu, 0x3BFCu, 0x3BFCu, 0x3BFCu, 0x3BFCu, 0x3BFCu, 0x3BFCu, 0x3BFDu,
    0x3BFDu, 0x3BFDu, 0x3BFDu, 0x3BFDu, 0x3BFDu, 0x3BFDu, 0x3BFDu, 0x3BFDu, 0x3BFDu, 0x3BFEu, 0x3BFEu, 0x3BFEu,
    0x3BFEu, 0x3BFEu, 0x3BFEu, 0x3BFEu, 0x3BFEu, 0x3BFEu, 0x3BFEu, 0x3BFEu, 0x3BFEu, 0x3BFEu, 0x3BFEu, 0x3BFEu,
    0x3BFEu, 0x3BFEu, 0x3BFFu, 0x3BFFu, 0x3BFFu,
};

static double activate(double x, int32_t type, double act_param)
{
    switch (type)
    {
    case HCT_FACT_SIGMOID:
        return 1.0 / (1.0 + exp(-x));
    case HCT_FACT_TANH:
        return tanh(x);
    case HCT_FACT_HARDSWISH:
    {
        double r = x + 3.0;
        r = r < 0.0 ? 0.0 : (r > 6.0 ? 6.0 : r);
        return x * r / 6.0;
    }
    case HCT_FACT_LEAKY_RELU:
        return x >= 0.0 ? x : x * act_param;
    case HCT_FACT_RELU:
        return x < 0.0 ? 0.0 : x;
    case HCT_FACT_RELU6:
        return x < 0.0 ? 0.0 : (x > 6.0 ? 6.0 : x);
    default:
        return x;
    }
}

static int32_t nn_activation(const HctFloatActivationTypeParams *p, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    if (p->activation_type < HCT_FACT_NONE || p->activation_type > HCT_FACT_HARDSWISH ||
        (p->activation_type == HCT_FACT_LEAKY_RELU && !isfinite(p->act_param)))
    {
        return HCT_E_PARAM;
    }
    for (int64_t i = 0; i < count; ++i)
    {
        /* LeakyRelu's x * a is exact in binary64, so its one rounding is the IEEE product. */
        const double y = activate((double)hct_load_f32(&inputs[0], i), p->activation_type, (double)p->act_param);
        if (dtype == HCT_FLOAT16)
        {
            ((uint16_t *)outputs[0].data)[i] = hct_f64_to_f16(y);
        }
        else
        {
            ((float *)outputs[0].data)[i] = (float)y;
        }
    }
    return HCT_OK;
}

int32_t hct_ref_nn_activation_f32(const HctFloatActivationTypeParams *params, const HctTensor *inputs,
                                  int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return nn_activation(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_nn_activation_f16(const HctFloatActivationTypeParams *params, const HctTensor *inputs,
                                  int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    return nn_activation(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}

/* ------------------------------------------------- CMSIS-NN float16 tanh */

static float lut(int32_t i)
{
    return hct_f16_to_f32(hct_cmsis_tanh_lut_f16[i]);
}

static uint16_t tanh_lut_one(uint16_t bits, int fused)
{
    if ((bits & 0x7FFFu) > 0x7C00u)
    {
        /* NaN: the scalar kernel passes it through; the vector one yields a quiet NaN. */
        return fused ? (uint16_t)((bits & 0x7FFFu) | 0x0200u) : bits;
    }
    const float x = hct_f16_to_f32(bits);
    const float ax = fabsf(x);
    float magnitude = 1.0f;
    if (!(ax > 4.0f))
    {
        /* Scaling by 64 and splitting off the integer part are exact in binary16. */
        const float t = ax * 64.0f;
        const int32_t idx = (int32_t)t < 255 ? (int32_t)t : 255;
        const float frac = t - (float)idx;
        const float y0 = lut(idx);
        const float diff = hct_round_f16(lut(idx + 1) - y0);
        magnitude = fused ? hct_f16_to_f32(hct_f64_to_f16((double)y0 + (double)diff * (double)frac))
                          : hct_round_f16(y0 + hct_round_f16(diff * frac));
    }
    if (fused)
    {
        return hct_f32_to_f16(x < 0.0f ? -magnitude : magnitude);
    }
    return (uint16_t)((hct_f32_to_f16(magnitude) & 0x7FFFu) | (bits & 0x8000u));
}

static int32_t tanh_lut(const HctNoParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs, int fused)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16, HCT_FLOAT16, &count));
    for (int64_t i = 0; i < count; ++i)
    {
        ((uint16_t *)outputs[0].data)[i] = tanh_lut_one(((const uint16_t *)inputs[0].data)[i], fused);
    }
    return HCT_OK;
}

int32_t hct_ref_tanh_lut_f16(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    return tanh_lut(params, inputs, num_inputs, outputs, num_outputs, 0);
}

int32_t hct_ref_tanh_lut_mve_f16(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                                 HctTensor *outputs, int32_t num_outputs)
{
    return tanh_lut(params, inputs, num_inputs, outputs, num_outputs, 1);
}

/* ------------------------------------------- kernel-faithful (ns-cmsis-nn arithmetic per target) */

static float bits_f32(uint32_t u)
{
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* VMAXNM / VMINNM on numbers: +0 beats -0. NaN operands are handled by the callers. */
static float maxnm(float a, float b)
{
    if (a == b)
    {
        return signbit(a) ? b : a;
    }
    return a > b ? a : b;
}

static float minnm(float a, float b)
{
    if (a == b)
    {
        return signbit(a) ? a : b;
    }
    return a < b ? a : b;
}

static int32_t check_activation(const HctFloatActivationTypeParams *p)
{
    if (p->activation_type < HCT_FACT_NONE || p->activation_type > HCT_FACT_HARDSWISH ||
        (p->activation_type == HCT_FACT_LEAKY_RELU && !isfinite(p->act_param)))
    {
        return HCT_E_PARAM;
    }
    return HCT_OK;
}

static float cmsis_f32(float x, int32_t type, float act_param, int32_t target)
{
    float y = x;
    switch (type)
    {
    case HCT_FACT_SIGMOID:
        (void)hct_cmsis_sigmoid_f32(x, target, &y);
        return y;
    case HCT_FACT_TANH:
        if (target == 0 && x != 0.0f && fabsf(x) < 0x1p-126f)
        {
            return copysignf(0.0f, x);
        }
        (void)hct_cmsis_tanh_f32(x, target, &y);
        if (target == 2 && isnan(x))
        {
            uint32_t b;
            memcpy(&b, &x, sizeof b);
            return bits_f32(b | 0x00400000u);
        }
        return y;
    case HCT_FACT_HARDSWISH:
    {
        float t = target == 2 ? (float)(x * (1.0f / 6.0f)) + 0.5f : fmaf(x, 1.0f / 6.0f, 0.5f);
        if (target == 0)
        {
            t = isnan(t) ? 0.0f : minnm(maxnm(t, 0.0f), 1.0f);
        }
        else
        {
            t = t < 1.0f ? t : 1.0f;
            t = t > 0.0f ? t : 0.0f;
        }
        return x * t;
    }
    case HCT_FACT_RELU:
        if (isnan(x))
        {
            return x;
        }
        return target == 0 ? maxnm(x, 0.0f) : (x < 0.0f ? 0.0f : x);
    case HCT_FACT_RELU6:
        if (isnan(x))
        {
            return x;
        }
        if (target == 0)
        {
            return minnm(maxnm(x, 0.0f), 6.0f);
        }
        y = x < 0.0f ? 0.0f : x;
        return y > 6.0f ? 6.0f : y;
    case HCT_FACT_LEAKY_RELU:
        return isnan(x) ? x : (x >= 0.0f ? x : act_param * x);
    default:
        return x;
    }
}

int32_t hct_ref_nn_activation_f32_cmsis(const HctFloatActivationTargetParams *params, const HctTensor *inputs,
                                        int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32, HCT_FLOAT32, &count));
    const HctFloatActivationTypeParams type = {params->activation_type, params->act_param};
    HCT_TRY(check_activation(&type));
    if (params->target < 0 || params->target > 2)
    {
        return HCT_E_PARAM;
    }
    for (int64_t i = 0; i < count; ++i)
    {
        ((float *)outputs[0].data)[i] = cmsis_f32(((const float *)inputs[0].data)[i], params->activation_type,
                                                  params->act_param, params->target);
    }
    return HCT_OK;
}

static float h16(float v)
{
    return hct_round_f16(v);
}

/* arm_nn_softmax_exp_lut_f16 on a float16 value, as GCC builds it (the interpolation fused). */
static float exp_lut_f16(float x16)
{
    float x = x16 < 80.0f ? x16 : 80.0f;
    x = x > -80.0f ? x : -80.0f;
    const float t = x * 1.44269504088896341f;
    const float tq = h16(t);
    int32_t n = (int32_t)tq;
    n = tq < (float)n ? n - 1 : n;
    const float idx_f = (t - (float)n) * 256.0f;
    int32_t idx = (int32_t)idx_f;
    idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
    const float frac = idx_f - (float)idx;
    const float y0 = hct_f16_to_f32(hct_cmsis_exp2_lut_f16[idx]);
    const float y1 = hct_f16_to_f32(hct_cmsis_exp2_lut_f16[idx + 1]);
    /* The 2^n scale stays float32 (exponent clamped to [-126, 15]) until the one float16 rounding, so a
     * result below the float16 normal range keeps its subnormal or zero (ns-cmsis-nn#861). */
    n = n > 15 ? 15 : (n < -126 ? -126 : n);
    return h16(fmaf(frac, y1 - y0, y0) * ldexpf(1.0f, n));
}

static uint16_t mve_f16(uint16_t bits, int32_t type, float act_param)
{
    const float x = hct_f16_to_f32(bits);
    switch (type)
    {
    case HCT_FACT_TANH:
        return tanh_lut_one(bits, 1);
    case HCT_FACT_HARDSWISH:
    {
        /* x * h(1/6) + 1/2 is exact in binary64 (at most 37 significant bits), so one narrowing is the fused
         * float16 result. */
        float t = hct_f16_to_f32(hct_f64_to_f16((double)x * (double)h16(1.0f / 6.0f) + 0.5));
        t = isnan(t) ? 0.0f : minnm(maxnm(t, 0.0f), 1.0f);
        return hct_f32_to_f16(x * t);
    }
    case HCT_FACT_SIGMOID:
    {
        if (isnan(x))
        {
            return bits;
        }
        if (x >= 0.0f)
        {
            const float e = exp_lut_f16(-x);
            return hct_f32_to_f16(1.0f / (1.0f + e));
        }
        const float e = exp_lut_f16(x);
        return hct_f32_to_f16(e / (1.0f + e));
    }
    case HCT_FACT_RELU:
        return isnan(x) ? bits : hct_f32_to_f16(maxnm(x, 0.0f));
    case HCT_FACT_RELU6:
        return isnan(x) ? bits : hct_f32_to_f16(minnm(maxnm(x, 0.0f), 6.0f));
    case HCT_FACT_LEAKY_RELU:
        /* max(x, 0) + h(a) * min(x, 0): one of the two terms is zero, so the sum is the product rounded once. */
        return isnan(x) ? bits : (x >= 0.0f ? bits : hct_f32_to_f16(h16(act_param) * x));
    default:
        return bits;
    }
}

int32_t hct_ref_nn_activation_f16_mve(const HctFloatActivationTypeParams *params, const HctTensor *inputs,
                                      int32_t num_inputs, HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16, HCT_FLOAT16, &count));
    HCT_TRY(check_activation(params));
    for (int64_t i = 0; i < count; ++i)
    {
        ((uint16_t *)outputs[0].data)[i] =
            mve_f16(((const uint16_t *)inputs[0].data)[i], params->activation_type, params->act_param);
    }
    return HCT_OK;
}
