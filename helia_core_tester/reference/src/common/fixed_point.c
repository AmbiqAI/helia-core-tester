/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fixed-point primitives with TFLite reference semantics (gemmlowp rounding,
 * double-rounding MultiplyByQuantizedMultiplier, frexp-based QuantizeMultiplier).
 */
#include <math.h>

#include "hct_ref_internal.h"

/* Right shift of a negative int32 is implementation-defined in C; every compiler
 * this library builds with shifts arithmetically, and the rounding below needs it. */
_Static_assert((-1 >> 1) == -1, "arithmetic right shift required");

int32_t hct_saturating_rounding_doubling_high_mul(int32_t a, int32_t b)
{
    if (a == INT32_MIN && b == INT32_MIN)
    {
        return INT32_MAX;
    }
    const int64_t ab = (int64_t)a * (int64_t)b;
    const int64_t nudge = ab >= 0 ? (1LL << 30) : (1LL - (1LL << 30));
    /* C division truncates toward zero, as gemmlowp's does. */
    return (int32_t)((ab + nudge) / (1LL << 31));
}

int32_t hct_rounding_divide_by_pot(int32_t x, int32_t exponent)
{
    const int32_t mask = (int32_t)((1LL << exponent) - 1);
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

int32_t hct_multiply_by_quantized_multiplier(int32_t x, int32_t multiplier, int32_t shift)
{
    const int32_t left_shift = shift > 0 ? shift : 0;
    const int32_t right_shift = shift > 0 ? 0 : -shift;
    /* x * 2^left_shift in int32, wrapping (defined via uint32) like the kernels' int32 math. */
    const int32_t scaled = (int32_t)((uint32_t)x << left_shift);
    return hct_rounding_divide_by_pot(hct_saturating_rounding_doubling_high_mul(scaled, multiplier), right_shift);
}

int32_t hct_quantize_multiplier_impl(double real, int32_t *multiplier, int32_t *shift)
{
    if (multiplier == NULL || shift == NULL)
    {
        return HCT_E_NULL;
    }
    if (!isfinite(real) || real < 0.0)
    {
        return HCT_E_PARAM;
    }
    if (real == 0.0)
    {
        *multiplier = 0;
        *shift = 0;
        return HCT_OK;
    }
    int exponent = 0;
    const double q = frexp(real, &exponent);
    /* TfLiteRound is std::round: halves away from zero. */
    int64_t q_fixed = (int64_t)round(q * (double)(1LL << 31));
    if (q_fixed == (1LL << 31))
    {
        q_fixed /= 2;
        ++exponent;
    }
    if (exponent < -31)
    {
        exponent = 0;
        q_fixed = 0;
    }
    if (exponent > 30)
    {
        return HCT_E_PARAM;
    }
    *multiplier = (int32_t)q_fixed;
    *shift = exponent;
    return HCT_OK;
}

int32_t hct_quantize_multiplier_smaller_than_one(double real, int32_t *multiplier, int32_t *shift)
{
    if (!(real > 0.0 && real < 1.0))
    {
        return HCT_E_PARAM;
    }
    HCT_TRY(hct_quantize_multiplier_impl(real, multiplier, shift));
    return *shift <= 0 ? HCT_OK : HCT_E_PARAM;
}

int32_t hct_dtype_range(int32_t dtype, int32_t *qmin, int32_t *qmax)
{
    switch (dtype)
    {
    case HCT_INT8:
        *qmin = INT8_MIN;
        *qmax = INT8_MAX;
        return HCT_OK;
    case HCT_INT16:
        *qmin = INT16_MIN;
        *qmax = INT16_MAX;
        return HCT_OK;
    default:
        return HCT_E_DTYPE;
    }
}

int32_t hct_activation_range_quantized(
    int32_t activation, int32_t dtype, float scale, int32_t zero_point, int32_t *act_min, int32_t *act_max)
{
    if (act_min == NULL || act_max == NULL)
    {
        return HCT_E_NULL;
    }
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(dtype, &qmin, &qmax));
    if (!(isfinite(scale) && scale > 0.0f) || zero_point < qmin || zero_point > qmax)
    {
        return HCT_E_PARAM;
    }
/* TFLite quantizes the bound in float: zero_point + (int32)roundf(f / scale). */
#define HCT_QUANTIZE(f) (zero_point + (int32_t)roundf((f) / scale))
    int32_t lo = qmin;
    int32_t hi = qmax;
    switch (activation)
    {
    case HCT_ACT_NONE:
        break;
    case HCT_ACT_RELU:
        lo = HCT_QUANTIZE(0.0f) > qmin ? HCT_QUANTIZE(0.0f) : qmin;
        break;
    case HCT_ACT_RELU6:
        lo = HCT_QUANTIZE(0.0f) > qmin ? HCT_QUANTIZE(0.0f) : qmin;
        hi = HCT_QUANTIZE(6.0f) < qmax ? HCT_QUANTIZE(6.0f) : qmax;
        break;
    case HCT_ACT_RELU_N1_TO_1:
        lo = HCT_QUANTIZE(-1.0f) > qmin ? HCT_QUANTIZE(-1.0f) : qmin;
        hi = HCT_QUANTIZE(1.0f) < qmax ? HCT_QUANTIZE(1.0f) : qmax;
        break;
    default:
        return HCT_E_PARAM;
    }
#undef HCT_QUANTIZE
    *act_min = lo;
    *act_max = hi;
    return HCT_OK;
}

int32_t hct_ref_quantize_multiplier(const HctQuantizeMultiplierIn *in, HctQuantizeMultiplierOut *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    return hct_quantize_multiplier_impl(in->real_multiplier, &out->multiplier, &out->shift);
}

int32_t hct_ref_activation_range_quantized(const HctActivationRangeIn *in, HctActivationRangeOut *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    return hct_activation_range_quantized(in->activation, in->dtype, in->scale, in->zero_point, &out->min, &out->max);
}
