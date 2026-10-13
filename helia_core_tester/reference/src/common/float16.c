/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * IEEE 754 binary16 <-> binary32 on bit patterns, so f16 semantics do not depend
 * on the host compiler's _Float16 support.
 */
#include <math.h>
#include <string.h>

#include "hct_ref_internal.h"

static uint32_t f32_bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

static float bits_f32(uint32_t u)
{
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

float hct_f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0x1Fu)
    {
        return bits_f32(sign | 0x7F800000u | (mant << 13));
    }
    if (exp != 0)
    {
        return bits_f32(sign | ((exp + 112u) << 23) | (mant << 13));
    }
    if (mant == 0)
    {
        return bits_f32(sign);
    }
    /* Subnormal half: normalize into a binary32 normal. */
    uint32_t e = 113u;
    while ((mant & 0x400u) == 0)
    {
        mant <<= 1;
        --e;
    }
    return bits_f32(sign | (e << 23) | ((mant & 0x3FFu) << 13));
}

/* value >> s with round-half-to-even, s in [1, 31]. */
static uint32_t shift_rne(uint32_t value, uint32_t s)
{
    const uint32_t kept = value >> s;
    const uint32_t rem = value & ((1u << s) - 1u);
    const uint32_t half = 1u << (s - 1u);
    return kept + ((rem > half || (rem == half && (kept & 1u))) ? 1u : 0u);
}

uint16_t hct_f32_to_f16(float f)
{
    const uint32_t u = f32_bits(f);
    const uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
    const int32_t exp = (int32_t)((u >> 23) & 0xFFu);
    const uint32_t mant = u & 0x7FFFFFu;
    if (exp == 0xFF)
    {
        if (mant == 0)
        {
            return (uint16_t)(sign | 0x7C00u);
        }
        /* NaN: keep the top payload bits and set the quiet bit. */
        return (uint16_t)(sign | 0x7E00u | (mant >> 13));
    }
    if (exp == 0)
    {
        return sign; /* binary32 zero or subnormal: below half's smallest subnormal / 2 */
    }
    const int32_t e = exp - 127;
    if (e >= 16)
    {
        return (uint16_t)(sign | 0x7C00u);
    }
    if (e >= -14)
    {
        /* Normal half; a rounding carry out of the mantissa bumps the exponent,
         * and out of exponent 30 lands exactly on infinity (0x7C00). */
        const uint32_t h = ((uint32_t)(e + 15) << 10) | (mant >> 13);
        const uint32_t rem = mant & 0x1FFFu;
        const uint32_t up = (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ? 1u : 0u;
        return (uint16_t)(sign | (h + up));
    }
    /* Subnormal half: value / 2^-24 = (1.mant) * 2^(e + 1) = full >> (-(e + 1)) with full = 1.mant * 2^23. */
    const uint32_t full = mant | 0x800000u;
    const uint32_t s = (uint32_t)(-(e + 1));
    if (s > 24u)
    {
        return sign;
    }
    return (uint16_t)(sign | shift_rne(full, s));
}

uint16_t hct_f64_to_f16(double d)
{
    /* Round to binary32 by round-to-odd, then to binary16: with 24 >= 11 + 2 bits the
     * second rounding is the correctly rounded binary16 of d, subnormals included. */
    float f = (float)d;
    if (isfinite(d) && isfinite(f) && (double)f != d)
    {
        if (fabs((double)f) > fabs(d))
        {
            f = nextafterf(f, 0.0f);
        }
        f = bits_f32(f32_bits(f) | 1u);
    }
    return hct_f32_to_f16(f);
}

int32_t hct_check_float_activation(float lo, float hi)
{
    if (isnan(lo) || isnan(hi) || lo > hi)
    {
        return HCT_E_PARAM;
    }
    return HCT_OK;
}
