/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exact accumulation of binary64 terms as a signed fixed-point integer of 32-bit limbs
 * (value = sum(limb[i] * 2^(32 i + HCT_XSUM_BASE))), and one correct rounding of the sum, or of
 * the sum divided by a positive integer, to binary32 or binary16.
 */
#include <math.h>
#include <string.h>

#include "hct_ref_internal.h"

#define LIMB_BITS 32
/* Limbs hold signed partial sums; carries are pushed up before any limb can overflow. */
#define CARRY_EVERY (1 << 28)

void hct_xsum_init(HctXsum *x)
{
    memset(x, 0, sizeof(*x));
}

static void carry(HctXsum *x)
{
    for (int i = 0; i < HCT_XSUM_LIMBS - 1; ++i)
    {
        const int64_t c = x->limb[i] >> LIMB_BITS; /* arithmetic: floor division */
        x->limb[i] -= c * ((int64_t)1 << LIMB_BITS);
        x->limb[i + 1] += c;
    }
    x->pending = 0;
}

void hct_xsum_add(HctXsum *x, double v)
{
    if (isnan(v))
    {
        x->nan = 1;
        return;
    }
    if (isinf(v))
    {
        if (v > 0)
        {
            x->pos_inf = 1;
        }
        else
        {
            x->neg_inf = 1;
        }
        return;
    }
    if (v == 0.0)
    {
        return;
    }
    int e = 0;
    const double frac = frexp(fabs(v), &e); /* |v| = frac * 2^e, frac in [0.5, 1) */
    const uint64_t m = (uint64_t)ldexp(frac, 53);
    const int32_t lsb = e - 53 - HCT_XSUM_BASE; /* bit index of m's lowest bit */
    if (lsb < 0 || lsb + 53 > LIMB_BITS * (HCT_XSUM_LIMBS - 2))
    {
        x->out_of_range = 1;
        return;
    }
    const int32_t idx = lsb / LIMB_BITS;
    const int32_t sh = lsb % LIMB_BITS;
    /* m << sh spans at most 85 bits: three limbs. */
    const uint64_t lo = (m << sh) & 0xFFFFFFFFu;
    const uint64_t mid = (sh == 0 ? m >> 32 : (m >> (32 - sh))) & 0xFFFFFFFFu;
    const uint64_t hi = sh == 0 ? 0 : m >> (64 - sh);
    const int64_t sign = v < 0 ? -1 : 1;
    x->limb[idx] += sign * (int64_t)lo;
    x->limb[idx + 1] += sign * (int64_t)mid;
    x->limb[idx + 2] += sign * (int64_t)hi;
    if (++x->pending >= CARRY_EVERY)
    {
        carry(x);
    }
}

void hct_xsum_add_product(HctXsum *x, float a, float b)
{
    /* A binary32 product (24 x 24 bits, exponent >= -298) is exact in binary64. */
    hct_xsum_add(x, (double)a * (double)b);
}

int32_t hct_xsum_round(const HctXsum *in, int64_t divisor, int32_t dtype, double *out)
{
    if (in->out_of_range || divisor < 1 || divisor > HCT_MAX_ELEMENTS ||
        (dtype != HCT_FLOAT32 && dtype != HCT_FLOAT16))
    {
        return HCT_E_PARAM;
    }
    if (in->nan || (in->pos_inf && in->neg_inf))
    {
        *out = NAN;
        return HCT_OK;
    }
    if (in->pos_inf || in->neg_inf)
    {
        *out = in->pos_inf ? INFINITY : -INFINITY;
        return HCT_OK;
    }
    HctXsum x = *in;
    carry(&x);
    const int negative = x.limb[HCT_XSUM_LIMBS - 1] < 0;
    uint32_t mag[HCT_XSUM_LIMBS];
    if (negative)
    {
        /* Two's complement negation of the limb array (each limb now in [0, 2^32) but the top). */
        int64_t c = 1;
        for (int i = 0; i < HCT_XSUM_LIMBS; ++i)
        {
            const int64_t t = (int64_t)(~(uint32_t)x.limb[i] & 0xFFFFFFFFu) + c;
            mag[i] = (uint32_t)t;
            c = t >> LIMB_BITS;
        }
    }
    else
    {
        for (int i = 0; i < HCT_XSUM_LIMBS; ++i)
        {
            mag[i] = (uint32_t)x.limb[i];
        }
    }
    /* Divide by the divisor: quotient in place, remainder as sticky. */
    uint64_t rem = 0;
    for (int i = HCT_XSUM_LIMBS - 1; i >= 0; --i)
    {
        const uint64_t cur = (rem << LIMB_BITS) | mag[i]; /* rem < divisor <= 2^26 */
        mag[i] = (uint32_t)(cur / (uint64_t)divisor);
        rem = cur % (uint64_t)divisor;
    }
    int sticky = rem != 0;
    int32_t top = -1;
    for (int i = HCT_XSUM_LIMBS - 1; i >= 0 && top < 0; --i)
    {
        if (mag[i] != 0)
        {
            int32_t b = 31;
            while (((mag[i] >> b) & 1u) == 0)
            {
                --b;
            }
            top = i * LIMB_BITS + b;
        }
    }
    if (top < 0)
    {
        /* Zero, or a remainder below 2^HCT_XSUM_BASE, far under half the smallest subnormal. */
        *out = negative ? -0.0 : 0.0;
        return HCT_OK;
    }
    const int32_t precision = dtype == HCT_FLOAT32 ? 24 : 11;
    const int32_t min_quantum = dtype == HCT_FLOAT32 ? -149 : -24;
    int32_t quantum = top + HCT_XSUM_BASE - (precision - 1);
    quantum = quantum < min_quantum ? min_quantum : quantum;
    const int32_t k = quantum - HCT_XSUM_BASE; /* bits below k are rounded off */
    uint64_t mant = 0;
    for (int32_t bit = top; bit >= k; --bit)
    {
        mant = (mant << 1) | ((mag[bit / LIMB_BITS] >> (bit % LIMB_BITS)) & 1u);
    }
    /* k > 0 always: HCT_XSUM_BASE is below both minimum quanta. */
    const int round_bit = (mag[(k - 1) / LIMB_BITS] >> ((k - 1) % LIMB_BITS)) & 1u;
    for (int32_t bit = k - 2; bit >= 0 && !sticky; --bit)
    {
        sticky = (mag[bit / LIMB_BITS] >> (bit % LIMB_BITS)) & 1u;
    }
    if (round_bit && (sticky || (mant & 1u)))
    {
        ++mant;
    }
    double value = ldexp((double)mant, quantum);
    const double limit = dtype == HCT_FLOAT32 ? ldexp(1.0, 128) : 65536.0;
    if (value >= limit)
    {
        value = INFINITY;
    }
    *out = negative ? -value : value;
    return HCT_OK;
}
