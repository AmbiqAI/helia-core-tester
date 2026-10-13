/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests of the shared reference primitives, built with the library sources
 * under AddressSanitizer/UBSan by tests/test_reference_c.py. Each primitive is
 * checked against an independent formulation (wide integers, exhaustive binary16
 * enumeration) rather than against itself.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "hct_ref_internal.h"

static int failures = 0;

#define CHECK(cond, ...)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            if (failures < 20)                                                                                         \
            {                                                                                                          \
                printf("FAIL %s:%d: ", __FILE__, __LINE__);                                                            \
                printf(__VA_ARGS__);                                                                                   \
                printf("\n");                                                                                          \
            }                                                                                                          \
            ++failures;                                                                                                \
        }                                                                                                              \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static uint32_t next_u32(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}

/* gemmlowp SRDHM as floor(ab / 2^31 + 1/2): ties round toward +infinity, which is
 * what its nudge (2^30, or 1 - 2^30 for a negative product) and truncation give. */
static int32_t model_srdhm(int32_t a, int32_t b)
{
    if (a == INT32_MIN && b == INT32_MIN)
    {
        return INT32_MAX;
    }
    const __int128 ab = (__int128)a * b;
    return (int32_t)((ab + ((__int128)1 << 30)) >> 31);
}

/* Round half away from zero of x / 2^e, via exact rationals. */
static int32_t model_rdbpot(int32_t x, int32_t e)
{
    if (e == 0)
    {
        return x;
    }
    const int64_t d = 1LL << e;
    const int64_t floor_q = (x >= 0) ? (int64_t)x / d : -((-(int64_t)x + d - 1) / d);
    const int64_t rem2 = 2 * ((int64_t)x - floor_q * d); /* 2 * remainder in [0, 2d) */
    if (rem2 > d || (rem2 == d && x >= 0))
    {
        return (int32_t)(floor_q + 1);
    }
    return (int32_t)floor_q;
}

static void test_srdhm(void)
{
    const int32_t edges[] = {0, 1, -1, 2, -2, INT32_MAX, INT32_MIN, INT32_MAX - 1, INT32_MIN + 1, 1 << 30, -(1 << 30),
                             (1 << 30) + 1, 12345, -12345};
    const size_t n = sizeof edges / sizeof edges[0];
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            CHECK(hct_saturating_rounding_doubling_high_mul(edges[i], edges[j]) == model_srdhm(edges[i], edges[j]),
                  "srdhm(%d, %d)", edges[i], edges[j]);
        }
    }
    for (int i = 0; i < 2000000; ++i)
    {
        const int32_t a = (int32_t)next_u32();
        const int32_t b = (int32_t)next_u32();
        CHECK(hct_saturating_rounding_doubling_high_mul(a, b) == model_srdhm(a, b), "srdhm(%d, %d)", a, b);
    }
}

static void test_rdbpot(void)
{
    for (int32_t e = 0; e <= 31; ++e)
    {
        const int32_t edges[] = {0, 1, -1, INT32_MAX, INT32_MIN, (int32_t)(1u << (e > 30 ? 30 : e)),
                                 -(int32_t)(1u << (e > 30 ? 30 : e)), 3, -3};
        for (size_t i = 0; i < sizeof edges / sizeof edges[0]; ++i)
        {
            CHECK(hct_rounding_divide_by_pot(edges[i], e) == model_rdbpot(edges[i], e), "rdbpot(%d, %d)", edges[i], e);
        }
        for (int i = 0; i < 100000; ++i)
        {
            const int32_t x = (int32_t)next_u32();
            CHECK(hct_rounding_divide_by_pot(x, e) == model_rdbpot(x, e), "rdbpot(%d, %d)", x, e);
        }
        /* exact ties: k * 2^e + 2^(e-1) */
        if (e > 0 && e < 30)
        {
            for (int64_t k = -5; k <= 5; ++k)
            {
                const int64_t wide = k * (1LL << e) + (1LL << (e - 1));
                if (wide < INT32_MIN || wide > INT32_MAX)
                {
                    continue;
                }
                const int32_t x = (int32_t)wide;
                CHECK(hct_rounding_divide_by_pot(x, e) == model_rdbpot(x, e), "tie rdbpot(%d, %d)", x, e);
            }
        }
    }
}

static void test_mbqm(void)
{
    for (int i = 0; i < 1000000; ++i)
    {
        const int32_t x = (int32_t)next_u32() >> (next_u32() % 24);
        const int32_t m = (int32_t)((1u << 30) | (next_u32() & 0x3FFFFFFFu));
        const int32_t shift = (int32_t)(next_u32() % 40) - 31; /* [-31, 8] */
        const int32_t left = shift > 0 ? shift : 0;
        const int32_t right = shift > 0 ? 0 : -shift;
        const int32_t scaled = (int32_t)((uint32_t)x << left);
        const int32_t want = model_rdbpot(model_srdhm(scaled, m), right);
        CHECK(hct_multiply_by_quantized_multiplier(x, m, shift) == want, "mbqm(%d, %d, %d)", x, m, shift);
    }
}

static void test_quantize_multiplier(void)
{
    int32_t m = 0;
    int32_t s = 0;
    CHECK(hct_quantize_multiplier_impl(0.0, &m, &s) == HCT_OK && m == 0 && s == 0, "zero");
    CHECK(hct_quantize_multiplier_impl(0.5, &m, &s) == HCT_OK && m == (1 << 30) && s == 0, "0.5 -> %d %d", m, s);
    CHECK(hct_quantize_multiplier_impl(1.0, &m, &s) == HCT_OK && m == (1 << 30) && s == 1, "1.0 -> %d %d", m, s);
    /* q rounds up to 2^31: renormalized to 2^30 with shift + 1 */
    CHECK(hct_quantize_multiplier_impl(0.99999999999, &m, &s) == HCT_OK && m == (1 << 30) && s == 1,
          "near 1 -> %d %d", m, s);
    CHECK(hct_quantize_multiplier_impl(1e-12, &m, &s) == HCT_OK && m == 0 && s == 0, "underflow -> %d %d", m, s);
    CHECK(hct_quantize_multiplier_impl(-0.5, &m, &s) == HCT_E_PARAM, "negative");
    CHECK(hct_quantize_multiplier_impl(NAN, &m, &s) == HCT_E_PARAM, "nan");
    CHECK(hct_quantize_multiplier_impl(INFINITY, &m, &s) == HCT_E_PARAM, "inf");
    CHECK(hct_quantize_multiplier_impl(1e12, &m, &s) == HCT_E_PARAM, "shift > 30");
    CHECK(hct_quantize_multiplier_impl(0.5, NULL, &s) == HCT_E_NULL, "null");
    CHECK(hct_quantize_multiplier_smaller_than_one(1.0, &m, &s) == HCT_E_PARAM, "not < 1");
    CHECK(hct_quantize_multiplier_smaller_than_one(0.0, &m, &s) == HCT_E_PARAM, "not > 0");
    /* m * 2^(shift - 31) reproduces the real within half a multiplier step */
    for (int i = 0; i < 200000; ++i)
    {
        const double real = ldexp((double)(next_u32() | 1u) / 4294967296.0, (int)(next_u32() % 40) - 30);
        CHECK(hct_quantize_multiplier_impl(real, &m, &s) == HCT_OK, "qm(%g)", real);
        if (m == 0)
        {
            continue;
        }
        CHECK(m >= (1 << 30), "normalized m for %g: %d", real, m);
        const double back = ldexp((double)m, s - 31);
        CHECK(fabs(back - real) <= ldexp(1.0, s - 32) * 1.0000001, "qm(%g) = %d, %d", real, m, s);
    }
}

static void test_activation_range(void)
{
    int32_t lo = 0;
    int32_t hi = 0;
    CHECK(hct_activation_range_quantized(HCT_ACT_NONE, HCT_INT8, 0.1f, 3, &lo, &hi) == HCT_OK && lo == -128 &&
              hi == 127,
          "none");
    CHECK(hct_activation_range_quantized(HCT_ACT_RELU, HCT_INT8, 0.1f, -5, &lo, &hi) == HCT_OK && lo == -5 && hi == 127,
          "relu");
    CHECK(hct_activation_range_quantized(HCT_ACT_RELU6, HCT_INT8, 0.05f, -100, &lo, &hi) == HCT_OK && lo == -100 &&
              hi == 20,
          "relu6 %d %d", lo, hi);
    CHECK(hct_activation_range_quantized(HCT_ACT_RELU6, HCT_INT8, 0.01f, 0, &lo, &hi) == HCT_OK && hi == 127,
          "relu6 saturates");
    CHECK(hct_activation_range_quantized(HCT_ACT_RELU_N1_TO_1, HCT_INT16, 1.0f / 32768.0f, 0, &lo, &hi) == HCT_OK &&
              lo == -32768 && hi == 32767,
          "n1_to_1 int16 %d %d", lo, hi);
    CHECK(hct_activation_range_quantized(99, HCT_INT8, 0.1f, 0, &lo, &hi) == HCT_E_PARAM, "unknown activation");
    CHECK(hct_activation_range_quantized(HCT_ACT_NONE, HCT_FLOAT32, 0.1f, 0, &lo, &hi) == HCT_E_DTYPE, "dtype");
    CHECK(hct_activation_range_quantized(HCT_ACT_NONE, HCT_INT8, 0.0f, 0, &lo, &hi) == HCT_E_PARAM, "zero scale");
    CHECK(hct_activation_range_quantized(HCT_ACT_NONE, HCT_INT8, NAN, 0, &lo, &hi) == HCT_E_PARAM, "nan scale");
    CHECK(hct_activation_range_quantized(HCT_ACT_NONE, HCT_INT8, 0.1f, 128, &lo, &hi) == HCT_E_PARAM, "zp range");
    CHECK(hct_activation_range_quantized(HCT_ACT_NONE, HCT_INT8, 0.1f, 0, NULL, &hi) == HCT_E_NULL, "null");
}

static uint32_t bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

static void test_float16(void)
{
    /* Every binary16 round-trips exactly through binary32. */
    for (uint32_t h = 0; h <= 0xFFFFu; ++h)
    {
        const float f = hct_f16_to_f32((uint16_t)h);
        const uint16_t back = hct_f32_to_f16(f);
        if (((h >> 10) & 0x1Fu) == 0x1Fu && (h & 0x3FFu))
        {
            CHECK(isnan(f) && (back & 0x7C00u) == 0x7C00u && (back & 0x3FFu) && (back & 0x8000u) == (h & 0x8000u),
                  "nan %04x -> %04x", h, back);
        }
        else
        {
            CHECK(back == h, "roundtrip %04x -> %04x", h, back);
        }
    }
    /* Midpoints between consecutive finite halves round to the even neighbour. */
    for (uint32_t h = 0; h < 0x7BFFu; ++h)
    {
        const double lo = (double)hct_f16_to_f32((uint16_t)h);
        const double hi = (double)hct_f16_to_f32((uint16_t)(h + 1));
        const float mid = (float)((lo + hi) / 2.0);
        if ((double)mid != (lo + hi) / 2.0)
        {
            continue; /* not representable in binary32: no exact tie */
        }
        const uint16_t want = (h & 1u) ? (uint16_t)(h + 1) : (uint16_t)h;
        CHECK(hct_f32_to_f16(mid) == want, "tie between %04x and %04x -> %04x", h, h + 1, hct_f32_to_f16(mid));
        CHECK(hct_f32_to_f16(nextafterf(mid, INFINITY)) == (uint16_t)(h + 1), "above tie %04x", h);
        CHECK(hct_f32_to_f16(nextafterf(mid, 0.0f)) == (uint16_t)h || mid == 0.0f, "below tie %04x", h);
    }
    CHECK(hct_f32_to_f16(65504.0f) == 0x7BFFu, "max");
    CHECK(hct_f32_to_f16(65519.99f) == 0x7BFFu, "below overflow");
    CHECK(hct_f32_to_f16(65520.0f) == 0x7C00u, "overflow tie -> inf");
    CHECK(hct_f32_to_f16(1e30f) == 0x7C00u && hct_f32_to_f16(-1e30f) == 0xFC00u, "large");
    CHECK(hct_f32_to_f16(INFINITY) == 0x7C00u && hct_f32_to_f16(-INFINITY) == 0xFC00u, "inf");
    CHECK(hct_f32_to_f16(ldexpf(1.0f, -24)) == 0x0001u, "min subnormal");
    CHECK(hct_f32_to_f16(ldexpf(1.0f, -25)) == 0x0000u, "half min subnormal ties to zero");
    CHECK(hct_f32_to_f16(ldexpf(1.5f, -25)) == 0x0001u, "above half min subnormal");
    CHECK(hct_f32_to_f16(-0.0f) == 0x8000u, "negative zero");
    CHECK(hct_f32_to_f16(1e-45f) == 0x0000u, "binary32 subnormal");
    /* binary64 -> binary16 rounds once: just above a binary16 tie rounds up even where
     * binary32 cannot hold the excess, which a binary32 intermediate would round onto the tie. */
    CHECK(hct_f64_to_f16(ldexp(1.0, -25) * (1.0 + ldexp(1.0, -40))) == 0x0001u, "above subnormal tie");
    CHECK(hct_f64_to_f16(ldexp(1.0, -25) * (1.0 - ldexp(1.0, -40))) == 0x0000u, "below subnormal tie");
    CHECK(hct_f64_to_f16(1.0 + ldexp(1.0, -11) + ldexp(1.0, -40)) == 0x3C01u, "above normal tie");
    CHECK(hct_f64_to_f16(-(1.0 + ldexp(1.0, -11) + ldexp(1.0, -40))) == 0xBC01u, "above normal tie, negative");
    CHECK(hct_f64_to_f16(1.0 + ldexp(1.0, -11)) == 0x3C00u, "exact normal tie to even");
    CHECK(hct_f64_to_f16(1e300) == 0x7C00u && hct_f64_to_f16(-1e300) == 0xFC00u, "binary64 overflow");
    CHECK(hct_f64_to_f16(65519.999999) == 0x7BFFu && hct_f64_to_f16(65520.0) == 0x7C00u, "overflow edge");
    CHECK(hct_f64_to_f16(1e-300) == 0x0000u && hct_f64_to_f16(-1e-300) == 0x8000u, "binary64 tiny");
    CHECK((hct_f64_to_f16(NAN) & 0x7E00u) == 0x7E00u, "nan");
    CHECK(hct_f64_to_f16(INFINITY) == 0x7C00u, "inf");
    CHECK(bits(hct_f16_to_f32(0x8000u)) == 0x80000000u, "widen -0");
    CHECK(hct_f16_to_f32(0x0001u) == ldexpf(1.0f, -24), "widen min subnormal");
}

static void test_tensor_and_broadcast(void)
{
    int8_t data[24] = {0};
    int64_t count = 0;
    HctTensor t = {HCT_INT8, 3, {2, 3, 4}, data};
    CHECK(hct_check_tensor(&t, HCT_INT8, &count) == HCT_OK && count == 24, "valid");
    CHECK(hct_check_tensor(&t, HCT_INT16, &count) == HCT_E_DTYPE, "dtype");
    CHECK(hct_check_tensor(NULL, HCT_INT8, &count) == HCT_E_NULL, "null tensor");
    HctTensor bad_rank = {HCT_INT8, HCT_MAX_RANK + 1, {1}, data};
    CHECK(hct_check_tensor(&bad_rank, HCT_INT8, &count) == HCT_E_SHAPE, "rank");
    HctTensor neg = {HCT_INT8, 1, {-1}, data};
    CHECK(hct_check_tensor(&neg, HCT_INT8, &count) == HCT_E_SHAPE, "negative dim");
    HctTensor no_data = {HCT_INT8, 1, {4}, NULL};
    CHECK(hct_check_tensor(&no_data, HCT_INT8, &count) == HCT_E_NULL, "null data");
    HctTensor empty = {HCT_INT8, 2, {3, 0}, NULL};
    CHECK(hct_check_tensor(&empty, HCT_INT8, &count) == HCT_OK && count == 0, "empty");
    HctTensor huge = {HCT_INT8, 4, {4096, 4096, 4096, 4096}, data};
    CHECK(hct_check_tensor(&huge, HCT_INT8, &count) == HCT_E_SIZE, "size");
    HctTensor scalar = {HCT_INT8, 0, {0}, data};
    CHECK(hct_check_tensor(&scalar, HCT_INT8, &count) == HCT_OK && count == 1, "rank 0");

    /* [2,1,4] + [3,1] -> [2,3,4]: a stretches dim 1, b stretches dims 0 and 2 */
    HctTensor a = {HCT_INT8, 3, {2, 1, 4}, data};
    HctTensor b = {HCT_INT8, 2, {3, 1}, data};
    HctTensor o = {HCT_INT8, 3, {2, 3, 4}, data};
    HctBroadcast2 bc;
    CHECK(hct_broadcast2_init(&bc, &a, &b, &o) == HCT_OK && bc.count == 24, "broadcast init");
    for (int64_t i = 0; i < 24; ++i)
    {
        int64_t ao = 0;
        int64_t bo = 0;
        hct_broadcast2_offsets(&bc, i, &ao, &bo);
        const int64_t n = i / 12, h = (i / 4) % 3, w = i % 4;
        CHECK(ao == n * 4 + w && bo == h, "offsets %lld -> %lld %lld", (long long)i, (long long)ao, (long long)bo);
    }
    HctTensor wrong_out = {HCT_INT8, 3, {2, 3, 5}, data};
    CHECK(hct_broadcast2_init(&bc, &a, &b, &wrong_out) == HCT_E_SHAPE, "output mismatch");
    HctTensor clash = {HCT_INT8, 1, {3}, data};
    CHECK(hct_broadcast2_init(&bc, &a, &clash, &o) == HCT_E_SHAPE, "incompatible");
    HctTensor out_rank = {HCT_INT8, 4, {1, 2, 3, 4}, data};
    CHECK(hct_broadcast2_init(&bc, &a, &b, &out_rank) == HCT_E_SHAPE, "output rank");
}

#define RUN(test)                                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        printf("%s\n", #test);                                                                                         \
        fflush(stdout);                                                                                                \
        test();                                                                                                        \
    } while (0)

int main(void)
{
    RUN(test_srdhm);
    RUN(test_rdbpot);
    RUN(test_mbqm);
    RUN(test_quantize_multiplier);
    RUN(test_activation_range);
    RUN(test_float16);
    RUN(test_tensor_and_broadcast);
    if (failures)
    {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("ok\n");
    return 0;
}
