/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variant: ns-cmsis-nn's arm_softmax_f16 on Helium, bit for bit. Rows are the input's last dim.
 *
 * Rows of eight lanes (one 1 x 2 tensor excepted): the row maximum is vmaxnm over the row and the lowest finite
 * float16 (a NaN never wins). Each x - max is rounded to float16 and goes through arm_nn_vexpq_poly_mve_f16 in
 * float32: m = trunc(x * log2e) (saturating, NaN -> 0), r = x - m ln2 fused, Estrin's polynomial with fused
 * multiply-adds, m << 23 added to the bits with saturation, zero when m < -126, rounded to float16 once. Each
 * group of eight exps (zero past the row) reduces in float16 as ((0+1)+(2+3))+((4+5)+(6+7)) and is added to a
 * float32 sum. 1 / sum is a float32 division rounded to float16, and every exp is multiplied by it in float16.
 *
 * rows == 1 and cols == 2 runs arm_nn_softmax_1x2_f16: the smaller input minus the larger in float16, the
 * float16 LUT exp (arm_nn_softmax_exp_lut_f16, with its interpolation fused as GCC builds it), the pair sum
 * 1 + e in float16, and both outputs divided by it in float16.
 *
 * A row whose float32 sum is not positive makes the kernel return ARM_CMSIS_NN_ARG_ERROR; only nonfinite
 * input reaches it, and this variant writes NaN for that row so a nonfinite probe sees it as affected.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hct_ref_internal.h"

#define LANES 8

/* arm_nn_exp_poly_coeffs_f16 (binary32 bits); tests pin them to the checkout's arm_nntables_flt.c. */
static const uint32_t poly_bits[8] = {0x3f800000u, 0x3d2aa390u, 0x3f00000au, 0x3ab91bcbu,
                                      0x3f800001u, 0x3c0897a6u, 0x3e2aaa6au, 0x394d4a8bu};

/* arm_nn_exp2_lut_f16: 2^(i/256) as binary16 bits, i = 0..256; pinned like poly_bits. */
static const uint16_t exp2_lut_f16[257] = {
    0x3c00u, 0x3c03u, 0x3c06u, 0x3c08u, 0x3c0bu, 0x3c0eu, 0x3c11u, 0x3c14u, 0x3c16u, 0x3c19u, 0x3c1cu, 0x3c1fu,
    0x3c22u, 0x3c25u, 0x3c28u, 0x3c2au, 0x3c2du, 0x3c30u, 0x3c33u, 0x3c36u, 0x3c39u, 0x3c3cu, 0x3c3fu, 0x3c42u,
    0x3c45u, 0x3c48u, 0x3c4bu, 0x3c4eu, 0x3c51u, 0x3c54u, 0x3c57u, 0x3c5au, 0x3c5du, 0x3c60u, 0x3c63u, 0x3c66u,
    0x3c69u, 0x3c6cu, 0x3c6fu, 0x3c72u, 0x3c75u, 0x3c78u, 0x3c7bu, 0x3c7eu, 0x3c82u, 0x3c85u, 0x3c88u, 0x3c8bu,
    0x3c8eu, 0x3c91u, 0x3c94u, 0x3c98u, 0x3c9bu, 0x3c9eu, 0x3ca1u, 0x3ca4u, 0x3ca8u, 0x3cabu, 0x3caeu, 0x3cb1u,
    0x3cb5u, 0x3cb8u, 0x3cbbu, 0x3cbeu, 0x3cc2u, 0x3cc5u, 0x3cc8u, 0x3cccu, 0x3ccfu, 0x3cd2u, 0x3cd6u, 0x3cd9u,
    0x3cdcu, 0x3ce0u, 0x3ce3u, 0x3ce7u, 0x3ceau, 0x3cedu, 0x3cf1u, 0x3cf4u, 0x3cf8u, 0x3cfbu, 0x3cffu, 0x3d02u,
    0x3d06u, 0x3d09u, 0x3d0cu, 0x3d10u, 0x3d14u, 0x3d17u, 0x3d1bu, 0x3d1eu, 0x3d22u, 0x3d25u, 0x3d29u, 0x3d2cu,
    0x3d30u, 0x3d34u, 0x3d37u, 0x3d3bu, 0x3d3eu, 0x3d42u, 0x3d46u, 0x3d49u, 0x3d4du, 0x3d51u, 0x3d54u, 0x3d58u,
    0x3d5cu, 0x3d60u, 0x3d63u, 0x3d67u, 0x3d6bu, 0x3d6fu, 0x3d72u, 0x3d76u, 0x3d7au, 0x3d7eu, 0x3d81u, 0x3d85u,
    0x3d89u, 0x3d8du, 0x3d91u, 0x3d95u, 0x3d99u, 0x3d9cu, 0x3da0u, 0x3da4u, 0x3da8u, 0x3dacu, 0x3db0u, 0x3db4u,
    0x3db8u, 0x3dbcu, 0x3dc0u, 0x3dc4u, 0x3dc8u, 0x3dccu, 0x3dd0u, 0x3dd4u, 0x3dd8u, 0x3ddcu, 0x3de0u, 0x3de4u,
    0x3de8u, 0x3decu, 0x3df0u, 0x3df5u, 0x3df9u, 0x3dfdu, 0x3e01u, 0x3e05u, 0x3e09u, 0x3e0eu, 0x3e12u, 0x3e16u,
    0x3e1au, 0x3e1eu, 0x3e23u, 0x3e27u, 0x3e2bu, 0x3e30u, 0x3e34u, 0x3e38u, 0x3e3cu, 0x3e41u, 0x3e45u, 0x3e49u,
    0x3e4eu, 0x3e52u, 0x3e57u, 0x3e5bu, 0x3e5fu, 0x3e64u, 0x3e68u, 0x3e6du, 0x3e71u, 0x3e76u, 0x3e7au, 0x3e7fu,
    0x3e83u, 0x3e88u, 0x3e8cu, 0x3e91u, 0x3e95u, 0x3e9au, 0x3e9eu, 0x3ea3u, 0x3ea8u, 0x3eacu, 0x3eb1u, 0x3eb5u,
    0x3ebau, 0x3ebfu, 0x3ec4u, 0x3ec8u, 0x3ecdu, 0x3ed2u, 0x3ed6u, 0x3edbu, 0x3ee0u, 0x3ee5u, 0x3ee9u, 0x3eeeu,
    0x3ef3u, 0x3ef8u, 0x3efdu, 0x3f02u, 0x3f06u, 0x3f0bu, 0x3f10u, 0x3f15u, 0x3f1au, 0x3f1fu, 0x3f24u, 0x3f29u,
    0x3f2eu, 0x3f33u, 0x3f38u, 0x3f3du, 0x3f42u, 0x3f47u, 0x3f4cu, 0x3f51u, 0x3f56u, 0x3f5bu, 0x3f60u, 0x3f65u,
    0x3f6au, 0x3f70u, 0x3f75u, 0x3f7au, 0x3f7fu, 0x3f84u, 0x3f8au, 0x3f8fu, 0x3f94u, 0x3f99u, 0x3f9fu, 0x3fa4u,
    0x3fa9u, 0x3faeu, 0x3fb4u, 0x3fb9u, 0x3fbfu, 0x3fc4u, 0x3fc9u, 0x3fcfu, 0x3fd4u, 0x3fdau, 0x3fdfu, 0x3fe4u,
    0x3feau, 0x3fefu, 0x3ff5u, 0x3ffau, 0x4000u,
};

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

static float h(float v)
{
    return hct_round_f16(v);
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

static float minnm(float a, float b)
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
        return signbit(a) ? a : b;
    }
    return a < b ? a : b;
}

/* One lane of arm_nn_vexpq_poly_mve_f16; x is a float16 value. */
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
    if (m < -126)
    {
        y = 0.0f;
    }
    return h(y);
}

/* arm_nn_softmax_exp_lut_f16 as GCC builds it (the interpolation is one fused multiply-add). */
static float exp_lut(float x16)
{
    float x = minnm(x16, 80.0f);
    x = maxnm(x, -80.0f);
    const float t = x * bits_f32(0x3fb8aa3bu);
    const float tq = h(t);
    int32_t n = cvt_s32(tq);
    if (tq < (float)n)
    {
        n -= 1;
    }
    const float idx_f = (t - (float)n) * 256.0f;
    int32_t idx = cvt_s32(idx_f);
    idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
    const float frac = idx_f - (float)idx;
    const float y0 = hct_f16_to_f32(exp2_lut_f16[idx]);
    const float y1 = hct_f16_to_f32(exp2_lut_f16[idx + 1]);
    const float y = fmaf(frac, y1 - y0, y0);
    n = n > 15 ? 15 : (n < -14 ? -14 : n);
    return h(y * hct_f16_to_f32((uint16_t)((n + 15) << 10)));
}

static void softmax_1x2(const float in[2], float out[2])
{
    if (in[0] >= in[1])
    {
        const float e1 = exp_lut(h(in[1] - in[0]));
        const float sum = h(1.0f + e1);
        out[0] = h(1.0f / sum);
        out[1] = h(e1 / sum);
    }
    else
    {
        const float e0 = exp_lut(h(in[0] - in[1]));
        const float sum = h(e0 + 1.0f);
        out[0] = h(e0 / sum);
        out[1] = h(1.0f / sum);
    }
}

static float reduce16(const float l[LANES])
{
    return h(h(h(l[0] + l[1]) + h(l[2] + l[3])) + h(h(l[4] + l[5]) + h(l[6] + l[7])));
}

int32_t hct_ref_softmax_f16_mve(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                                HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    int64_t count = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], HCT_FLOAT16, &count));
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT16, &count));
    HCT_TRY(hct_check_same_shape(&inputs[0], &outputs[0]));
    if (inputs[0].rank < 1)
    {
        return HCT_E_SHAPE;
    }
    const int64_t cols = inputs[0].dims[inputs[0].rank - 1];
    if (count == 0)
    {
        return HCT_OK;
    }
    const int64_t rows = count / cols;
    float *e = malloc((size_t)cols * sizeof(float));
    if (e == NULL)
    {
        return HCT_E_SIZE;
    }
    for (int64_t r = 0; r < rows; ++r)
    {
        const int64_t base = r * cols;
        if (rows == 1 && cols == 2)
        {
            const float in[2] = {hct_load_f32(&inputs[0], 0), hct_load_f32(&inputs[0], 1)};
            float out[2];
            softmax_1x2(in, out);
            hct_store_f32(&outputs[0], 0, out[0]);
            hct_store_f32(&outputs[0], 1, out[1]);
            break;
        }
        float max = -65504.0f;
        for (int64_t c = 0; c < cols; ++c)
        {
            max = maxnm(max, hct_load_f32(&inputs[0], base + c));
        }
        float sum = 0.0f;
        for (int64_t c0 = 0; c0 < cols; c0 += LANES)
        {
            float lane[LANES];
            for (int64_t k = 0; k < LANES; ++k)
            {
                lane[k] = 0.0f;
                if (c0 + k < cols)
                {
                    lane[k] = exp_poly(h(hct_load_f32(&inputs[0], base + c0 + k) - max));
                    e[c0 + k] = lane[k];
                }
            }
            sum = sum + reduce16(lane);
        }
        if (sum <= 0.0f)
        {
            for (int64_t c = 0; c < cols; ++c)
            {
                hct_store_f32(&outputs[0], base + c, NAN);
            }
            continue;
        }
        const float inv = h(1.0f / sum);
        for (int64_t c = 0; c < cols; ++c)
        {
            hct_store_f32(&outputs[0], base + c, h(e[c] * inv));
        }
    }
    free(e);
    return HCT_OK;
}
