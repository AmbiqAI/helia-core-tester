/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Softmax over the innermost dimension.
 *   softmax_s8, softmax_s8_s16: TFLite's reference_ops::Softmax<int8_t, int8_t|int16_t> (gemmlowp fixed point).
 *   softmax_s16:                TFLite's reference_ops::SoftmaxInt16 with the LUTs TFLM's prepare generates.
 *   softmax_f32/f16:            exp in binary64, the quotient rounded once to the output type.
 */
#include <math.h>

#include "hct_ref_internal.h"

/* LUTPopulate<int16_t> output for exp over [-10, 0] and 1 / (1 + x) over [0, 1] (513 entries,
 * the last only for the final slope); fixed by the op, so stored rather than regenerated. */
static const int16_t exp_lut[HCT_LUT_S16_SIZE] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7,
    7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10,
    10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13,
    13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18,
    18, 19, 19, 19, 20, 20, 21, 21, 21, 22, 22, 23, 23, 24, 24, 25,
    25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 34,
    34, 35, 36, 36, 37, 37, 38, 39, 40, 40, 42, 42, 43, 44, 45, 45,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 60, 62,
    63, 65, 65, 67, 68, 69, 71, 73, 74, 75, 77, 78, 80, 81, 83, 85,
    86, 88, 90, 92, 93, 95, 97, 99, 101, 103, 105, 107, 109, 112, 114, 116,
    118, 121, 123, 126, 128, 131, 133, 135, 139, 141, 144, 147, 149, 152, 155, 158,
    162, 165, 168, 171, 174, 178, 181, 185, 189, 192, 196, 200, 204, 208, 212, 217,
    221, 225, 230, 234, 239, 243, 248, 253, 258, 263, 268, 273, 279, 284, 290, 296,
    302, 308, 314, 320, 327, 333, 340, 346, 353, 360, 366, 374, 381, 389, 397, 404,
    413, 421, 429, 437, 446, 455, 464, 473, 482, 492, 501, 511, 522, 532, 543, 553,
    564, 575, 586, 598, 610, 622, 634, 646, 659, 672, 685, 699, 713, 727, 741, 756,
    771, 786, 801, 817, 833, 850, 866, 884, 901, 919, 937, 955, 974, 993, 1013, 1033,
    1053, 1074, 1095, 1117, 1139, 1161, 1184, 1207, 1232, 1256, 1281, 1306, 1332, 1358, 1385, 1412,
    1440, 1468, 1497, 1527, 1557, 1587, 1619, 1651, 1683, 1716, 1750, 1785, 1820, 1856, 1892, 1930,
    1968, 2006, 2046, 2087, 2128, 2170, 2212, 2256, 2300, 2346, 2392, 2439, 2488, 2537, 2587, 2638,
    2690, 2743, 2796, 2852, 2908, 2966, 3024, 3084, 3145, 3207, 3270, 3334, 3400, 3467, 3535, 3605,
    3677, 3749, 3822, 3898, 3975, 4053, 4133, 4214, 4297, 4383, 4469, 4557, 4647, 4739, 4833, 4927,
    5024, 5124, 5225, 5328, 5433, 5541, 5649, 5761, 5875, 5991, 6109, 6230, 6352, 6477, 6605, 6736,
    6868, 7004, 7141, 7282, 7427, 7572, 7722, 7874, 8030, 8188, 8350, 8514, 8683, 8854, 9028, 9206,
    9387, 9572, 9762, 9954, 10151, 10351, 10555, 10763, 10976, 11191, 11412, 11637, 11867, 12102, 12341, 12583,
    12831, 13085, 13342, 13606, 13874, 14148, 14427, 14711, 15002, 15297, 15599, 15907, 16221, 16541, 16867, 17199,
    17539, 17884, 18237, 18597, 18964, 19338, 19719, 20108, 20505, 20909, 21322, 21742, 22171, 22608, 23054, 23509,
    23973, 24445, 24928, 25419, 25921, 26432, 26953, 27485, 28027, 28580, 29143, 29718, 30304, 30902, 31512, 32133,
    32767,
};

static const int16_t one_over_one_plus_x_lut[HCT_LUT_S16_SIZE] = {
    32767, 32704, 32640, 32578, 32514, 32451, 32388, 32326, 32264, 32202, 32141, 32079, 32018, 31957, 31896, 31835,
    31775, 31715, 31655, 31596, 31537, 31476, 31418, 31359, 31301, 31242, 31184, 31127, 31069, 31011, 30954, 30897,
    30840, 30784, 30727, 30671, 30615, 30560, 30504, 30449, 30394, 30339, 30283, 30229, 30175, 30121, 30067, 30013,
    29960, 29906, 29853, 29800, 29746, 29694, 29642, 29589, 29537, 29486, 29434, 29382, 29331, 29280, 29229, 29177,
    29127, 29076, 29026, 28976, 28926, 28877, 28827, 28777, 28728, 28679, 28630, 28581, 28532, 28484, 28436, 28388,
    28340, 28292, 28244, 28197, 28150, 28103, 28056, 28008, 27962, 27915, 27869, 27823, 27777, 27731, 27685, 27640,
    27594, 27549, 27504, 27459, 27413, 27369, 27324, 27280, 27236, 27192, 27148, 27104, 27060, 27016, 26973, 26930,
    26887, 26844, 26801, 26758, 26715, 26673, 26630, 26588, 26546, 26504, 26463, 26421, 26380, 26338, 26297, 26255,
    26214, 26174, 26132, 26092, 26051, 26011, 25971, 25931, 25891, 25851, 25811, 25772, 25732, 25693, 25653, 25614,
    25575, 25536, 25497, 25458, 25420, 25381, 25343, 25305, 25267, 25229, 25191, 25153, 25116, 25078, 25041, 25003,
    24966, 24928, 24892, 24855, 24818, 24781, 24745, 24709, 24672, 24636, 24600, 24564, 24528, 24492, 24457, 24421,
    24385, 24350, 24315, 24280, 24245, 24210, 24175, 24140, 24105, 24070, 24036, 24002, 23967, 23933, 23899, 23865,
    23831, 23798, 23764, 23730, 23697, 23664, 23630, 23597, 23564, 23530, 23498, 23465, 23432, 23399, 23366, 23334,
    23302, 23269, 23237, 23205, 23173, 23141, 23109, 23077, 23046, 23014, 22982, 22951, 22920, 22888, 22857, 22826,
    22795, 22764, 22733, 22703, 22672, 22641, 22611, 22580, 22550, 22520, 22490, 22459, 22429, 22400, 22370, 22340,
    22310, 22281, 22251, 22221, 22192, 22163, 22134, 22104, 22075, 22046, 22017, 21988, 21959, 21931, 21902, 21874,
    21845, 21817, 21788, 21760, 21732, 21704, 21676, 21648, 21620, 21592, 21565, 21537, 21509, 21482, 21455, 21427,
    21400, 21372, 21345, 21318, 21291, 21264, 21237, 21210, 21183, 21157, 21130, 21103, 21077, 21050, 21024, 20998,
    20971, 20945, 20919, 20893, 20867, 20841, 20816, 20790, 20764, 20738, 20713, 20687, 20662, 20636, 20611, 20586,
    20560, 20535, 20510, 20485, 20460, 20435, 20410, 20385, 20360, 20336, 20311, 20287, 20262, 20238, 20213, 20189,
    20165, 20141, 20117, 20092, 20068, 20044, 20021, 19997, 19973, 19949, 19926, 19902, 19878, 19855, 19832, 19808,
    19784, 19762, 19738, 19715, 19692, 19668, 19645, 19622, 19600, 19577, 19553, 19531, 19508, 19485, 19463, 19440,
    19418, 19395, 19373, 19351, 19328, 19306, 19284, 19262, 19240, 19218, 19196, 19174, 19152, 19130, 19109, 19087,
    19065, 19044, 19022, 19000, 18979, 18958, 18936, 18915, 18893, 18872, 18851, 18830, 18809, 18787, 18766, 18745,
    18725, 18704, 18682, 18662, 18641, 18620, 18600, 18579, 18559, 18538, 18518, 18497, 18477, 18457, 18436, 18416,
    18396, 18376, 18356, 18336, 18316, 18296, 18276, 18256, 18236, 18216, 18197, 18177, 18157, 18138, 18118, 18099,
    18079, 18059, 18040, 18021, 18001, 17982, 17963, 17944, 17924, 17905, 17886, 17867, 17848, 17829, 17810, 17791,
    17772, 17754, 17735, 17716, 17697, 17679, 17660, 17641, 17623, 17604, 17586, 17568, 17549, 17531, 17513, 17494,
    17476, 17458, 17440, 17422, 17404, 17386, 17368, 17350, 17332, 17314, 17296, 17278, 17261, 17243, 17225, 17208,
    17190, 17172, 17155, 17137, 17120, 17102, 17085, 17067, 17050, 17033, 17015, 16999, 16981, 16964, 16947, 16930,
    16913, 16895, 16878, 16862, 16845, 16828, 16810, 16794, 16777, 16760, 16743, 16727, 16710, 16693, 16677, 16660,
    16644, 16627, 16611, 16594, 16578, 16562, 16545, 16529, 16513, 16497, 16480, 16464, 16448, 16432, 16416, 16400,
    16384,
};

#define SCALED_DIFF_INTEGER_BITS 5
#define ACCUMULATION_INTEGER_BITS 12
#define MAX_S8_ROW 4095   /* keeps the int8 sum of exps (each <= 2^19) inside int32 */
#define MAX_S16_ROW 65535 /* each exp <= 32767 */

static int32_t clz32(uint32_t x)
{
    int32_t n = 0;
    while (n < 32 && !(x & 0x80000000u))
    {
        x <<= 1;
        ++n;
    }
    return n;
}

/* ---------------------------------------------------------------- prepare */

/* CalculateInputRadius(5, left_shift, 31), as a double floored to int. */
static int32_t input_radius(int32_t left_shift)
{
    const double max_rescaled = 1.0 * ((1 << SCALED_DIFF_INTEGER_BITS) - 1) *
                                (double)(1LL << (31 - SCALED_DIFF_INTEGER_BITS)) / (double)(1LL << left_shift);
    return (int32_t)floor(max_rescaled);
}

int32_t hct_ref_softmax_prepare(const HctSoftmaxQuant *in, HctSoftmaxParams *out)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    if (!(isfinite(in->beta) && in->beta > 0.0f && isfinite(in->input_scale) && in->input_scale > 0.0f &&
          isfinite(in->output_scale) && in->output_scale > 0.0f))
    {
        return HCT_E_PARAM;
    }
    if (in->input_dtype == HCT_INT16)
    {
        if (in->output_dtype != HCT_INT16)
        {
            return HCT_E_DTYPE;
        }
        if (in->input_zero_point != 0 || in->output_zero_point != 0 ||
            fabsf(in->output_scale - 1.0f / 32768) > 0.001f * 1.0f / 32768)
        {
            return HCT_E_PARAM;
        }
        const double rescale = (double)in->input_scale * (double)in->beta / (10.0 / 65535.0);
        HCT_TRY(hct_quantize_multiplier_impl(rescale, &out->input_multiplier, &out->input_left_shift));
        out->diff_min = 0;
        return HCT_OK;
    }
    if (in->input_dtype != HCT_INT8)
    {
        return HCT_E_DTYPE;
    }
    if (in->input_zero_point < INT8_MIN || in->input_zero_point > INT8_MAX)
    {
        return HCT_E_PARAM;
    }
    if (in->output_dtype == HCT_INT16)
    {
        if (in->output_zero_point != -32768 || fabsf(in->output_scale - 1.0f / 65536) > 0.001f * 1.0f / 65536)
        {
            return HCT_E_PARAM;
        }
    }
    else if (in->output_dtype == HCT_INT8)
    {
        if (in->output_zero_point != -128 || in->output_scale != 1.0f / 256)
        {
            return HCT_E_PARAM;
        }
    }
    else
    {
        return HCT_E_DTYPE;
    }
    /* PreprocessSoftmaxScaling, then QuantizeMultiplierGreaterThanOne. */
    double real = (double)in->beta * (double)in->input_scale * (double)(1LL << (31 - SCALED_DIFF_INTEGER_BITS));
    if (real > (double)(1LL << 31) - 1.0)
    {
        real = (double)(1LL << 31) - 1.0;
    }
    if (!(real > 1.0))
    {
        return HCT_E_PARAM;
    }
    HCT_TRY(hct_quantize_multiplier_impl(real, &out->input_multiplier, &out->input_left_shift));
    if (out->input_left_shift < 0)
    {
        return HCT_E_PARAM;
    }
    out->diff_min = -input_radius(out->input_left_shift);
    return HCT_OK;
}

/* ------------------------------------------------- gemmlowp fixed point */

static int32_t srdhm(int32_t a, int32_t b)
{
    return hct_saturating_rounding_doubling_high_mul(a, b);
}

/* SaturatingRoundingMultiplyByPOT<exponent> for exponent > 0. */
static int32_t sat_mul_pot(int32_t x, int32_t exponent)
{
    const int32_t threshold = (int32_t)((1LL << (31 - exponent)) - 1);
    if (x > threshold)
    {
        return INT32_MAX;
    }
    if (x < -threshold)
    {
        return INT32_MIN;
    }
    return (int32_t)((uint32_t)x << exponent);
}

/* gemmlowp exp_on_interval_between_negative_one_quarter_and_0_excl, Q0.31 in and out. */
static int32_t exp_on_interval(int32_t a)
{
    const int32_t constant_term = 1895147668;     /* exp(-1/8) */
    const int32_t constant_1_over_3 = 715827883; /* 1/3 */
    const int32_t x = a + (1 << 28);
    const int32_t x2 = srdhm(x, x);
    const int32_t x3 = srdhm(x2, x);
    const int32_t x4 = srdhm(x2, x2);
    const int32_t x4_over_4 = hct_rounding_divide_by_pot(x4, 2);
    const int32_t poly =
        hct_rounding_divide_by_pot(srdhm(x4_over_4 + x3, constant_1_over_3) + x2, 1);
    return constant_term + srdhm(constant_term, x + poly);
}

/* gemmlowp exp_on_negative_values for a Q5.26 input; the result is Q0.31. */
static int32_t exp_on_negative_values(int32_t a)
{
    static const int32_t multipliers[7] = {1672461947, 1302514674, 790015084, 290630308, 39332535, 720401, 242};
    const int32_t one_quarter = 1 << 24;
    const int32_t a_mod_quarter_minus_one_quarter = (a & (one_quarter - 1)) - one_quarter;
    /* Rescale<0> from 5 integer bits: a saturating x32, exact here as the value is in [-1/4, 0). */
    int32_t result = exp_on_interval(sat_mul_pot(a_mod_quarter_minus_one_quarter, SCALED_DIFF_INTEGER_BITS));
    const int32_t remainder = a_mod_quarter_minus_one_quarter - a;
    for (int32_t k = 0; k < 7; ++k)
    {
        if (remainder & (1 << (24 + k)))
        {
            result = srdhm(result, multipliers[k]);
        }
    }
    return a == 0 ? INT32_MAX : result;
}

/* gemmlowp one_over_one_plus_x_for_x_in_0_1 (Q0.31 in, Q0.31 out) by Newton-Raphson in Q2.29. */
static int32_t one_over_one_plus_x(int32_t a)
{
    const int64_t sum = (int64_t)a + INT32_MAX;
    const int32_t half_denominator = (int32_t)((sum + (sum >= 0 ? 1 : -1)) / 2);
    const int32_t constant_48_over_17 = 1515870810;
    const int32_t constant_neg_32_over_17 = -1010580540;
    int32_t x = constant_48_over_17 + srdhm(half_denominator, constant_neg_32_over_17);
    for (int i = 0; i < 3; ++i)
    {
        const int32_t half_denominator_times_x = srdhm(half_denominator, x);
        const int32_t one_minus = (1 << 29) - half_denominator_times_x;
        x = x + sat_mul_pot(srdhm(x, one_minus), 2);
    }
    return sat_mul_pot(x, 1);
}

/* RoundingDivideByPOT, extended past exponent 31 as the exact half-away-from-zero rounding. */
static int32_t rounding_divide_by_pot_wide(int32_t x, int32_t exponent)
{
    if (exponent <= 31)
    {
        return hct_rounding_divide_by_pot(x, exponent);
    }
    return x == INT32_MIN && exponent == 32 ? -1 : 0;
}

/* ----------------------------------------------------------- int8 input */

static int32_t check_s8_params(const HctSoftmaxParams *p)
{
    if (p->input_multiplier < 0 || p->input_left_shift < 0 || p->input_left_shift > 30 || p->diff_min > 0 ||
        p->diff_min < -input_radius(p->input_left_shift))
    {
        return HCT_E_PARAM;
    }
    return HCT_OK;
}

static int32_t row_layout(const HctTensor *t, int64_t count, int64_t *rows, int32_t *depth)
{
    if (t->rank < 1)
    {
        return HCT_E_SHAPE;
    }
    *depth = t->dims[t->rank - 1];
    *rows = *depth > 0 ? count / *depth : 0;
    return HCT_OK;
}

static int32_t softmax_s8(const HctSoftmaxParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                          int32_t num_outputs, int32_t out_dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, HCT_INT8, out_dtype, &count));
    HCT_TRY(check_s8_params(p));
    int64_t rows = 0;
    int32_t depth = 0;
    HCT_TRY(row_layout(&inputs[0], count, &rows, &depth));
    if (depth > MAX_S8_ROW)
    {
        return HCT_E_SHAPE;
    }
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(out_dtype, &qmin, &qmax));
    const int32_t out_bits = out_dtype == HCT_INT16 ? 16 : 8;
    for (int64_t r = 0; r < rows; ++r)
    {
        const int64_t base = r * depth;
        int32_t max_in_row = INT8_MIN;
        for (int32_t c = 0; c < depth; ++c)
        {
            const int32_t v = hct_load_i32(&inputs[0], base + c);
            max_in_row = v > max_in_row ? v : max_in_row;
        }
        int32_t sum_of_exps = 0;
        for (int32_t c = 0; c < depth; ++c)
        {
            const int32_t diff = hct_load_i32(&inputs[0], base + c) - max_in_row;
            if (diff >= p->diff_min)
            {
                const int32_t rescaled = srdhm(diff * (1 << p->input_left_shift), p->input_multiplier);
                sum_of_exps += hct_rounding_divide_by_pot(exp_on_negative_values(rescaled), ACCUMULATION_INTEGER_BITS);
            }
        }
        /* GetReciprocal: the max element alone contributes 2^19, so the sum is never 0. */
        const int32_t headroom_plus_one = clz32((uint32_t)sum_of_exps);
        const int32_t num_bits_over_unit = ACCUMULATION_INTEGER_BITS - headroom_plus_one;
        const int32_t shifted_sum_minus_one =
            (int32_t)(((uint32_t)sum_of_exps << headroom_plus_one) - ((uint32_t)1 << 31));
        const int32_t shifted_scale = one_over_one_plus_x(shifted_sum_minus_one);
        for (int32_t c = 0; c < depth; ++c)
        {
            const int32_t diff = hct_load_i32(&inputs[0], base + c) - max_in_row;
            int32_t v = qmin;
            if (diff >= p->diff_min)
            {
                const int32_t rescaled = srdhm(diff * (1 << p->input_left_shift), p->input_multiplier);
                const int32_t unsat = rounding_divide_by_pot_wide(srdhm(shifted_scale, exp_on_negative_values(rescaled)),
                                                                  num_bits_over_unit + 31 - out_bits);
                v = unsat + qmin;
                v = v < qmin ? qmin : (v > qmax ? qmax : v);
            }
            hct_store_i32(&outputs[0], base + c, v);
        }
    }
    return HCT_OK;
}

int32_t hct_ref_softmax_s8(const HctSoftmaxParams *params, const HctTensor *inputs, int32_t num_inputs,
                           HctTensor *outputs, int32_t num_outputs)
{
    return softmax_s8(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_softmax_s8_s16(const HctSoftmaxParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    return softmax_s8(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

/* ---------------------------------------------------------- int16 input */

static int32_t clamp16(int64_t v)
{
    return v < INT16_MIN ? INT16_MIN : (v > INT16_MAX ? INT16_MAX : (int32_t)v);
}

int32_t hct_ref_softmax_s16(const HctSoftmaxParams *params, const HctTensor *inputs, int32_t num_inputs,
                            HctTensor *outputs, int32_t num_outputs)
{
    const HctSoftmaxParams *p = params;
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, HCT_INT16, HCT_INT16, &count));
    if (p->input_multiplier < 0 || p->input_left_shift < -31 || p->input_left_shift > 30 || p->diff_min != 0)
    {
        return HCT_E_PARAM;
    }
    int64_t rows = 0;
    int32_t depth = 0;
    HCT_TRY(row_layout(&inputs[0], count, &rows, &depth));
    if (depth > MAX_S16_ROW)
    {
        return HCT_E_SHAPE;
    }
    for (int64_t r = 0; r < rows; ++r)
    {
        const int64_t base = r * depth;
        int32_t max_in_row = INT16_MIN;
        for (int32_t c = 0; c < depth; ++c)
        {
            const int32_t v = hct_load_i32(&inputs[0], base + c);
            max_in_row = v > max_in_row ? v : max_in_row;
        }
        /* The exps are cached in the output, as the reference does. */
        int32_t sum_of_exps = 0;
        for (int32_t c = 0; c < depth; ++c)
        {
            const int32_t diff = hct_load_i32(&inputs[0], base + c) - max_in_row;
            const int32_t scaled = hct_multiply_by_quantized_multiplier(diff, p->input_multiplier, p->input_left_shift);
            const int32_t e = hct_lut_lookup_s16(clamp16((int64_t)scaled + 32767), exp_lut);
            hct_store_i32(&outputs[0], base + c, e);
            sum_of_exps += e;
        }
        if (sum_of_exps <= 0)
        {
            return HCT_E_PARAM; /* only a corrupted LUT reading could get here */
        }
        const int32_t headroom_plus_one = clz32((uint32_t)sum_of_exps);
        const int32_t shifted_sum = (int32_t)((((int64_t)sum_of_exps << (headroom_plus_one - 1)) + (1 << 13)) >> 14);
        const int32_t sym_shifted_sum = shifted_sum + (-((1 << 15) + (1 << 16)));
        const int32_t reciprocal = hct_lut_lookup_s16(clamp16(sym_shifted_sum), one_over_one_plus_x_lut);
        const int32_t right_shift = 31 - headroom_plus_one;
        const int64_t round = (int64_t)1 << (right_shift - 1);
        for (int32_t c = 0; c < depth; ++c)
        {
            const int64_t e = hct_load_i32(&outputs[0], base + c);
            const int64_t v = (e * reciprocal + round) >> right_shift;
            hct_store_i32(&outputs[0], base + c, v < 0 ? 0 : (v > INT16_MAX ? INT16_MAX : (int32_t)v));
        }
    }
    return HCT_OK;
}

/* ---------------------------------------------------------------- float */

static int32_t softmax_float(const HctNoParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                             int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    int64_t rows = 0;
    int32_t depth = 0;
    HCT_TRY(row_layout(&inputs[0], count, &rows, &depth));
    for (int64_t r = 0; r < rows; ++r)
    {
        const int64_t base = r * depth;
        /* The row maximum propagates NaN, as numpy's does. */
        double max_in_row = -INFINITY;
        for (int32_t c = 0; c < depth; ++c)
        {
            const double v = (double)hct_load_f32(&inputs[0], base + c);
            if (isnan(v) || isnan(max_in_row))
            {
                max_in_row = NAN;
            }
            else if (v > max_in_row)
            {
                max_in_row = v;
            }
        }
        double sum = 0.0;
        for (int32_t c = 0; c < depth; ++c)
        {
            sum += exp((double)hct_load_f32(&inputs[0], base + c) - max_in_row);
        }
        for (int32_t c = 0; c < depth; ++c)
        {
            const double y = exp((double)hct_load_f32(&inputs[0], base + c) - max_in_row) / sum;
            if (dtype == HCT_FLOAT16)
            {
                ((uint16_t *)outputs[0].data)[base + c] = hct_f64_to_f16(y);
            }
            else
            {
                ((float *)outputs[0].data)[base + c] = (float)y;
            }
        }
    }
    return HCT_OK;
}

int32_t hct_ref_softmax_f32(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                            int32_t num_outputs)
{
    return softmax_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_softmax_f16(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                            int32_t num_outputs)
{
    return softmax_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
