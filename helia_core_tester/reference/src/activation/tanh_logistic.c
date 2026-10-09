/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * int16 Tanh and Logistic: TFLite's reference_integer_ops::Tanh/Logistic (a
 * 256-entry sigmoid table with linear interpolation) with TFLM's prepare.
 */
#include <math.h>
#include <stdlib.h>

#include "hct_ref_internal.h"

/* TFLite's sigmoid_table_uint16: 65536 * sigmoid(i * 32 / 768) over [0, 10.67) to
 * within about one count (tuned for the interpolation, not plain rounding), saturating at 65535. */
static const uint16_t sigmoid_table[256] = {
    32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498, 40149, 40794, 41432, 42064, 42688,
    43304, 43912, 44511, 45102, 45683, 46255, 46817, 47369, 47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409,
    51865, 52311, 52745, 53169, 53581, 53983, 54374, 54755, 55125, 55485, 55834, 56174, 56503, 56823, 57133, 57433,
    57724, 58007, 58280, 58544, 58800, 59048, 59288, 59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110,
    61279, 61441, 61599, 61750, 61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886, 62990, 63090, 63186,
    63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835, 63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308,
    64357, 64405, 64450, 64494, 64536, 64576, 64614, 64652, 64687, 64721, 64754, 64786, 64816, 64845, 64873, 64900,
    64926, 64950, 64974, 64997, 65019, 65039, 65060, 65079, 65097, 65115, 65132, 65149, 65164, 65179, 65194, 65208,
    65221, 65234, 65246, 65258, 65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360, 65367,
    65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425, 65429, 65433, 65438, 65442, 65445, 65449,
    65453, 65456, 65459, 65462, 65465, 65468, 65471, 65474, 65476, 65479, 65481, 65483, 65485, 65488, 65489, 65491,
    65493, 65495, 65497, 65498, 65500, 65501, 65503, 65504, 65505, 65507, 65508, 65509, 65510, 65511, 65512, 65513,
    65514, 65515, 65516, 65517, 65517, 65518, 65519, 65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524,
    65525, 65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529, 65529, 65529, 65530, 65530,
    65530, 65530, 65531, 65531, 65531, 65531, 65531, 65532, 65532, 65532, 65532, 65532, 65532, 65533, 65533, 65533,
    65533, 65533, 65533, 65533, 65533, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65535,
};

/* CheckedLog2 in float, as TFLite computes it. */
static int checked_log2(float x, int32_t *log2_result)
{
    const float x_log2 = logf(x) * (1.0f / logf(2.0f));
    const float rounded = roundf(x_log2);
    *log2_result = (int32_t)rounded;
    return fabsf(x_log2 - rounded) < 1e-3f;
}

/* TanhPrepare / CalculateArithmeticOpData for int16. A power-of-two input scale
 * whose shift lands in pot_max_shift or below needs no multiplier. */
static int32_t prepare(const HctTanhLogisticQuant *in, HctTanhLogisticParams *out, int32_t pot_max_shift)
{
    if (in == NULL || out == NULL)
    {
        return HCT_E_NULL;
    }
    if (!(isfinite(in->input_scale) && in->input_scale > 0.0f && isfinite(in->output_scale) &&
          in->output_scale > 0.0f) ||
        in->input_zero_point != 0 || in->output_zero_point != 0)
    {
        return HCT_E_PARAM;
    }
    int32_t log2_in = 0;
    const int pot = checked_log2(in->input_scale, &log2_in);
    const int32_t pot_shift = 12 + log2_in;
    if (pot && pot_shift >= 0 && pot_shift <= pot_max_shift)
    {
        out->input_multiplier = 0;
        out->input_left_shift = pot_shift;
    }
    else
    {
        double multiplier = (double)in->input_scale * 4096.0 * 3.0;
        int32_t shift = 0;
        while (multiplier <= 32767.0 / 2.0 && shift <= 30)
        {
            ++shift;
            multiplier *= 2.0;
        }
        if (multiplier > 32767.0)
        {
            return HCT_E_PARAM;
        }
        out->input_multiplier = (int32_t)multiplier;
        out->input_left_shift = shift;
    }
    int32_t log2_out = 0;
    if (!checked_log2(in->output_scale, &log2_out) || log2_out != -15)
    {
        return HCT_E_PARAM;
    }
    return HCT_OK;
}

int32_t hct_ref_tanh_prepare(const HctTanhLogisticQuant *in, HctTanhLogisticParams *out)
{
    return prepare(in, out, 1);
}

int32_t hct_ref_logistic_prepare(const HctTanhLogisticQuant *in, HctTanhLogisticParams *out)
{
    return prepare(in, out, 0);
}

/* The multiplier 0 form becomes 3 << shift with no shift, as the kernels do. */
static int32_t effective(const HctTanhLogisticParams *p, int32_t *multiplier, int32_t *shift)
{
    if (p->input_multiplier < 0 || p->input_multiplier > INT16_MAX || p->input_left_shift < 0 ||
        p->input_left_shift > 31)
    {
        return HCT_E_PARAM;
    }
    if (p->input_multiplier == 0)
    {
        if (p->input_left_shift > 13)
        {
            return HCT_E_PARAM;
        }
        *multiplier = 3 << p->input_left_shift;
        *shift = 0;
    }
    else
    {
        *multiplier = p->input_multiplier;
        *shift = p->input_left_shift;
    }
    return HCT_OK;
}

static int32_t run(const HctTanhLogisticParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                   int32_t num_outputs, int is_tanh)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, HCT_INT16, HCT_INT16, &count));
    int32_t multiplier = 0;
    int32_t shift = 0;
    HCT_TRY(effective(p, &multiplier, &shift));
    const int32_t round = shift > 0 ? (int32_t)(1u << (shift - 1)) : 0;
    /* tanh(x) = 2 sigmoid(2x) - 1: tanh reads the table at twice logistic's rate. */
    const uint32_t frac_bits = is_tanh ? 8u : 9u;
    for (int64_t i = 0; i < count; ++i)
    {
        /* |x * multiplier| + round < 2^31 for int16 x and multiplier <= 32767. */
        const int32_t x = (hct_load_i32(&inputs[0], i) * multiplier + round) >> shift;
        const uint32_t ax = (uint32_t)abs(x);
        const uint32_t uh = ax >> frac_bits;
        int32_t v = 0;
        if (is_tanh)
        {
            int32_t result = 0xFFFF << 8;
            if (uh < 255)
            {
                const uint32_t ua = sigmoid_table[uh];
                const uint32_t ub = sigmoid_table[uh + 1];
                result = (int32_t)((ua << 8) + (ax & 0xFFu) * (ub - ua));
            }
            result = x >= 0 ? result - (1 << 23) + (1 << 7) : -result + (1 << 23) + (1 << 7) - 1;
            v = result >> 8;
        }
        else
        {
            uint32_t result = 0x7FFFu << 10;
            if (uh < 255)
            {
                const uint32_t ua = sigmoid_table[uh];
                const uint32_t ub = sigmoid_table[uh + 1];
                result = (ua << 9) + (ax & 0x1FFu) * (ub - ua);
            }
            result = x >= 0 ? result + (1u << 9) : (1u << 25) - result + (1u << 9) - 1u;
            v = (int32_t)(result >> 10);
        }
        hct_store_i32(&outputs[0], i, v);
    }
    return HCT_OK;
}

int32_t hct_ref_tanh_s16(const HctTanhLogisticParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return run(params, inputs, num_inputs, outputs, num_outputs, 1);
}

int32_t hct_ref_logistic_s16(const HctTanhLogisticParams *params, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    return run(params, inputs, num_inputs, outputs, num_outputs, 0);
}
