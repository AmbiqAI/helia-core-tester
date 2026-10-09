/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Activation entries of the hct_ref shim (see hct_ref.h): each *_prepare
 * mirrors the TFLM micro prepare function, each eval calls the reference
 * kernel TFLM's micro eval dispatches to.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "hct_ref.h"
#include "hct_ref_internal.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/hard_swish.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"
#include "tensorflow/lite/kernels/internal/reference/leaky_relu.h"
#include "tensorflow/lite/kernels/internal/reference/prelu.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace {

using namespace hct;

constexpr int kInt16LutSize = 513;

bool valid_scale(float scale)
{
    return std::isfinite(scale) && scale > 0.0f;
}

template <typename T> bool valid_zero_point(int32_t zp)
{
    return zp >= std::numeric_limits<T>::min() && zp <= std::numeric_limits<T>::max();
}

int32_t check_unary(const HctShape *shape, const void *input, const void *output)
{
    HCT_TRY(check_shape(shape, 0));
    return check_buffers(input, output);
}

// TanhPrepare / LogisticPrepare, int16 path (micro/kernels/{tanh,logistic_common}.cc).
int32_t lut_act_prepare_s16(bool logistic, float input_scale, float output_scale, HctLutActParams *out)
{
    static constexpr int kInputIntegerBits = 3;
    static constexpr int kOutputFractionalBits = 15;
    int input_log2 = 0;
    bool pot = tflite::CheckedLog2(input_scale, &input_log2);
    int32_t left_shift = (15 - kInputIntegerBits) + input_log2;
    pot &= logistic ? (left_shift == 0) : (left_shift == 0 || left_shift == 1);
    int32_t multiplier = 0;
    if (!pot)
    {
        double m = static_cast<double>(input_scale) * 4096.0 * 3.0;
        left_shift = 0;
        while (m <= 32767.0 / 2.0 && left_shift <= 30)
        {
            ++left_shift;
            m *= 2.0;
        }
        multiplier = static_cast<int32_t>(m);
    }
    int output_log2 = 0;
    if (multiplier > 32767 || !tflite::CheckedLog2(output_scale, &output_log2) ||
        output_log2 != -kOutputFractionalBits)
    {
        return HCT_REF_E_PARAM;
    }
    out->input_zero_point = 0;
    out->input_range_radius = 0;
    out->input_multiplier = multiplier;
    out->input_left_shift = left_shift;
    return HCT_REF_OK;
}

template <typename T> int32_t check_leaky_relu(const HctLeakyReluParams *p)
{
    if (!valid_zero_point<T>(p->input_zero_point) || !valid_zero_point<T>(p->output_zero_point))
    {
        return HCT_REF_E_PARAM;
    }
    if (sizeof(T) == 2 && (p->input_zero_point != 0 || p->output_zero_point != 0))
    {
        return HCT_REF_E_PARAM;
    }
    return HCT_REF_OK;
}

template <typename T>
int32_t leaky_relu(const HctLeakyReluParams *params, const HctShape *shape, const T *input, T *output)
{
    if (params == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_unary(shape, input, output));
    HCT_TRY(check_leaky_relu<T>(params));
    tflite::LeakyReluParams p = {};
    p.input_offset = params->input_zero_point;
    p.output_offset = params->output_zero_point;
    p.output_multiplier_alpha = params->multiplier_alpha;
    p.output_shift_alpha = params->shift_alpha;
    p.output_multiplier_identity = params->multiplier_identity;
    p.output_shift_identity = params->shift_identity;
    const RuntimeShape s = to_runtime(shape);
    tflite::reference_ops::QuantizeLeakyRelu(p, s, input, s, output);
    return HCT_REF_OK;
}

template <typename T>
int32_t relu(const HctReluParams *params, const HctShape *shape, const T *input, T *output)
{
    if (params == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_unary(shape, input, output));
    if (!valid_zero_point<T>(params->input_zero_point) || !valid_zero_point<T>(params->output_zero_point) ||
        !valid_zero_point<T>(params->act_min) || !valid_zero_point<T>(params->act_max) ||
        params->act_min > params->act_max)
    {
        return HCT_REF_E_PARAM;
    }
    // ReluQuantized (micro/kernels/activations_common.cc).
    const int64_t n = flat_size(shape);
    for (int64_t i = 0; i < n; ++i)
    {
        const int32_t val = static_cast<int32_t>(input[i]);
        int32_t clamped =
            params->output_zero_point + tflite::MultiplyByQuantizedMultiplier(val - params->input_zero_point,
                                                                              params->output_multiplier,
                                                                              params->output_shift);
        clamped = std::max(params->act_min, clamped);
        clamped = std::min(params->act_max, clamped);
        output[i] = static_cast<T>(clamped);
    }
    return HCT_REF_OK;
}

template <typename T>
int32_t check_rsqrt(const HctRsqrtParams *params, const HctShape *shape, const T *input, T *output)
{
    if (params == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_unary(shape, input, output));
    if (!valid_zero_point<T>(params->input_zero_point) || !valid_zero_point<T>(params->output_zero_point) ||
        (sizeof(T) == 2 && (params->input_zero_point != 0 || params->output_zero_point != 0)))
    {
        return HCT_REF_E_PARAM;
    }
    // Rsqrt is only defined at or above the input zero point.
    const int64_t n = flat_size(shape);
    for (int64_t i = 0; i < n; ++i)
    {
        if (static_cast<int32_t>(input[i]) < params->input_zero_point)
        {
            return HCT_REF_E_PARAM;
        }
    }
    return HCT_REF_OK;
}

} // namespace

extern "C" {

int32_t hct_ref_relu_prepare(float input_scale,
                             int32_t input_zero_point,
                             float output_scale,
                             int32_t output_zero_point,
                             float act_min_real,
                             float act_max_real,
                             int32_t qmin,
                             int32_t qmax,
                             HctReluParams *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (!valid_scale(input_scale) || !valid_scale(output_scale) || !std::isfinite(act_min_real) ||
        std::isnan(act_max_real) || act_min_real > act_max_real || qmin > qmax ||
        (!(qmin == -128 && qmax == 127) && !(qmin == -32768 && qmax == 32767)))
    {
        return HCT_REF_E_PARAM;
    }
    // CalculateReluOpData: float32 ratio; act bounds through roundf.
    const double real_multiplier = static_cast<double>(input_scale / output_scale);
    int shift = 0;
    tflite::QuantizeMultiplier(real_multiplier, &out->output_multiplier, &shift);
    out->output_shift = shift;
    out->act_min = std::max(qmin, output_zero_point + static_cast<int32_t>(std::round(act_min_real / output_scale)));
    out->act_max = std::isinf(act_max_real)
                       ? qmax
                       : std::min(qmax, output_zero_point + static_cast<int32_t>(std::round(act_max_real / output_scale)));
    out->input_zero_point = input_zero_point;
    out->output_zero_point = output_zero_point;
    return HCT_REF_OK;
}

int32_t hct_ref_relu_s8(const HctReluParams *params, const HctShape *shape, const int8_t *input, int8_t *output)
{
    return relu<int8_t>(params, shape, input, output);
}

int32_t hct_ref_relu_s16(const HctReluParams *params, const HctShape *shape, const int16_t *input, int16_t *output)
{
    return relu<int16_t>(params, shape, input, output);
}

int32_t hct_ref_rsqrt_prepare(float input_scale,
                              int32_t input_zero_point,
                              float output_scale,
                              int32_t output_zero_point,
                              HctRsqrtParams *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (!valid_scale(input_scale) || !valid_scale(output_scale))
    {
        return HCT_REF_E_PARAM;
    }
    // SetRsqrtOutputMultiplier: sqrt and product in float32, reciprocal in double.
    const double scale = 1. / static_cast<double>((std::sqrt(input_scale) * output_scale));
    int shift = 0;
    tflite::QuantizeMultiplier(scale, &out->multiplier, &shift);
    out->shift = shift;
    out->input_zero_point = input_zero_point;
    out->output_zero_point = output_zero_point;
    return HCT_REF_OK;
}

int32_t hct_ref_rsqrt_s8(const HctRsqrtParams *params, const HctShape *shape, const int8_t *input, int8_t *output)
{
    HCT_TRY(check_rsqrt<int8_t>(params, shape, input, output));
    // RsqrtEvalQuantizedInt8 (TFLite and TFLM micro elementwise.cc).
    const int32_t k_shift = 20;
    const int64_t n = flat_size(shape);
    for (int64_t i = 0; i < n; ++i)
    {
        const int32_t value = static_cast<int32_t>(input[i]) - params->input_zero_point;
        if (value == 0)
        {
            output[i] = std::numeric_limits<int8_t>::max();
            continue;
        }
        int32_t inv_sqrt_multiplier = 0;
        int inv_sqrt_shift = 0;
        tflite::GetInvSqrtQuantizedMultiplierExp(value, tflite::kReverseShift, &inv_sqrt_multiplier, &inv_sqrt_shift);
        const int32_t data = tflite::MultiplyByQuantizedMultiplier(static_cast<int32_t>(1), inv_sqrt_multiplier,
                                                                   inv_sqrt_shift + k_shift);
        const int32_t out =
            tflite::MultiplyByQuantizedMultiplier(data, params->multiplier, params->shift - k_shift) +
            params->output_zero_point;
        output[i] = static_cast<int8_t>(std::min<int32_t>(std::max<int32_t>(out, -128), 127));
    }
    return HCT_REF_OK;
}

int32_t hct_ref_rsqrt_s16(const HctRsqrtParams *params,
                          float input_scale,
                          float output_scale,
                          const HctShape *shape,
                          const int16_t *input,
                          int16_t *output)
{
    HCT_TRY(check_rsqrt<int16_t>(params, shape, input, output));
    if (!valid_scale(input_scale) || !valid_scale(output_scale))
    {
        return HCT_REF_E_PARAM;
    }
    // TFLite's int16 RSQRT (lite/kernels/elementwise.cc): a LUTPopulate<int16_t> table of
    // 1/sqrt, saturating at and below zero, read through the interpolating LUTLookup.
    int16_t lut[kInt16LutSize];
    tflite::LUTPopulate<int16_t>(
        input_scale, params->input_zero_point, output_scale, params->output_zero_point,
        [](float value, const void *transform_params) {
            if (value <= 0.0f)
            {
                return std::numeric_limits<int16_t>::max() * *static_cast<const float *>(transform_params);
            }
            return 1.0f / std::sqrt(value);
        },
        &output_scale, lut);
    const int64_t n = flat_size(shape);
    for (int64_t i = 0; i < n; ++i)
    {
        output[i] = tflite::LUTLookup(input[i], lut);
    }
    return HCT_REF_OK;
}

int32_t hct_ref_softmax_s8(const HctSoftmaxParams *params, const HctShape *shape, const int8_t *input, int8_t *output)
{
    if (params == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_unary(shape, input, output));
    if (params->input_multiplier <= 0 || params->diff_min > 0)
    {
        return HCT_REF_E_PARAM;
    }
    tflite::SoftmaxParams p = {};
    p.input_multiplier = params->input_multiplier;
    p.input_left_shift = params->input_left_shift;
    p.diff_min = params->diff_min;
    const RuntimeShape s = to_runtime(shape);
    tflite::reference_ops::Softmax(p, s, input, s, output);
    return HCT_REF_OK;
}

int32_t hct_ref_softmax_s16(const HctSoftmaxParams *params, const HctShape *shape, const int16_t *input, int16_t *output)
{
    if (params == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_unary(shape, input, output));
    if (params->input_multiplier <= 0)
    {
        return HCT_REF_E_PARAM;
    }
    // InitializeLutForInt16 (micro/kernels/softmax_common.cc).
    int16_t exp_lut[kInt16LutSize];
    int16_t one_over_one_plus_x_lut[kInt16LutSize];
    const int32_t range = std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::min();
    tflite::LUTPopulate<int16_t>(
        10.0f / range, std::numeric_limits<int16_t>::max(), 2.0f / range, 0,
        [](float value) { return std::exp(value); }, exp_lut);
    tflite::LUTPopulate<int16_t>(
        1.0f / range, std::numeric_limits<int16_t>::min(), 2.0f / range, 0,
        [](float value) { return 1.0f / (1.0f + value); }, one_over_one_plus_x_lut);
    tflite::SoftmaxParams p = {};
    p.input_multiplier = params->input_multiplier;
    p.input_left_shift = params->input_left_shift;
    p.exp_lut = exp_lut;
    p.one_over_one_plus_x_lut = one_over_one_plus_x_lut;
    const RuntimeShape s = to_runtime(shape);
    tflite::reference_ops::SoftmaxInt16(p, s, input, s, output);
    return HCT_REF_OK;
}

int32_t hct_ref_tanh_logistic_s16_prepare(int32_t logistic, float input_scale, float output_scale, HctLutActParams *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if ((logistic != 0 && logistic != 1) || !valid_scale(input_scale) || !valid_scale(output_scale))
    {
        return HCT_REF_E_PARAM;
    }
    return lut_act_prepare_s16(logistic == 1, input_scale, output_scale, out);
}

int32_t hct_ref_tanh_s16(const HctLutActParams *params, const HctShape *shape, const int16_t *input, int16_t *output)
{
    if (params == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_unary(shape, input, output));
    if (params->input_multiplier < 0 || params->input_multiplier > 32767 || params->input_left_shift < 0 ||
        params->input_left_shift > 31)
    {
        return HCT_REF_E_PARAM;
    }
    const RuntimeShape s = to_runtime(shape);
    tflite::reference_integer_ops::Tanh(params->input_multiplier, params->input_left_shift, s, input, s, output);
    return HCT_REF_OK;
}

int32_t hct_ref_logistic_s16(const HctLutActParams *params,
                             const HctShape *shape,
                             const int16_t *input,
                             int16_t *output)
{
    if (params == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_unary(shape, input, output));
    if (params->input_multiplier < 0 || params->input_multiplier > 32767 || params->input_left_shift < 0 ||
        params->input_left_shift > 31)
    {
        return HCT_REF_E_PARAM;
    }
    tflite::reference_integer_ops::Logistic(params->input_multiplier, params->input_left_shift,
                                            static_cast<int32_t>(flat_size(shape)), input, output);
    return HCT_REF_OK;
}

int32_t hct_ref_leaky_relu_prepare(float input_scale,
                                   int32_t input_zero_point,
                                   float alpha,
                                   float output_scale,
                                   int32_t output_zero_point,
                                   HctLeakyReluParams *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (!valid_scale(input_scale) || !valid_scale(output_scale) || !std::isfinite(alpha))
    {
        return HCT_REF_E_PARAM;
    }
    // CalculateOpDataLeakyRelu: both ratios formed in float32, then widened.
    const double alpha_multiplier = static_cast<double>(input_scale * alpha / output_scale);
    const double identity_multiplier = static_cast<double>(input_scale / output_scale);
    int shift_alpha = 0;
    int shift_identity = 0;
    tflite::QuantizeMultiplier(alpha_multiplier, &out->multiplier_alpha, &shift_alpha);
    tflite::QuantizeMultiplier(identity_multiplier, &out->multiplier_identity, &shift_identity);
    out->shift_alpha = shift_alpha;
    out->shift_identity = shift_identity;
    out->input_zero_point = input_zero_point;
    out->output_zero_point = output_zero_point;
    return HCT_REF_OK;
}

int32_t hct_ref_leaky_relu_s8(const HctLeakyReluParams *params, const HctShape *shape, const int8_t *input, int8_t *output)
{
    return leaky_relu<int8_t>(params, shape, input, output);
}

int32_t hct_ref_leaky_relu_s16(const HctLeakyReluParams *params,
                               const HctShape *shape,
                               const int16_t *input,
                               int16_t *output)
{
    return leaky_relu<int16_t>(params, shape, input, output);
}

int32_t hct_ref_prelu_prepare(float input_scale,
                              int32_t input_zero_point,
                              float alpha_scale,
                              int32_t alpha_zero_point,
                              float output_scale,
                              int32_t output_zero_point,
                              HctPreluParams *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (!valid_scale(input_scale) || !valid_scale(alpha_scale) || !valid_scale(output_scale))
    {
        return HCT_REF_E_PARAM;
    }
    // CalculatePreluParams: double arithmetic over the float32 scales.
    const double m1 = static_cast<double>(input_scale) / static_cast<double>(output_scale);
    const double m2 =
        static_cast<double>(input_scale) * static_cast<double>(alpha_scale) / static_cast<double>(output_scale);
    int shift_1 = 0;
    int shift_2 = 0;
    tflite::QuantizeMultiplier(m1, &out->multiplier_1, &shift_1);
    tflite::QuantizeMultiplier(m2, &out->multiplier_2, &shift_2);
    out->shift_1 = shift_1;
    out->shift_2 = shift_2;
    out->input_offset = -input_zero_point;
    out->alpha_offset = -alpha_zero_point;
    out->output_offset = output_zero_point;
    return HCT_REF_OK;
}

int32_t hct_ref_prelu_s8(const HctPreluParams *params,
                         const HctShape *input_shape,
                         const int8_t *input,
                         const HctShape *alpha_shape,
                         const int8_t *alpha,
                         const HctShape *output_shape,
                         int8_t *output)
{
    if (params == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_shape(input_shape, 0));
    HCT_TRY(check_shape(alpha_shape, 0));
    HCT_TRY(check_shape(output_shape, 0));
    HCT_TRY(check_buffers(input, alpha, output));
    if (input_shape->rank > 4 || alpha_shape->rank > 4 || output_shape->rank != input_shape->rank)
    {
        return HCT_REF_E_UNSUPPORTED;
    }
    for (int32_t i = 0; i < input_shape->rank; ++i)
    {
        if (output_shape->dims[i] != input_shape->dims[i])
        {
            return HCT_REF_E_DIMS;
        }
    }
    // Alpha broadcasts onto the input (right-aligned, each dimension equal or 1).
    if (alpha_shape->rank > input_shape->rank)
    {
        return HCT_REF_E_DIMS;
    }
    for (int32_t i = 0; i < alpha_shape->rank; ++i)
    {
        const int32_t a = alpha_shape->dims[alpha_shape->rank - 1 - i];
        if (a != 1 && a != input_shape->dims[input_shape->rank - 1 - i])
        {
            return HCT_REF_E_DIMS;
        }
    }
    if (params->input_offset < -127 || params->input_offset > 128 || params->alpha_offset < -127 ||
        params->alpha_offset > 128 || !valid_zero_point<int8_t>(params->output_offset))
    {
        return HCT_REF_E_PARAM;
    }
    tflite::PreluParams p = {};
    p.input_offset = params->input_offset;
    p.alpha_offset = params->alpha_offset;
    p.output_offset = params->output_offset;
    p.output_multiplier_1 = params->multiplier_1;
    p.output_shift_1 = params->shift_1;
    p.output_multiplier_2 = params->multiplier_2;
    p.output_shift_2 = params->shift_2;
    tflite::reference_ops::BroadcastPrelu4DSlow(p, to_runtime(input_shape), input, to_runtime(alpha_shape), alpha,
                                                to_runtime(output_shape), output);
    return HCT_REF_OK;
}

int32_t hct_ref_hard_swish_prepare(float input_scale,
                                   int32_t input_zero_point,
                                   float output_scale,
                                   int32_t output_zero_point,
                                   HctHardSwishParams *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (!valid_scale(input_scale) || !valid_scale(output_scale) || !valid_zero_point<int8_t>(input_zero_point) ||
        !valid_zero_point<int8_t>(output_zero_point))
    {
        return HCT_REF_E_PARAM;
    }
    // HardSwishPrepare (micro/kernels/hard_swish_common.cc), int8.
    const float hires_input_scale = (1.0f / 128.0f) * input_scale;
    const float reluish_scale = 3.0f / 32768.0f;
    int32_t output_q31 = 0;
    int32_t reluish_q31 = 0;
    int output_exponent = 0;
    int reluish_exponent = 0;
    tflite::QuantizeMultiplier(static_cast<double>(hires_input_scale / output_scale), &output_q31, &output_exponent);
    tflite::QuantizeMultiplier(static_cast<double>(hires_input_scale / reluish_scale), &reluish_q31, &reluish_exponent);
    if (output_exponent > 0)
    {
        return HCT_REF_E_PARAM;
    }
    int16_t output_q15 = 0;
    int16_t reluish_q15 = 0;
    tflite::DownScaleInt32ToInt16Multiplier(output_q31, &output_q15);
    tflite::DownScaleInt32ToInt16Multiplier(reluish_q31, &reluish_q15);
    out->input_zero_point = input_zero_point;
    out->output_zero_point = output_zero_point;
    out->reluish_multiplier_fixedpoint_int16 = reluish_q15;
    out->reluish_multiplier_exponent = reluish_exponent;
    out->output_multiplier_fixedpoint_int16 = output_q15;
    out->output_multiplier_exponent = output_exponent;
    return HCT_REF_OK;
}

int32_t hct_ref_hard_swish_s8(const HctHardSwishParams *params, const HctShape *shape, const int8_t *input, int8_t *output)
{
    if (params == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_unary(shape, input, output));
    if (!valid_zero_point<int8_t>(params->input_zero_point) || !valid_zero_point<int8_t>(params->output_zero_point) ||
        params->output_multiplier_exponent > 0 || params->reluish_multiplier_fixedpoint_int16 < 0 ||
        params->reluish_multiplier_fixedpoint_int16 > 32767 || params->output_multiplier_fixedpoint_int16 < 0 ||
        params->output_multiplier_fixedpoint_int16 > 32767)
    {
        return HCT_REF_E_PARAM;
    }
    tflite::HardSwishParams p = {};
    p.input_zero_point = static_cast<int16_t>(params->input_zero_point);
    p.output_zero_point = static_cast<int16_t>(params->output_zero_point);
    p.reluish_multiplier_fixedpoint_int16 = static_cast<int16_t>(params->reluish_multiplier_fixedpoint_int16);
    p.reluish_multiplier_exponent = params->reluish_multiplier_exponent;
    p.output_multiplier_fixedpoint_int16 = static_cast<int16_t>(params->output_multiplier_fixedpoint_int16);
    p.output_multiplier_exponent = params->output_multiplier_exponent;
    const RuntimeShape s = to_runtime(shape);
    tflite::reference_ops::HardSwish(p, s, input, s, output);
    return HCT_REF_OK;
}

} // extern "C"
