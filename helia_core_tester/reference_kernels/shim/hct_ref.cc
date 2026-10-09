/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Thin C ABI over the vendored TFLM reference kernels. See hct_ref.h.
 */
#include "hct_ref.h"
#include "hct_ref_internal.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/transpose_conv.h"
#include "tensorflow/lite/kernels/internal/reference/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/transpose_conv.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace {

using namespace hct;

// The int64 (16-bit activation) MultiplyByQuantizedMultiplier asserts a shift
// below 8; everything else accepts any shift an int32 rescale can use.
template <typename In> constexpr int32_t max_shift()
{
    return sizeof(In) == 2 ? 7 : 31;
}

int32_t check_quant(const HctPerChannelQuant *quant, int32_t channels, int32_t shift_max = 31)
{
    if (quant == nullptr || quant->multiplier == nullptr || quant->shift == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (quant->count != channels)
    {
        return HCT_REF_E_PARAM;
    }
    for (int32_t c = 0; c < channels; ++c)
    {
        // Q0.31 multipliers are non-negative; shifts beyond +-31 are
        // meaningless for an int32 rescale.
        if (quant->multiplier[c] < 0 || quant->shift[c] < -31 || quant->shift[c] > shift_max)
        {
            return HCT_REF_E_PARAM;
        }
    }
    return HCT_REF_OK;
}

template <typename B> int32_t check_bias(const B *bias, int32_t bias_len, int32_t channels)
{
    if (bias == nullptr)
    {
        return bias_len == 0 ? HCT_REF_OK : HCT_REF_E_NULL;
    }
    return bias_len == channels ? HCT_REF_OK : HCT_REF_E_PARAM;
}

int32_t check_window(int32_t in,
                     int32_t out,
                     int32_t filter,
                     int32_t stride,
                     int32_t dilation,
                     int32_t pad,
                     int32_t pad_offset)
{
    if (stride < 1 || dilation < 1 || pad < 0 || pad_offset < 0 || pad_offset > 1 || filter < 1)
    {
        return HCT_REF_E_PARAM;
    }
    const int64_t effective = int64_t{dilation} * (filter - 1) + 1;
    const int64_t padded = int64_t{in} + 2 * int64_t{pad} + pad_offset;
    if (padded < effective)
    {
        return HCT_REF_E_DIMS;
    }
    return ((padded - effective) / stride + 1 == out) ? HCT_REF_OK : HCT_REF_E_DIMS;
}

tflite::PaddingValues padding(int32_t pad_h, int32_t pad_w, int32_t pad_h_offset, int32_t pad_w_offset)
{
    tflite::PaddingValues values;
    values.height = static_cast<int16_t>(pad_h);
    values.width = static_cast<int16_t>(pad_w);
    values.height_offset = static_cast<int16_t>(pad_h_offset);
    values.width_offset = static_cast<int16_t>(pad_w_offset);
    return values;
}

int32_t check_int16_range(int32_t value)
{
    return (value >= 0 && value <= std::numeric_limits<int16_t>::max()) ? HCT_REF_OK : HCT_REF_E_PARAM;
}

tflite::ConvParams to_conv_params(const HctConvParams *p)
{
    tflite::ConvParams params = {};
    params.padding_type = tflite::PaddingType::kSame;
    params.padding_values = padding(p->pad_h, p->pad_w, p->pad_h_offset, p->pad_w_offset);
    params.stride_height = static_cast<int16_t>(p->stride_h);
    params.stride_width = static_cast<int16_t>(p->stride_w);
    params.dilation_height_factor = static_cast<int16_t>(p->dilation_h);
    params.dilation_width_factor = static_cast<int16_t>(p->dilation_w);
    params.input_offset = p->input_offset;
    params.weights_offset = 0;
    params.output_offset = p->output_offset;
    params.quantized_activation_min = p->act.min;
    params.quantized_activation_max = p->act.max;
    params.float_activation_min = p->act.fmin;
    params.float_activation_max = p->act.fmax;
    return params;
}

int32_t check_conv_common(const HctConvParams *p,
                          const HctShape *in,
                          const HctShape *filter,
                          const HctShape *out,
                          int32_t filter_in_channels_divides)
{
    if (p == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_shape(in, 4));
    HCT_TRY(check_shape(filter, 4));
    HCT_TRY(check_shape(out, 4));
    HCT_TRY(check_int16_range(p->stride_h));
    HCT_TRY(check_int16_range(p->stride_w));
    HCT_TRY(check_int16_range(p->dilation_h));
    HCT_TRY(check_int16_range(p->dilation_w));
    HCT_TRY(check_int16_range(p->pad_h));
    HCT_TRY(check_int16_range(p->pad_w));
    if (in->dims[0] != out->dims[0])
    {
        return HCT_REF_E_DIMS;
    }
    if (filter_in_channels_divides && in->dims[3] % filter->dims[3] != 0)
    {
        return HCT_REF_E_DIMS;
    }
    HCT_TRY(check_window(in->dims[1], out->dims[1], filter->dims[1], p->stride_h, p->dilation_h, p->pad_h,
                         p->pad_h_offset));
    HCT_TRY(check_window(in->dims[2], out->dims[2], filter->dims[2], p->stride_w, p->dilation_w, p->pad_w,
                         p->pad_w_offset));
    return HCT_REF_OK;
}

int32_t check_conv(const HctConvParams *p, const HctShape *in, const HctShape *filter, const HctShape *out)
{
    HCT_TRY(check_conv_common(p, in, filter, out, 1));
    if (filter->dims[0] != out->dims[3])
    {
        return HCT_REF_E_DIMS;
    }
    // Grouped conv: each group's output channels must divide evenly.
    const int32_t groups = in->dims[3] / filter->dims[3];
    return (out->dims[3] % groups == 0) ? HCT_REF_OK : HCT_REF_E_DIMS;
}

int32_t check_dwconv(const HctDwConvParams *p, const HctShape *in, const HctShape *filter, const HctShape *out)
{
    if (p == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_conv_common(&p->conv, in, filter, out, 0));
    if (p->depth_multiplier < 1 || p->depth_multiplier > std::numeric_limits<int16_t>::max())
    {
        return HCT_REF_E_PARAM;
    }
    if (filter->dims[0] != 1 || filter->dims[3] != out->dims[3] ||
        int64_t{in->dims[3]} * p->depth_multiplier != out->dims[3])
    {
        return HCT_REF_E_DIMS;
    }
    return HCT_REF_OK;
}

tflite::DepthwiseParams to_dw_params(const HctDwConvParams *p)
{
    const HctConvParams *c = &p->conv;
    tflite::DepthwiseParams params = {};
    params.padding_type = tflite::PaddingType::kSame;
    params.padding_values = padding(c->pad_h, c->pad_w, c->pad_h_offset, c->pad_w_offset);
    params.stride_height = static_cast<int16_t>(c->stride_h);
    params.stride_width = static_cast<int16_t>(c->stride_w);
    params.dilation_height_factor = static_cast<int16_t>(c->dilation_h);
    params.dilation_width_factor = static_cast<int16_t>(c->dilation_w);
    params.depth_multiplier = static_cast<int16_t>(p->depth_multiplier);
    params.input_offset = c->input_offset;
    params.weights_offset = 0;
    params.output_offset = c->output_offset;
    params.quantized_activation_min = c->act.min;
    params.quantized_activation_max = c->act.max;
    params.float_activation_min = c->act.fmin;
    params.float_activation_max = c->act.fmax;
    return params;
}

int32_t check_tconv(const HctConvParams *p, const HctShape *in, const HctShape *filter, const HctShape *out)
{
    if (p == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_shape(in, 4));
    HCT_TRY(check_shape(filter, 4));
    HCT_TRY(check_shape(out, 4));
    HCT_TRY(check_int16_range(p->pad_h));
    HCT_TRY(check_int16_range(p->pad_w));
    if (p->stride_h < 1 || p->stride_w < 1 || p->stride_h > std::numeric_limits<int16_t>::max() ||
        p->stride_w > std::numeric_limits<int16_t>::max())
    {
        return HCT_REF_E_PARAM;
    }
    // The reference transpose conv has no dilation.
    if (p->dilation_h != 1 || p->dilation_w != 1)
    {
        return HCT_REF_E_UNSUPPORTED;
    }
    if (in->dims[0] != out->dims[0] || in->dims[3] != filter->dims[3] || filter->dims[0] != out->dims[3])
    {
        return HCT_REF_E_DIMS;
    }
    return HCT_REF_OK;
}

int32_t check_fc(const HctFcParams *p, const HctShape *in, const HctShape *filter, const HctShape *out)
{
    if (p == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_shape(in, 0));
    HCT_TRY(check_shape(filter, 2));
    HCT_TRY(check_shape(out, 0));
    const int32_t out_depth = out->dims[out->rank - 1];
    const int32_t accum_depth = filter->dims[1];
    if (filter->dims[0] != out_depth)
    {
        return HCT_REF_E_DIMS;
    }
    // The input flattens to [batches, accum_depth] with batches taken from
    // the output, exactly as the reference kernel indexes it.
    const int64_t batches = flat_size(out) / out_depth;
    if (flat_size(in) != batches * accum_depth)
    {
        return HCT_REF_E_DIMS;
    }
    return HCT_REF_OK;
}

tflite::FullyConnectedParams to_fc_params(const HctFcParams *p, const HctPerChannelQuant *quant)
{
    tflite::FullyConnectedParams params = {};
    params.input_offset = p->input_offset;
    params.weights_offset = p->weights_offset;
    params.output_offset = p->output_offset;
    params.output_multiplier = quant ? quant->multiplier[0] : 0;
    params.output_shift = quant ? quant->shift[0] : 0;
    params.quantized_activation_min = p->act.min;
    params.quantized_activation_max = p->act.max;
    params.float_activation_min = p->act.fmin;
    params.float_activation_max = p->act.fmax;
    return params;
}

int32_t check_pool(const HctPoolParams *p, const HctShape *in, const HctShape *out)
{
    if (p == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_shape(in, 4));
    HCT_TRY(check_shape(out, 4));
    HCT_TRY(check_int16_range(p->pad_h));
    HCT_TRY(check_int16_range(p->pad_w));
    if (in->dims[0] != out->dims[0] || in->dims[3] != out->dims[3])
    {
        return HCT_REF_E_DIMS;
    }
    HCT_TRY(check_window(in->dims[1], out->dims[1], p->filter_h, p->stride_h, 1, p->pad_h, p->pad_h_offset));
    HCT_TRY(check_window(in->dims[2], out->dims[2], p->filter_w, p->stride_w, 1, p->pad_w, p->pad_w_offset));
    return HCT_REF_OK;
}

tflite::PoolParams to_pool_params(const HctPoolParams *p)
{
    tflite::PoolParams params = {};
    params.activation = tflite::FusedActivationFunctionType::kNone;
    params.padding_type = tflite::PaddingType::kSame;
    params.padding_values = padding(p->pad_h, p->pad_w, p->pad_h_offset, p->pad_w_offset);
    params.stride_height = p->stride_h;
    params.stride_width = p->stride_w;
    params.filter_height = p->filter_h;
    params.filter_width = p->filter_w;
    params.quantized_activation_min = p->act.min;
    params.quantized_activation_max = p->act.max;
    params.float_activation_min = p->act.fmin;
    params.float_activation_max = p->act.fmax;
    return params;
}

// Unpacks an int4 filter; returns nullptr on allocation failure.
std::unique_ptr<int8_t[]> unpack_int4(const int8_t *packed, int64_t count)
{
    std::unique_ptr<int8_t[]> unpacked(new (std::nothrow) int8_t[static_cast<size_t>(count)]);
    if (unpacked)
    {
        tflite::tensor_utils::UnpackDenseInt4IntoInt8(packed, static_cast<int>(count), unpacked.get());
    }
    return unpacked;
}

template <typename In, typename Bias>
int32_t conv_quantized(const HctConvParams *params,
                       const HctPerChannelQuant *quant,
                       const HctShape *input_shape,
                       const In *input,
                       const HctShape *filter_shape,
                       const int8_t *filter,
                       const Bias *bias,
                       int32_t bias_len,
                       const HctShape *output_shape,
                       In *output)
{
    HCT_TRY(check_conv(params, input_shape, filter_shape, output_shape));
    HCT_TRY(check_buffers(input, filter, output));
    const int32_t channels = output_shape->dims[3];
    HCT_TRY(check_quant(quant, channels, max_shift<In>()));
    HCT_TRY(check_bias(bias, bias_len, channels));
    HCT_TRY(check_activation<In>(params->act));
    HCT_TRY(check_offsets<In>(params->input_offset, params->output_offset));
    const tflite::ConvParams p = to_conv_params(params);
    const RuntimeShape bias_shape(1, &channels);
    tflite::reference_integer_ops::ConvPerChannel(p, quant->multiplier, quant->shift, to_runtime(input_shape), input,
                                                  to_runtime(filter_shape), filter, bias_shape, bias,
                                                  to_runtime(output_shape), output);
    return HCT_REF_OK;
}

template <typename In, typename Bias>
int32_t dwconv_quantized(const HctDwConvParams *params,
                         const HctPerChannelQuant *quant,
                         const HctShape *input_shape,
                         const In *input,
                         const HctShape *filter_shape,
                         const int8_t *filter,
                         const Bias *bias,
                         int32_t bias_len,
                         const HctShape *output_shape,
                         In *output)
{
    HCT_TRY(check_dwconv(params, input_shape, filter_shape, output_shape));
    HCT_TRY(check_buffers(input, filter, output));
    const int32_t channels = output_shape->dims[3];
    HCT_TRY(check_quant(quant, channels, max_shift<In>()));
    HCT_TRY(check_bias(bias, bias_len, channels));
    HCT_TRY(check_activation<In>(params->conv.act));
    HCT_TRY(check_offsets<In>(params->conv.input_offset, params->conv.output_offset));
    const tflite::DepthwiseParams p = to_dw_params(params);
    // The reference kernel reads bias_shape.FlatSize() even without bias.
    const RuntimeShape bias_shape(1, &channels);
    std::vector<Bias> zero_bias;
    if (bias == nullptr)
    {
        zero_bias.assign(static_cast<size_t>(channels), Bias{0});
        bias = zero_bias.data();
    }
    tflite::reference_integer_ops::DepthwiseConvPerChannel(p, quant->multiplier, quant->shift, to_runtime(input_shape),
                                                           input, to_runtime(filter_shape), filter, bias_shape, bias,
                                                           to_runtime(output_shape), output);
    return HCT_REF_OK;
}

template <typename In, typename Bias>
int32_t fc_quantized(const HctFcParams *params,
                     const HctPerChannelQuant *quant,
                     const HctShape *input_shape,
                     const In *input,
                     const HctShape *filter_shape,
                     const int8_t *filter,
                     const Bias *bias,
                     int32_t bias_len,
                     const HctShape *output_shape,
                     In *output)
{
    HCT_TRY(check_fc(params, input_shape, filter_shape, output_shape));
    HCT_TRY(check_buffers(input, filter, output));
    const int32_t channels = output_shape->dims[output_shape->rank - 1];
    if (quant == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    // One entry selects the per-tensor kernel; anything else must be one per
    // output channel.
    const bool per_channel = quant->count != 1;
    HCT_TRY(check_quant(quant, per_channel ? channels : 1, max_shift<In>()));
    HCT_TRY(check_bias(bias, bias_len, channels));
    HCT_TRY(check_activation<In>(params->act));
    HCT_TRY(check_offsets<In>(params->input_offset, params->output_offset));
    if (params->weights_offset < -127 || params->weights_offset > 128)
    {
        return HCT_REF_E_PARAM;
    }
    const tflite::FullyConnectedParams p = to_fc_params(params, quant);
    const RuntimeShape bias_shape(1, &channels);
    if (per_channel && params->weights_offset != 0)
    {
        // TFLite's per-channel kernel assumes symmetric weights; CMSIS-NN's still
        // honours a filter offset. Widen the weights with the offset applied and
        // run the same TFLM per-channel kernel on them.
        const int64_t count = flat_size(filter_shape);
        std::vector<int16_t> offset_filter;
        try
        {
            offset_filter.resize(static_cast<size_t>(count));
        }
        catch (const std::bad_alloc &)
        {
            return HCT_REF_E_PARAM;
        }
        for (int64_t i = 0; i < count; ++i)
        {
            offset_filter[static_cast<size_t>(i)] = static_cast<int16_t>(filter[i] + params->weights_offset);
        }
        tflite::reference_integer_ops::FullyConnectedPerChannel<In, int16_t, In, Bias>(
            p, quant->multiplier, quant->shift, to_runtime(input_shape), input, to_runtime(filter_shape),
            offset_filter.data(), bias_shape, bias, to_runtime(output_shape), output);
    }
    else if (per_channel)
    {
        tflite::reference_integer_ops::FullyConnectedPerChannel<In, int8_t, In, Bias>(
            p, quant->multiplier, quant->shift, to_runtime(input_shape), input, to_runtime(filter_shape), filter,
            bias_shape, bias, to_runtime(output_shape), output);
    }
    else
    {
        tflite::reference_integer_ops::FullyConnected<In, int8_t, In, Bias>(p, to_runtime(input_shape), input,
                                                                            to_runtime(filter_shape), filter,
                                                                            bias_shape, bias, to_runtime(output_shape),
                                                                            output);
    }
    return HCT_REF_OK;
}

template <typename In, typename Bias>
int32_t tconv_quantized(const HctConvParams *params,
                        const HctPerChannelQuant *quant,
                        const HctShape *input_shape,
                        const In *input,
                        const HctShape *filter_shape,
                        const int8_t *filter,
                        const Bias *bias,
                        int32_t bias_len,
                        const HctShape *output_shape,
                        In *output)
{
    HCT_TRY(check_tconv(params, input_shape, filter_shape, output_shape));
    HCT_TRY(check_buffers(input, filter, output));
    const int32_t channels = output_shape->dims[3];
    HCT_TRY(check_quant(quant, channels, max_shift<In>()));
    HCT_TRY(check_bias(bias, bias_len, channels));
    HCT_TRY(check_activation<In>(params->act));
    HCT_TRY(check_offsets<In>(params->input_offset, params->output_offset));
    const tflite::ConvParams p = to_conv_params(params);
    const RuntimeShape bias_shape(1, &channels);
    using Scratch = typename std::conditional<sizeof(In) == 1, int32_t, Bias>::type;
    std::vector<Scratch> scratch;
    try
    {
        scratch.resize(static_cast<size_t>(flat_size(output_shape)));
    }
    catch (const std::bad_alloc &)
    {
        return HCT_REF_E_PARAM;
    }
    const RuntimeShape no_im2col;
    tflite::reference_integer_ops::TransposeConv(p, quant->multiplier, quant->shift, to_runtime(input_shape), input,
                                                 to_runtime(filter_shape), filter, bias_shape, bias,
                                                 to_runtime(output_shape), output, no_im2col, nullptr,
                                                 scratch.data());
    return HCT_REF_OK;
}

template <typename T, typename Kernel>
int32_t pool_quantized(const HctPoolParams *params,
                       const HctShape *input_shape,
                       const T *input,
                       const HctShape *output_shape,
                       T *output,
                       Kernel kernel)
{
    HCT_TRY(check_pool(params, input_shape, output_shape));
    HCT_TRY(check_buffers(input, output));
    HCT_TRY(check_activation<T>(params->act));
    return kernel(to_pool_params(params), to_runtime(input_shape), input, to_runtime(output_shape), output);
}

} // namespace

extern "C" {

int32_t hct_ref_abi_version(void)
{
    return HCT_REF_ABI_VERSION;
}

int32_t hct_ref_sizeof(const char *type_name)
{
    if (type_name == nullptr)
    {
        return -1;
    }
    struct Entry
    {
        const char *name;
        size_t size;
    };
    static const Entry entries[] = {
        {"HctShape", sizeof(HctShape)},
        {"HctActivation", sizeof(HctActivation)},
        {"HctConvParams", sizeof(HctConvParams)},
        {"HctDwConvParams", sizeof(HctDwConvParams)},
        {"HctFcParams", sizeof(HctFcParams)},
        {"HctPoolParams", sizeof(HctPoolParams)},
        {"HctQuant", sizeof(HctQuant)},
        {"HctBinaryParams", sizeof(HctBinaryParams)},
        {"HctPerChannelQuant", sizeof(HctPerChannelQuant)},
        {"HctSoftmaxParams", sizeof(HctSoftmaxParams)},
        {"HctLutActParams", sizeof(HctLutActParams)},
        {"HctLeakyReluParams", sizeof(HctLeakyReluParams)},
        {"HctPreluParams", sizeof(HctPreluParams)},
        {"HctHardSwishParams", sizeof(HctHardSwishParams)},
        {"HctReluParams", sizeof(HctReluParams)},
        {"HctMeanParams", sizeof(HctMeanParams)},
        {"HctRsqrtParams", sizeof(HctRsqrtParams)},
        {"HctBmmParams", sizeof(HctBmmParams)},
    };
    for (const Entry &entry : entries)
    {
        if (std::strcmp(entry.name, type_name) == 0)
        {
            return static_cast<int32_t>(entry.size);
        }
    }
    return -1;
}

int32_t hct_ref_quantize_multiplier(double real_multiplier, HctQuant *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (!std::isfinite(real_multiplier) || real_multiplier < 0.0)
    {
        return HCT_REF_E_PARAM;
    }
    int shift = 0;
    tflite::QuantizeMultiplier(real_multiplier, &out->multiplier, &shift);
    out->shift = shift;
    return HCT_REF_OK;
}

int32_t hct_ref_quantize_multiplier_smaller_than_one_exp(double real_multiplier, HctQuant *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (!std::isfinite(real_multiplier) || real_multiplier <= 0.0 || real_multiplier >= 1.0)
    {
        return HCT_REF_E_PARAM;
    }
    int shift = 0;
    tflite::QuantizeMultiplierSmallerThanOneExp(real_multiplier, &out->multiplier, &shift);
    out->shift = shift;
    return HCT_REF_OK;
}

int32_t hct_ref_preprocess_softmax_scaling(double beta, double input_scale, int32_t input_integer_bits, HctQuant *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (!std::isfinite(beta) || !std::isfinite(input_scale) || beta <= 0.0 || input_scale <= 0.0 ||
        input_integer_bits < 0 || input_integer_bits > 31)
    {
        return HCT_REF_E_PARAM;
    }
    int shift = 0;
    tflite::PreprocessSoftmaxScaling(beta, input_scale, input_integer_bits, &out->multiplier, &shift);
    out->shift = shift;
    return HCT_REF_OK;
}

int32_t hct_ref_calculate_input_radius(int32_t input_integer_bits,
                                       int32_t input_left_shift,
                                       int32_t total_signed_bits,
                                       int32_t *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (input_integer_bits < 0 || input_integer_bits > 30 || input_left_shift < 0 || input_left_shift > 62 ||
        total_signed_bits < input_integer_bits || total_signed_bits > 62)
    {
        return HCT_REF_E_PARAM;
    }
    *out = tflite::CalculateInputRadius(input_integer_bits, input_left_shift, total_signed_bits);
    return HCT_REF_OK;
}

int32_t hct_ref_activation_range_quantized(int32_t activation,
                                           float scale,
                                           int32_t zero_point,
                                           int32_t qmin,
                                           int32_t qmax,
                                           int32_t *act_min,
                                           int32_t *act_max)
{
    if (act_min == nullptr || act_max == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (!std::isfinite(scale) || scale <= 0.0f || qmin > qmax)
    {
        return HCT_REF_E_PARAM;
    }
    // Same arithmetic as CalculateActivationRangeQuantizedImpl in TFLite's
    // kernel_util.cc: float division, TfLiteRound (std::round), then clamp
    // against the storage range.
    const auto quantize = [scale, zero_point](float f) {
        return zero_point + static_cast<int32_t>(std::round(f / scale));
    };
    switch (activation)
    {
    case HCT_REF_ACT_NONE:
        *act_min = qmin;
        *act_max = qmax;
        break;
    case HCT_REF_ACT_RELU:
        *act_min = std::max(qmin, quantize(0.0f));
        *act_max = qmax;
        break;
    case HCT_REF_ACT_RELU6:
        *act_min = std::max(qmin, quantize(0.0f));
        *act_max = std::min(qmax, quantize(6.0f));
        break;
    case HCT_REF_ACT_RELU_N1_TO_1:
        *act_min = std::max(qmin, quantize(-1.0f));
        *act_max = std::min(qmax, quantize(1.0f));
        break;
    default:
        return HCT_REF_E_UNSUPPORTED;
    }
    return HCT_REF_OK;
}

int32_t hct_ref_downscale_multiplier_to_s16(int32_t multiplier, int16_t *out)
{
    if (out == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    if (multiplier < 0)
    {
        return HCT_REF_E_PARAM;
    }
    tflite::DownScaleInt32ToInt16Multiplier(multiplier, out);
    return HCT_REF_OK;
}

int32_t hct_ref_conv_s8(const HctConvParams *params,
                        const HctPerChannelQuant *quant,
                        const HctShape *input_shape,
                        const int8_t *input,
                        const HctShape *filter_shape,
                        const int8_t *filter,
                        const int32_t *bias,
                        int32_t bias_len,
                        const HctShape *output_shape,
                        int8_t *output)
{
    return conv_quantized<int8_t, int32_t>(params, quant, input_shape, input, filter_shape, filter, bias, bias_len,
                                           output_shape, output);
}

int32_t hct_ref_conv_s4(const HctConvParams *params,
                        const HctPerChannelQuant *quant,
                        const HctShape *input_shape,
                        const int8_t *input,
                        const HctShape *filter_shape,
                        const int8_t *packed_filter,
                        const int32_t *bias,
                        int32_t bias_len,
                        const HctShape *output_shape,
                        int8_t *output)
{
    HCT_TRY(check_shape(filter_shape, 4));
    if (packed_filter == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    const std::unique_ptr<int8_t[]> filter = unpack_int4(packed_filter, flat_size(filter_shape));
    if (!filter)
    {
        return HCT_REF_E_PARAM;
    }
    return conv_quantized<int8_t, int32_t>(params, quant, input_shape, input, filter_shape, filter.get(), bias,
                                           bias_len, output_shape, output);
}

int32_t hct_ref_conv_s16(const HctConvParams *params,
                         const HctPerChannelQuant *quant,
                         const HctShape *input_shape,
                         const int16_t *input,
                         const HctShape *filter_shape,
                         const int8_t *filter,
                         const int64_t *bias,
                         int32_t bias_len,
                         const HctShape *output_shape,
                         int16_t *output)
{
    return conv_quantized<int16_t, int64_t>(params, quant, input_shape, input, filter_shape, filter, bias, bias_len,
                                            output_shape, output);
}

int32_t hct_ref_conv_s16_b32(const HctConvParams *params,
                             const HctPerChannelQuant *quant,
                             const HctShape *input_shape,
                             const int16_t *input,
                             const HctShape *filter_shape,
                             const int8_t *filter,
                             const int32_t *bias,
                             int32_t bias_len,
                             const HctShape *output_shape,
                             int16_t *output)
{
    return conv_quantized<int16_t, int32_t>(params, quant, input_shape, input, filter_shape, filter, bias, bias_len,
                                            output_shape, output);
}

int32_t hct_ref_conv_f32(const HctConvParams *params,
                         const HctShape *input_shape,
                         const float *input,
                         const HctShape *filter_shape,
                         const float *filter,
                         const float *bias,
                         int32_t bias_len,
                         const HctShape *output_shape,
                         float *output)
{
    HCT_TRY(check_conv(params, input_shape, filter_shape, output_shape));
    HCT_TRY(check_buffers(input, filter, output));
    const int32_t channels = output_shape->dims[3];
    HCT_TRY(check_bias(bias, bias_len, channels));
    HCT_TRY(check_activation<float>(params->act));
    HCT_TRY(check_offsets<float>(params->input_offset, params->output_offset));
    const RuntimeShape bias_shape(1, &channels);
    const RuntimeShape no_im2col;
    tflite::reference_ops::Conv(to_conv_params(params), to_runtime(input_shape), input, to_runtime(filter_shape),
                                filter, bias_shape, bias, to_runtime(output_shape), output, no_im2col, nullptr);
    return HCT_REF_OK;
}

int32_t hct_ref_dwconv_s8(const HctDwConvParams *params,
                          const HctPerChannelQuant *quant,
                          const HctShape *input_shape,
                          const int8_t *input,
                          const HctShape *filter_shape,
                          const int8_t *filter,
                          const int32_t *bias,
                          int32_t bias_len,
                          const HctShape *output_shape,
                          int8_t *output)
{
    return dwconv_quantized<int8_t, int32_t>(params, quant, input_shape, input, filter_shape, filter, bias, bias_len,
                                             output_shape, output);
}

int32_t hct_ref_dwconv_s4(const HctDwConvParams *params,
                          const HctPerChannelQuant *quant,
                          const HctShape *input_shape,
                          const int8_t *input,
                          const HctShape *filter_shape,
                          const int8_t *packed_filter,
                          const int32_t *bias,
                          int32_t bias_len,
                          const HctShape *output_shape,
                          int8_t *output)
{
    HCT_TRY(check_shape(filter_shape, 4));
    if (packed_filter == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    const std::unique_ptr<int8_t[]> filter = unpack_int4(packed_filter, flat_size(filter_shape));
    if (!filter)
    {
        return HCT_REF_E_PARAM;
    }
    return dwconv_quantized<int8_t, int32_t>(params, quant, input_shape, input, filter_shape, filter.get(), bias,
                                             bias_len, output_shape, output);
}

int32_t hct_ref_dwconv_s16(const HctDwConvParams *params,
                           const HctPerChannelQuant *quant,
                           const HctShape *input_shape,
                           const int16_t *input,
                           const HctShape *filter_shape,
                           const int8_t *filter,
                           const int64_t *bias,
                           int32_t bias_len,
                           const HctShape *output_shape,
                           int16_t *output)
{
    return dwconv_quantized<int16_t, int64_t>(params, quant, input_shape, input, filter_shape, filter, bias, bias_len,
                                              output_shape, output);
}

int32_t hct_ref_dwconv_f32(const HctDwConvParams *params,
                           const HctShape *input_shape,
                           const float *input,
                           const HctShape *filter_shape,
                           const float *filter,
                           const float *bias,
                           int32_t bias_len,
                           const HctShape *output_shape,
                           float *output)
{
    HCT_TRY(check_dwconv(params, input_shape, filter_shape, output_shape));
    HCT_TRY(check_buffers(input, filter, output));
    const int32_t channels = output_shape->dims[3];
    HCT_TRY(check_bias(bias, bias_len, channels));
    HCT_TRY(check_activation<float>(params->conv.act));
    HCT_TRY(check_offsets<float>(params->conv.input_offset, params->conv.output_offset));
    const RuntimeShape bias_shape(1, &channels);
    std::vector<float> zero_bias;
    if (bias == nullptr)
    {
        zero_bias.assign(static_cast<size_t>(channels), 0.0f);
        bias = zero_bias.data();
    }
    tflite::reference_ops::DepthwiseConv(to_dw_params(params), to_runtime(input_shape), input,
                                         to_runtime(filter_shape), filter, bias_shape, bias, to_runtime(output_shape),
                                         output);
    return HCT_REF_OK;
}

int32_t hct_ref_fc_s8(const HctFcParams *params,
                      const HctPerChannelQuant *quant,
                      const HctShape *input_shape,
                      const int8_t *input,
                      const HctShape *filter_shape,
                      const int8_t *filter,
                      const int32_t *bias,
                      int32_t bias_len,
                      const HctShape *output_shape,
                      int8_t *output)
{
    return fc_quantized<int8_t, int32_t>(params, quant, input_shape, input, filter_shape, filter, bias, bias_len,
                                         output_shape, output);
}

int32_t hct_ref_fc_s4(const HctFcParams *params,
                      const HctPerChannelQuant *quant,
                      const HctShape *input_shape,
                      const int8_t *input,
                      const HctShape *filter_shape,
                      const int8_t *packed_filter,
                      const int32_t *bias,
                      int32_t bias_len,
                      const HctShape *output_shape,
                      int8_t *output)
{
    HCT_TRY(check_shape(filter_shape, 2));
    if (packed_filter == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    const std::unique_ptr<int8_t[]> filter = unpack_int4(packed_filter, flat_size(filter_shape));
    if (!filter)
    {
        return HCT_REF_E_PARAM;
    }
    return fc_quantized<int8_t, int32_t>(params, quant, input_shape, input, filter_shape, filter.get(), bias, bias_len,
                                         output_shape, output);
}

int32_t hct_ref_fc_s16(const HctFcParams *params,
                       const HctPerChannelQuant *quant,
                       const HctShape *input_shape,
                       const int16_t *input,
                       const HctShape *filter_shape,
                       const int8_t *filter,
                       const int64_t *bias,
                       int32_t bias_len,
                       const HctShape *output_shape,
                       int16_t *output)
{
    return fc_quantized<int16_t, int64_t>(params, quant, input_shape, input, filter_shape, filter, bias, bias_len,
                                          output_shape, output);
}

int32_t hct_ref_fc_f32(const HctFcParams *params,
                       const HctShape *input_shape,
                       const float *input,
                       const HctShape *filter_shape,
                       const float *filter,
                       const float *bias,
                       int32_t bias_len,
                       const HctShape *output_shape,
                       float *output)
{
    HCT_TRY(check_fc(params, input_shape, filter_shape, output_shape));
    HCT_TRY(check_buffers(input, filter, output));
    const int32_t channels = output_shape->dims[output_shape->rank - 1];
    HCT_TRY(check_bias(bias, bias_len, channels));
    HCT_TRY(check_activation<float>(params->act));
    if (params->input_offset != 0 || params->weights_offset != 0 || params->output_offset != 0)
    {
        return HCT_REF_E_PARAM;
    }
    const RuntimeShape bias_shape(1, &channels);
    tflite::reference_ops::FullyConnected(to_fc_params(params, nullptr), to_runtime(input_shape), input,
                                          to_runtime(filter_shape), filter, bias_shape, bias,
                                          to_runtime(output_shape), output);
    return HCT_REF_OK;
}

int32_t hct_ref_tconv_s8(const HctConvParams *params,
                         const HctPerChannelQuant *quant,
                         const HctShape *input_shape,
                         const int8_t *input,
                         const HctShape *filter_shape,
                         const int8_t *filter,
                         const int32_t *bias,
                         int32_t bias_len,
                         const HctShape *output_shape,
                         int8_t *output)
{
    return tconv_quantized<int8_t, int32_t>(params, quant, input_shape, input, filter_shape, filter, bias, bias_len,
                                            output_shape, output);
}

int32_t hct_ref_tconv_s16(const HctConvParams *params,
                          const HctPerChannelQuant *quant,
                          const HctShape *input_shape,
                          const int16_t *input,
                          const HctShape *filter_shape,
                          const int8_t *filter,
                          const int64_t *bias,
                          int32_t bias_len,
                          const HctShape *output_shape,
                          int16_t *output)
{
    return tconv_quantized<int16_t, int64_t>(params, quant, input_shape, input, filter_shape, filter, bias, bias_len,
                                             output_shape, output);
}

int32_t hct_ref_tconv_f32(const HctConvParams *params,
                          const HctShape *input_shape,
                          const float *input,
                          const HctShape *filter_shape,
                          const float *filter,
                          const float *bias,
                          int32_t bias_len,
                          const HctShape *output_shape,
                          float *output)
{
    HCT_TRY(check_tconv(params, input_shape, filter_shape, output_shape));
    HCT_TRY(check_buffers(input, filter, output));
    const int32_t channels = output_shape->dims[3];
    HCT_TRY(check_bias(bias, bias_len, channels));
    HCT_TRY(check_activation<float>(params->act));
    HCT_TRY(check_offsets<float>(params->input_offset, params->output_offset));
    const RuntimeShape bias_shape(1, &channels);
    const RuntimeShape no_im2col;
    tflite::reference_ops::TransposeConv(to_conv_params(params), to_runtime(input_shape), input,
                                         to_runtime(filter_shape), filter, bias_shape, bias, to_runtime(output_shape),
                                         output, no_im2col, nullptr);
    return HCT_REF_OK;
}

int32_t hct_ref_avgpool_s8(const HctPoolParams *params,
                           const HctShape *input_shape,
                           const int8_t *input,
                           const HctShape *output_shape,
                           int8_t *output)
{
    return pool_quantized<int8_t>(params, input_shape, input, output_shape, output,
                                  [](const tflite::PoolParams &p, const RuntimeShape &is, const int8_t *i,
                                     const RuntimeShape &os, int8_t *o) {
                                      return tflite::reference_integer_ops::AveragePool(p, is, i, os, o)
                                                 ? HCT_REF_OK
                                                 : HCT_REF_E_PARAM;
                                  });
}

int32_t hct_ref_avgpool_s16(const HctPoolParams *params,
                            const HctShape *input_shape,
                            const int16_t *input,
                            const HctShape *output_shape,
                            int16_t *output)
{
    return pool_quantized<int16_t>(params, input_shape, input, output_shape, output,
                                   [](const tflite::PoolParams &p, const RuntimeShape &is, const int16_t *i,
                                      const RuntimeShape &os, int16_t *o) {
                                       return tflite::reference_integer_ops::AveragePool(p, is, i, os, o)
                                                  ? HCT_REF_OK
                                                  : HCT_REF_E_PARAM;
                                   });
}

int32_t hct_ref_avgpool_f32(const HctPoolParams *params,
                            const HctShape *input_shape,
                            const float *input,
                            const HctShape *output_shape,
                            float *output)
{
    return pool_quantized<float>(params, input_shape, input, output_shape, output,
                                 [](const tflite::PoolParams &p, const RuntimeShape &is, const float *i,
                                    const RuntimeShape &os, float *o) {
                                     return tflite::reference_ops::AveragePool(p, is, i, os, o) ? HCT_REF_OK
                                                                                                : HCT_REF_E_PARAM;
                                 });
}

int32_t hct_ref_maxpool_s8(const HctPoolParams *params,
                           const HctShape *input_shape,
                           const int8_t *input,
                           const HctShape *output_shape,
                           int8_t *output)
{
    return pool_quantized<int8_t>(params, input_shape, input, output_shape, output,
                                  [](const tflite::PoolParams &p, const RuntimeShape &is, const int8_t *i,
                                     const RuntimeShape &os, int8_t *o) {
                                      tflite::reference_integer_ops::MaxPool(p, is, i, os, o);
                                      return HCT_REF_OK;
                                  });
}

int32_t hct_ref_maxpool_s16(const HctPoolParams *params,
                            const HctShape *input_shape,
                            const int16_t *input,
                            const HctShape *output_shape,
                            int16_t *output)
{
    return pool_quantized<int16_t>(params, input_shape, input, output_shape, output,
                                   [](const tflite::PoolParams &p, const RuntimeShape &is, const int16_t *i,
                                      const RuntimeShape &os, int16_t *o) {
                                       tflite::reference_integer_ops::MaxPool(p, is, i, os, o);
                                       return HCT_REF_OK;
                                   });
}

int32_t hct_ref_maxpool_f32(const HctPoolParams *params,
                            const HctShape *input_shape,
                            const float *input,
                            const HctShape *output_shape,
                            float *output)
{
    return pool_quantized<float>(params, input_shape, input, output_shape, output,
                                 [](const tflite::PoolParams &p, const RuntimeShape &is, const float *i,
                                    const RuntimeShape &os, float *o) {
                                     tflite::reference_ops::MaxPool(p, is, i, os, o);
                                     return HCT_REF_OK;
                                 });
}

} // extern "C"
