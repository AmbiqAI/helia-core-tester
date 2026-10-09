/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Integer UNIDIRECTIONAL_SEQUENCE_LSTM entries of the hct_ref shim (see hct_ref.h).
 * The prepare is TFLM's (micro/kernels/unidirectional_sequence_lstm.cc and
 * lstm_eval_common.cc, integer path, no peephole/projection/layer norm); the
 * evaluation is TFLM's own lstm_internal::EvalLstm over tensors the shim owns.
 */
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "hct_ref.h"
#include "hct_ref_internal.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"

namespace {

using namespace hct;

constexpr float kGateInputScale = 0.000244140625f;  // 2^-12, Q3.12
constexpr float kGateOutputScale = 0.000030517578125f; // 2^-15, Q0.15

bool valid_scale(float scale)
{
    return std::isfinite(scale) && scale > 0.0f;
}

// TfLiteIntArray has a flexible array member: back it with ints the caller owns.
struct Dims
{
    std::vector<int> storage;
    explicit Dims(std::initializer_list<int> dims) : storage(1 + dims.size())
    {
        storage[0] = static_cast<int>(dims.size());
        int i = 1;
        for (int d : dims)
        {
            storage[i++] = d;
        }
    }
    TfLiteIntArray *get()
    {
        return reinterpret_cast<TfLiteIntArray *>(storage.data());
    }
};

TfLiteEvalTensor eval_tensor(const void *data, TfLiteIntArray *dims, TfLiteType type)
{
    TfLiteEvalTensor t;
    t.data.raw = const_cast<char *>(static_cast<const char *>(data));
    t.dims = dims;
    t.type = type;
    return t;
}

// CreateGateParams -> CalculateOpDataFullyConnected (per tensor, no activation,
// int16 output at the 2^-12 gate scale) -> FullyConnectedParamsQuantized.
tflite::FullyConnectedParams gate_fc(float input_scale, int32_t input_zero_point, float weight_scale)
{
    tflite::FullyConnectedParams p = {};
    const double real = static_cast<double>(input_scale) * static_cast<double>(weight_scale) /
                        static_cast<double>(kGateInputScale);
    int shift = 0;
    tflite::QuantizeMultiplier(real, &p.output_multiplier, &shift);
    p.output_shift = shift;
    p.input_offset = -input_zero_point;
    p.weights_offset = 0;
    p.output_offset = 0;
    p.quantized_activation_min = std::numeric_limits<int16_t>::min();
    p.quantized_activation_max = std::numeric_limits<int16_t>::max();
    return p;
}

// CreateInterGateMulParams.
tflite::ArithmeticParams inter_gate_mul(float s1, float s2, float out_scale, bool int8_out, int32_t out_zp)
{
    tflite::ArithmeticParams p = {};
    p.quantized_activation_min = int8_out ? std::numeric_limits<int8_t>::min() : std::numeric_limits<int16_t>::min();
    p.quantized_activation_max = int8_out ? std::numeric_limits<int8_t>::max() : std::numeric_limits<int16_t>::max();
    p.input1_offset = 0;
    p.input2_offset = 0;
    p.output_offset = out_zp;
    const double effective = static_cast<double>(s1) * static_cast<double>(s2) / static_cast<double>(out_scale);
    int shift = 0;
    tflite::QuantizeMultiplier(effective, &p.output_multiplier, &shift);
    p.output_shift = shift;
    return p;
}

template <typename Act> int32_t check_params(const HctLstmParams *p)
{
    if (p->batch < 1 || p->time_steps < 1 || p->input_size < 1 || p->hidden_size < 1 ||
        (p->time_major != 0 && p->time_major != 1))
    {
        return HCT_REF_E_DIMS;
    }
    const int64_t elems = static_cast<int64_t>(p->batch) * p->time_steps *
                          (static_cast<int64_t>(p->input_size) + p->hidden_size);
    if (elems > kMaxElements || static_cast<int64_t>(p->hidden_size) * (p->input_size + p->hidden_size) > kMaxElements)
    {
        return HCT_REF_E_DIMS;
    }
    if (!valid_scale(p->input_scale) || !valid_scale(p->output_scale) || !valid_scale(p->cell_scale) ||
        !std::isfinite(p->cell_clip) || p->cell_clip < 0.0f)
    {
        return HCT_REF_E_PARAM;
    }
    for (float s : p->weight_scales)
    {
        if (!valid_scale(s))
        {
            return HCT_REF_E_PARAM;
        }
    }
    int log2 = 0;
    if (!tflite::CheckedLog2(p->cell_scale, &log2))
    {
        return HCT_REF_E_PARAM; // the integer tanh needs a power-of-two cell scale
    }
    const int32_t lo = std::numeric_limits<Act>::min();
    const int32_t hi = std::numeric_limits<Act>::max();
    if (p->input_zero_point < lo || p->input_zero_point > hi || p->output_zero_point < lo || p->output_zero_point > hi)
    {
        return HCT_REF_E_PARAM;
    }
    if (sizeof(Act) == 2 && (p->input_zero_point != 0 || p->output_zero_point != 0))
    {
        return HCT_REF_E_PARAM;
    }
    return HCT_REF_OK;
}

template <typename Act, typename Bias>
int32_t lstm(const HctLstmParams *p,
             const Act *input,
             const int8_t *const *weights,
             const Bias *const *biases,
             Act *output,
             TfLiteType act_type,
             TfLiteType bias_type)
{
    if (p == nullptr || weights == nullptr || biases == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_buffers(input, output));
    for (int i = 0; i < 8; ++i)
    {
        if (weights[i] == nullptr)
        {
            return HCT_REF_E_NULL;
        }
    }
    for (int i = 0; i < 4; ++i)
    {
        if (biases[i] == nullptr)
        {
            return HCT_REF_E_NULL;
        }
    }
    HCT_TRY(check_params<Act>(p));

    const int B = p->batch, T = p->time_steps, I = p->input_size, H = p->hidden_size;
    const bool int8_act = sizeof(Act) == 1;

    tflite::OpDataLSTM op = {};
    op.size_info.time_major = p->time_major != 0;
    op.size_info.batch_size = B;
    op.size_info.time_steps = T;
    op.size_info.input_dimension = I;
    op.size_info.state_dimension = H;
    // CreateLstmCellStateInfo.
    int power = 0;
    tflite::CheckedLog2(p->cell_scale, &power);
    op.cell_state_info.cell_state_scale_power = power;
    op.cell_state_info.cell_clip = p->cell_clip;
    op.cell_state_info.quantized_cell_clip = static_cast<int16_t>(
        std::min(std::max(static_cast<double>(p->cell_clip) / static_cast<double>(p->cell_scale), -32768.0), 32767.0));
    op.cell_gate_nonlinear_type = kTfLiteActTanh;

    // weights: input-to-{input, forget, cell, output}, recurrent-to-{input, forget, cell, output}.
    tflite::GateParameters *gates[4] = {&op.input_gate_parameters, &op.forget_gate_parameters,
                                         &op.cell_gate_parameters, &op.output_gate_parameters};
    for (int g = 0; g < 4; ++g)
    {
        gates[g]->input_fc_params = gate_fc(p->input_scale, p->input_zero_point, p->weight_scales[g]);
        gates[g]->recurrent_fc_params = gate_fc(p->output_scale, p->output_zero_point, p->weight_scales[4 + g]);
    }
    op.inter_gate_parameters.forget_cell_mul_params =
        inter_gate_mul(kGateOutputScale, p->cell_scale, p->cell_scale, false, 0);
    op.inter_gate_parameters.input_mul_params = inter_gate_mul(kGateOutputScale, kGateOutputScale, p->cell_scale, false, 0);
    op.inter_gate_parameters.output_mul_params =
        inter_gate_mul(kGateOutputScale, kGateOutputScale, p->output_scale, int8_act, p->output_zero_point);

    Dims in_dims = p->time_major ? Dims{T, B, I} : Dims{B, T, I};
    Dims out_dims = p->time_major ? Dims{T, B, H} : Dims{B, T, H};
    Dims w_in_dims{H, I}, w_rec_dims{H, H}, bias_dims{H}, state_dims{B, H};

    std::vector<Act> hidden_state(static_cast<size_t>(B) * H, static_cast<Act>(p->output_zero_point));
    std::vector<int16_t> cell_state(static_cast<size_t>(B) * H, 0);

    TfLiteEvalTensor tensors[24];
    std::memset(tensors, 0, sizeof(tensors));
    tflite::LSTMKernelContents content = {};
    for (int i = 0; i < 24; ++i)
    {
        content.internal_tensors[i] = nullptr;
    }
    tensors[tflite::kLstmInputTensor] = eval_tensor(input, in_dims.get(), act_type);
    const int in_index[4] = {tflite::kLstmInputToInputWeightsTensor, tflite::kLstmInputToForgetWeightsTensor,
                             tflite::kLstmInputToCellWeightsTensor, tflite::kLstmInputToOutputWeightsTensor};
    const int rec_index[4] = {tflite::kLstmRecurrentToInputWeightsTensor, tflite::kLstmRecurrentToForgetWeightsTensor,
                              tflite::kLstmRecurrentToCellWeightsTensor, tflite::kLstmRecurrentToOutputWeightsTensor};
    const int bias_index[4] = {tflite::kLstmInputGateBiasTensor, tflite::kLstmForgetGateBiasTensor,
                               tflite::kLstmCellGateBiasTensor, tflite::kLstmOutputGateBiasTensor};
    for (int g = 0; g < 4; ++g)
    {
        tensors[in_index[g]] = eval_tensor(weights[g], w_in_dims.get(), kTfLiteInt8);
        tensors[rec_index[g]] = eval_tensor(weights[4 + g], w_rec_dims.get(), kTfLiteInt8);
        tensors[bias_index[g]] = eval_tensor(biases[g], bias_dims.get(), bias_type);
    }
    tensors[tflite::kLstmOutputStateTensor] = eval_tensor(hidden_state.data(), state_dims.get(), act_type);
    tensors[tflite::kLstmCellStateTensor] = eval_tensor(cell_state.data(), state_dims.get(), kTfLiteInt16);
    const int used[] = {tflite::kLstmInputTensor,       in_index[0],   in_index[1],   in_index[2],   in_index[3],
                        rec_index[0],                   rec_index[1],  rec_index[2],  rec_index[3],  bias_index[0],
                        bias_index[1],                  bias_index[2], bias_index[3], tflite::kLstmOutputStateTensor,
                        tflite::kLstmCellStateTensor};
    for (int i : used)
    {
        content.internal_tensors[i] = &tensors[i];
    }
    TfLiteEvalTensor out = eval_tensor(output, out_dims.get(), act_type);
    content.output_tensor = &out;

    // Four cell-type scratch buffers of batch * state elements (UnidirectionalSequenceLstmPrepare).
    std::vector<int16_t> scratch(static_cast<size_t>(4) * B * H);
    tflite::LSTMBuffers<int16_t> buffers;
    buffers.buffer0 = scratch.data();
    buffers.buffer1 = scratch.data() + static_cast<size_t>(B) * H;
    buffers.buffer2 = scratch.data() + static_cast<size_t>(2) * B * H;
    buffers.buffer3 = scratch.data() + static_cast<size_t>(3) * B * H;

    const TfLiteStatus status =
        tflite::EvalLstm<Act, int8_t, int16_t, Bias>(op, content, buffers);
    return status == kTfLiteOk ? HCT_REF_OK : HCT_REF_E_PARAM;
}

} // namespace

extern "C" {

int32_t hct_ref_lstm_s8(const HctLstmParams *params,
                        const int8_t *input,
                        const int8_t *const *weights,
                        const int32_t *const *biases,
                        int8_t *output)
{
    return lstm<int8_t, int32_t>(params, input, weights, biases, output, kTfLiteInt8, kTfLiteInt32);
}

int32_t hct_ref_lstm_s16(const HctLstmParams *params,
                         const int16_t *input,
                         const int8_t *const *weights,
                         const int64_t *const *biases,
                         int16_t *output)
{
    return lstm<int16_t, int64_t>(params, input, weights, biases, output, kTfLiteInt16, kTfLiteInt64);
}

} // extern "C"
