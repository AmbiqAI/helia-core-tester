/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Elementwise binary arithmetic entries of the hct_ref shim (see hct_ref.h),
 * dispatched exactly as TFLM's micro add/sub/mul kernels do.
 */
#include <cstdint>

#include "hct_ref.h"
#include "hct_ref_internal.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace {

using namespace hct;
using tflite::RuntimeShape;

enum class BinaryOp
{
    kAdd,
    kSub,
    kMul
};

// The output shape must be the broadcast of the inputs' shapes (right-aligned,
// every dimension equal or 1).
int32_t check_broadcast(const HctShape *a, const HctShape *b, const HctShape *out)
{
    HCT_TRY(check_shape(a, 0));
    HCT_TRY(check_shape(b, 0));
    HCT_TRY(check_shape(out, 0));
    if (a->rank > 4 || b->rank > 4 || out->rank > 4)
    {
        return HCT_REF_E_UNSUPPORTED;
    }
    const int32_t rank = out->rank;
    if (a->rank > rank || b->rank > rank)
    {
        return HCT_REF_E_DIMS;
    }
    for (int32_t i = 0; i < rank; ++i)
    {
        const int32_t ia = i - (rank - a->rank);
        const int32_t ib = i - (rank - b->rank);
        const int32_t da = ia >= 0 ? a->dims[ia] : 1;
        const int32_t db = ib >= 0 ? b->dims[ib] : 1;
        const int32_t expected = da == 1 ? db : da;
        if ((da != 1 && db != 1 && da != db) || out->dims[i] != expected)
        {
            return HCT_REF_E_DIMS;
        }
    }
    return HCT_REF_OK;
}

template <typename T> int32_t check_binary_offsets(const HctBinaryParams *p, BinaryOp op)
{
    if (sizeof(T) == 2)
    {
        return (p->input1_offset == 0 && p->input2_offset == 0 && p->output_offset == 0) ? HCT_REF_OK
                                                                                          : HCT_REF_E_PARAM;
    }
    const bool in_ok = p->input1_offset >= -127 && p->input1_offset <= 128 && p->input2_offset >= -127 &&
                       p->input2_offset <= 128;
    const bool out_ok = p->output_offset >= -128 && p->output_offset <= 127;
    if (!in_ok || !out_ok)
    {
        return HCT_REF_E_PARAM;
    }
    if (op != BinaryOp::kMul && (p->left_shift < 0 || p->left_shift > 20))
    {
        return HCT_REF_E_PARAM;
    }
    return HCT_REF_OK;
}

int32_t check_multiplier(int32_t multiplier, int32_t shift)
{
    return (multiplier >= 0 && shift >= -31 && shift <= 31) ? HCT_REF_OK : HCT_REF_E_PARAM;
}

tflite::ArithmeticParams to_arithmetic(const HctBinaryParams *p)
{
    tflite::ArithmeticParams params = {};
    params.left_shift = p->left_shift;
    params.input1_offset = p->input1_offset;
    params.input1_multiplier = p->input1_multiplier;
    params.input1_shift = p->input1_shift;
    params.input2_offset = p->input2_offset;
    params.input2_multiplier = p->input2_multiplier;
    params.input2_shift = p->input2_shift;
    params.output_offset = p->output_offset;
    params.output_multiplier = p->output_multiplier;
    params.output_shift = p->output_shift;
    params.quantized_activation_min = p->act.min;
    params.quantized_activation_max = p->act.max;
    return params;
}

template <typename T>
int32_t binary(BinaryOp op,
               const HctBinaryParams *p,
               const HctShape *s1,
               const T *in1,
               const HctShape *s2,
               const T *in2,
               const HctShape *so,
               T *out)
{
    if (p == nullptr)
    {
        return HCT_REF_E_NULL;
    }
    HCT_TRY(check_broadcast(s1, s2, so));
    HCT_TRY(check_buffers(in1, in2, out));
    HCT_TRY(check_activation<T>(p->act));
    HCT_TRY(check_binary_offsets<T>(p, op));
    HCT_TRY(check_multiplier(p->output_multiplier, p->output_shift));
    if (op != BinaryOp::kMul)
    {
        HCT_TRY(check_multiplier(p->input1_multiplier, p->input1_shift));
        HCT_TRY(check_multiplier(p->input2_multiplier, p->input2_shift));
    }
    tflite::ArithmeticParams params = to_arithmetic(p);
    const RuntimeShape a = to_runtime(s1), b = to_runtime(s2), o = to_runtime(so);
    const bool broadcast = tflite::reference_ops::ProcessBroadcastShapes(a, b, &params);
    switch (op)
    {
    case BinaryOp::kAdd:
        if (sizeof(T) == 1)
        {
            if (broadcast)
                tflite::reference_integer_ops::BroadcastAdd4DSlow(params, a, reinterpret_cast<const int8_t *>(in1), b,
                                                                  reinterpret_cast<const int8_t *>(in2), o,
                                                                  reinterpret_cast<int8_t *>(out));
            else
                tflite::reference_integer_ops::Add(params, a, reinterpret_cast<const int8_t *>(in1), b,
                                                   reinterpret_cast<const int8_t *>(in2), o,
                                                   reinterpret_cast<int8_t *>(out));
        }
        else
        {
            if (broadcast)
                tflite::reference_ops::BroadcastAdd4DSlow(params, a, reinterpret_cast<const int16_t *>(in1), b,
                                                          reinterpret_cast<const int16_t *>(in2), o,
                                                          reinterpret_cast<int16_t *>(out));
            else
                tflite::reference_ops::Add(params, a, reinterpret_cast<const int16_t *>(in1), b,
                                           reinterpret_cast<const int16_t *>(in2), o,
                                           reinterpret_cast<int16_t *>(out), false);
        }
        break;
    case BinaryOp::kSub:
        if (broadcast)
            tflite::reference_ops::BroadcastQuantSubSlow(params, a, in1, b, in2, o, out);
        else
            tflite::reference_ops::Sub(params, a, in1, b, in2, o, out);
        break;
    case BinaryOp::kMul:
        if (broadcast)
            tflite::reference_integer_ops::BroadcastMul4DSlow(params, a, in1, b, in2, o, out);
        else
            tflite::reference_integer_ops::Mul(params, a, in1, b, in2, o, out);
        break;
    }
    return HCT_REF_OK;
}

} // namespace

extern "C" {

int32_t hct_ref_add_s8(const HctBinaryParams *p,
                       const HctShape *s1,
                       const int8_t *in1,
                       const HctShape *s2,
                       const int8_t *in2,
                       const HctShape *so,
                       int8_t *out)
{
    return binary<int8_t>(BinaryOp::kAdd, p, s1, in1, s2, in2, so, out);
}

int32_t hct_ref_add_s16(const HctBinaryParams *p,
                        const HctShape *s1,
                        const int16_t *in1,
                        const HctShape *s2,
                        const int16_t *in2,
                        const HctShape *so,
                        int16_t *out)
{
    return binary<int16_t>(BinaryOp::kAdd, p, s1, in1, s2, in2, so, out);
}

int32_t hct_ref_sub_s8(const HctBinaryParams *p,
                       const HctShape *s1,
                       const int8_t *in1,
                       const HctShape *s2,
                       const int8_t *in2,
                       const HctShape *so,
                       int8_t *out)
{
    return binary<int8_t>(BinaryOp::kSub, p, s1, in1, s2, in2, so, out);
}

int32_t hct_ref_sub_s16(const HctBinaryParams *p,
                        const HctShape *s1,
                        const int16_t *in1,
                        const HctShape *s2,
                        const int16_t *in2,
                        const HctShape *so,
                        int16_t *out)
{
    return binary<int16_t>(BinaryOp::kSub, p, s1, in1, s2, in2, so, out);
}

int32_t hct_ref_mul_s8(const HctBinaryParams *p,
                       const HctShape *s1,
                       const int8_t *in1,
                       const HctShape *s2,
                       const int8_t *in2,
                       const HctShape *so,
                       int8_t *out)
{
    return binary<int8_t>(BinaryOp::kMul, p, s1, in1, s2, in2, so, out);
}

int32_t hct_ref_mul_s16(const HctBinaryParams *p,
                        const HctShape *s1,
                        const int16_t *in1,
                        const HctShape *s2,
                        const int16_t *in2,
                        const HctShape *so,
                        int16_t *out)
{
    return binary<int16_t>(BinaryOp::kMul, p, s1, in1, s2, in2, so, out);
}

} // extern "C"
