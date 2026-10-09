/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stand-in for TFLM's micro/kernels/kernel_util.h, pulled in by the vendored
 * micro/kernels/lstm_eval.{h,cc}. The shim drives EvalLstm directly over
 * TfLiteEvalTensors it owns, so only the tensor accessors are real. The micro
 * context lstm_eval.cc's LstmTensors would use is never constructed: its
 * methods return nothing, and nothing else of tensorflow/lite/micro is vendored.
 */
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_UTIL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_UTIL_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"

namespace tflite {

class MicroContext
{
  public:
    TfLiteTensor *AllocateTempInputTensor(const TfLiteNode *, int)
    {
        return nullptr;
    }
    TfLiteTensor *AllocateTempOutputTensor(const TfLiteNode *, int)
    {
        return nullptr;
    }
    void DeallocateTempTfLiteTensor(TfLiteTensor *) {}
};

inline MicroContext *GetMicroContext(const TfLiteContext *)
{
    return nullptr;
}

namespace micro {

template <typename T> T *GetTensorData(TfLiteEvalTensor *tensor)
{
    TFLITE_DCHECK(tensor != nullptr);
    return reinterpret_cast<T *>(tensor->data.raw);
}

template <typename T> const T *GetTensorData(const TfLiteEvalTensor *tensor)
{
    TFLITE_DCHECK(tensor != nullptr);
    return reinterpret_cast<const T *>(tensor->data.raw);
}

template <typename T> T *GetOptionalTensorData(TfLiteEvalTensor *tensor)
{
    return tensor == nullptr ? nullptr : reinterpret_cast<T *>(tensor->data.raw);
}

template <typename T> const T *GetOptionalTensorData(const TfLiteEvalTensor *tensor)
{
    return tensor == nullptr ? nullptr : reinterpret_cast<const T *>(tensor->data.raw);
}

inline const RuntimeShape GetTensorShape(const TfLiteEvalTensor *tensor)
{
    if (tensor == nullptr || tensor->dims == nullptr)
    {
        return RuntimeShape();
    }
    return RuntimeShape(tensor->dims->size, reinterpret_cast<const int32_t *>(tensor->dims->data));
}

} // namespace micro
} // namespace tflite

#endif
