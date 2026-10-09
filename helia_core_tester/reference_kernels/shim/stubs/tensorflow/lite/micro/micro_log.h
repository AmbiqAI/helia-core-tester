/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stand-in for TFLM's micro_log.h (pulled in by kernels/op_macros.h): the host
 * reference build strips every log string, so only the no-op forms remain and
 * nothing from tensorflow/lite/micro is vendored.
 */
#ifndef TENSORFLOW_LITE_MICRO_MICRO_LOG_H_
#define TENSORFLOW_LITE_MICRO_MICRO_LOG_H_

#ifndef TF_LITE_STRIP_ERROR_STRINGS
#error "hct_ref builds with TF_LITE_STRIP_ERROR_STRINGS; this stub has no printing back end"
#endif

namespace tflite {

template <typename... Args> void Unused(Args &&...args)
{
    (void)(sizeof...(args));
}

template <typename T, typename... Args> T Unused(Args &&...args)
{
    (void)(sizeof...(args));
    return static_cast<T>(0);
}

} // namespace tflite

#define MicroPrintf(...) tflite::Unused(__VA_ARGS__)
#define VMicroPrintf(...) tflite::Unused(__VA_ARGS__)
#define MicroSnprintf(...) tflite::Unused<int>(__VA_ARGS__)
#define MicroVsnprintf(...) tflite::Unused<int>(__VA_ARGS__)

#endif
