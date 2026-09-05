#ifndef SOFTMAX_FLOAT_DEFAULT_F32_SOFTMAX_H
#define SOFTMAX_FLOAT_DEFAULT_F32_SOFTMAX_H

#include <stdint.h>
// Golden arrays may carry NAN/INFINITY, and this header is included ahead of any
// other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims softmax_float_default_f32_input_dims = {
    .n = 1, .h = 4,
    .w = 4, .c = 3
};

// Output dimensions
static const cmsis_nn_dims softmax_float_default_f32_output_dims = {
    .n = 1, .h = 4,
    .w = 4, .c = 3
};

// Input data (for testing)
static const float softmax_float_default_f32_input[] = {
    -0.189511791f, 0.220630303f, -0.994163513f, 0.406811386f, 0.886255264f, 0.985261977f, 0.167530119f, 0.056889545f, -0.219931617f, -0.208892226f, 0.420646787f, 0.261283129f, 0.250427216f, 0.598819852f, -0.64019984f, -0.371896595f,
    -0.481595665f, 0.645454466f, 0.864155233f, 0.994167864f, -0.157216415f, -0.182117999f, -0.872848988f, 0.72147423f, 0.374576688f, -0.316026539f, 0.075343594f, 0.611457288f, 0.711894274f, -0.245280847f, -0.345137864f, -0.810523987f,
    0.194442898f, 0.738222778f, -0.513788879f, 0.649517775f, -0.379350394f, 0.159976035f, -0.976656795f, -0.498208433f, 0.529953361f, 0.441719562f, 0.254784554f, -0.322555453f, -0.531378865f, 0.025891691f, 0.268367469f, -0.597357094f,
};

// Expected output (golden)
static const float softmax_float_default_f32_expected_output[] = {
    0.338492483f, 0.510118961f, 0.151388615f, 0.227352872f, 0.367214859f, 0.405432284f, 0.388494641f, 0.347803891f, 0.263701469f, 0.223362878f, 0.419195175f, 0.357441962f, 0.353708476f, 0.501130104f, 0.145161375f, 0.214501947f,
    0.192216009f, 0.593282044f, 0.400169104f, 0.455729693f, 0.144101158f, 0.251910508f, 0.126259953f, 0.621829569f, 0.44589901f, 0.223517388f, 0.330583543f, 0.395225912f, 0.436983079f, 0.167791039f, 0.299116403f, 0.187812984f,
    0.513070643f, 0.454329729f, 0.129906058f, 0.415764183f, 0.306265295f, 0.525199473f, 0.168535307f, 0.157338634f, 0.439906448f, 0.402754933f, 0.495790988f, 0.278331935f, 0.225877076f, 0.35579592f, 0.45342645f, 0.190777645f,
};


#endif