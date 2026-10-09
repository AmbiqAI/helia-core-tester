#ifndef FULLY_CONNECTED_FLOAT_DEFAULT_F16_HARNESS_H
#define FULLY_CONNECTED_FLOAT_DEFAULT_F16_HARNESS_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims fully_connected_float_default_f16_input_dims = {
    .n = 1,
    .h = 1,
    .w = 1,
    .c = 12
};

// Filter dimensions
static const cmsis_nn_dims fully_connected_float_default_f16_filter_dims = {
    .n = 12,
    .h = 1,
    .w = 1,
    .c = 5
};

// Bias dimensions
static const cmsis_nn_dims fully_connected_float_default_f16_bias_dims = {
    .n = 1,
    .h = 1,
    .w = 1,
    .c = 5
};

// Output dimensions
static const cmsis_nn_dims fully_connected_float_default_f16_output_dims = {
    .n = 1,
    .h = 1,
    .w = 1,
    .c = 5
};

// Fully connected parameters
static const cmsis_nn_fc_params_f16 fully_connected_float_default_f16_fc_params = {
    .activation = {.min = -1.0e+30f, .max = 1.0e+30f},
    .weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD
};

// Weights
static const float16_t fully_connected_float_default_f16_weights[] = {
    (float16_t)-0.53125f, (float16_t)0.292724609f, (float16_t)-0.356445312f, (float16_t)-0.100402832f, (float16_t)-0.161010742f, (float16_t)-0.301025391f, (float16_t)0.187744141f, (float16_t)-0.50390625f, (float16_t)-0.504882812f, (float16_t)-0.282958984f, (float16_t)0.10357666f, (float16_t)0.254394531f, (float16_t)-0.199584961f, (float16_t)0.192504883f, (float16_t)-0.100891113f, (float16_t)0.322265625f,
    (float16_t)-0.381591797f, (float16_t)-0.338623047f, (float16_t)0.566894531f, (float16_t)0.287841797f, (float16_t)-0.323974609f, (float16_t)-0.221557617f, (float16_t)0.552734375f, (float16_t)0.217285156f, (float16_t)-0.496582031f, (float16_t)0.466064453f, (float16_t)0.384521484f, (float16_t)-0.366210938f, (float16_t)-0.51171875f, (float16_t)-0.499267578f, (float16_t)-0.080200195f, (float16_t)-0.361328125f,
    (float16_t)-0.3671875f, (float16_t)0.159057617f, (float16_t)-0.565917969f, (float16_t)-0.140136719f, (float16_t)-0.328613281f, (float16_t)0.217529297f, (float16_t)-0.579589844f, (float16_t)0.005592346f, (float16_t)-0.454101562f, (float16_t)0.423339844f, (float16_t)0.24987793f, (float16_t)-0.077331543f, (float16_t)0.339599609f, (float16_t)-0.190673828f, (float16_t)0.411865234f, (float16_t)-0.400390625f,
    (float16_t)-0.142822266f, (float16_t)0.418701172f, (float16_t)-0.146484375f, (float16_t)-0.579101562f, (float16_t)0.325927734f, (float16_t)-0.39453125f, (float16_t)-0.259765625f, (float16_t)-0.291503906f, (float16_t)-0.373535156f, (float16_t)0.157226562f, (float16_t)-0.404296875f, (float16_t)-0.344482422f
};

// Biases
static const float16_t fully_connected_float_default_f16_biases[] = {
    (float16_t)-0.070373535f, (float16_t)0.073059082f, (float16_t)0.081237793f, (float16_t)-0.065795898f, (float16_t)-0.028366089f
};

// Input data (for testing)
static const float16_t fully_connected_float_default_f16_input[] = {
    (float16_t)0.506835938f, (float16_t)0.921875f, (float16_t)0.248779297f, (float16_t)0.33203125f, (float16_t)-0.759765625f, (float16_t)0.065246582f, (float16_t)0.7265625f, (float16_t)-0.000869751f, (float16_t)-0.673339844f, (float16_t)-0.840332031f, (float16_t)-0.015823364f, (float16_t)-0.926269531f
};

// Expected output (golden)
static const float16_t fully_connected_float_default_f16_expected_output[] = {
    (float16_t)0.388183594f, (float16_t)1.10546875f, (float16_t)0.783691406f, (float16_t)0.67578125f, (float16_t)0.039550781f
};

#endif