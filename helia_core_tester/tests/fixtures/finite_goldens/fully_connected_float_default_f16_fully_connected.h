#ifndef FULLY_CONNECTED_FLOAT_DEFAULT_F16_FULLY_CONNECTED_H
#define FULLY_CONNECTED_FLOAT_DEFAULT_F16_FULLY_CONNECTED_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions: input shape is [N, features] or [N, H, W, C] (flattened to [N, features])
static const cmsis_nn_dims fully_connected_float_default_f16_input_dims = {
    .n = 1, .h = 1,
    .w = 1, .c = 12
};

// Filter dimensions: weights shape is [output_units, input_features]
// CMSIS format: n=input_features (col_dim), c=output_units (row_dim), h=1, w=1
static const cmsis_nn_dims fully_connected_float_default_f16_filter_dims = {
    .n = 12, .h = 1,
    .w = 1, .c = 5
};

// Bias dimensions: bias shape is [output_units]
static const cmsis_nn_dims fully_connected_float_default_f16_bias_dims = {
    .n = 1, .h = 1,
    .w = 1, .c = 5
};

// Output dimensions: output shape is [N, output_units]
static const cmsis_nn_dims fully_connected_float_default_f16_output_dims = {
    .n = 1, .h = 1,
    .w = 1, .c = 5
};

// Fully connected parameters
static const cmsis_nn_fc_params_f16 fully_connected_float_default_f16_fc_params = {
    .activation = {.min = -1.0e+30f, .max = 1.0e+30f},
    .weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD
};


// Weights
static const float16_t fully_connected_float_default_f16_weights[] = {
    (float16_t)0.287109375f, (float16_t)-0.342529297f, (float16_t)-0.026535034f, (float16_t)-0.272705078f, (float16_t)0.510253906f, (float16_t)0.150390625f, (float16_t)0.467041016f, (float16_t)-0.188354492f, (float16_t)-0.520996094f, (float16_t)-0.51171875f, (float16_t)0.390136719f, (float16_t)0.430664062f, (float16_t)0.560546875f, (float16_t)-0.037384033f, (float16_t)0.084106445f, (float16_t)0.14050293f,
    (float16_t)0.347167969f, (float16_t)0.122924805f, (float16_t)0.545410156f, (float16_t)-0.418212891f, (float16_t)-0.110351562f, (float16_t)-0.005901337f, (float16_t)-0.254394531f, (float16_t)0.541992188f, (float16_t)-0.352539062f, (float16_t)0.579589844f, (float16_t)-0.234985352f, (float16_t)-0.061340332f, (float16_t)0.212402344f, (float16_t)-0.354003906f, (float16_t)-0.119384766f, (float16_t)0.294677734f,
    (float16_t)0.585449219f, (float16_t)0.505371094f, (float16_t)0.519042969f, (float16_t)0.258056641f, (float16_t)0.33984375f, (float16_t)0.568847656f, (float16_t)0.368652344f, (float16_t)0.391113281f, (float16_t)-0.232543945f, (float16_t)0.247802734f, (float16_t)0.560058594f, (float16_t)-0.173461914f, (float16_t)0.052185059f, (float16_t)-0.027099609f, (float16_t)0.182617188f, (float16_t)0.366210938f,
    (float16_t)-0.095214844f, (float16_t)-0.336181641f, (float16_t)-0.098449707f, (float16_t)0.30859375f, (float16_t)0.517578125f, (float16_t)-0.035003662f, (float16_t)0.044342041f, (float16_t)-0.016281128f, (float16_t)0.040222168f, (float16_t)0.092712402f, (float16_t)-0.248901367f, (float16_t)-0.2578125f
};

// Biases. Emitted whenever bias data exists, including when the kernel is
// called with a NULL bias because the bias is folded into the weight sum:
// consumers read the bias out of this decl.
static const float16_t fully_connected_float_default_f16_biases[] = {
    (float16_t)0.080871582f, (float16_t)-0.224975586f, (float16_t)0.219482422f, (float16_t)0.21472168f, (float16_t)0.221557617f
};


// Input data (for testing)
static const float16_t fully_connected_float_default_f16_input[] = {
    (float16_t)0.506835938f, (float16_t)0.921875f, (float16_t)0.248779297f, (float16_t)0.33203125f, (float16_t)-0.759765625f, (float16_t)0.065246582f, (float16_t)0.7265625f, (float16_t)-0.000869751f, (float16_t)-0.673339844f, (float16_t)-0.840332031f, (float16_t)-0.015823364f, (float16_t)-0.926269531f
};

// Expected output (golden)
static const float16_t fully_connected_float_default_f16_expected_output[] = {
    (float16_t)0.150878906f, (float16_t)-0.185791016f, (float16_t)-0.840820312f, (float16_t)1.377929688f, (float16_t)-0.284179688f
};

#endif