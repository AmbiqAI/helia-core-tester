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
    (float16_t)-0.561035156f, (float16_t)-0.055908203f, (float16_t)-0.297851562f, (float16_t)0.559570312f, (float16_t)-0.373535156f, (float16_t)-0.448242188f, (float16_t)0.580566406f, (float16_t)0.489990234f, (float16_t)-0.493896484f, (float16_t)0.017593384f, (float16_t)0.349121094f, (float16_t)0.590820312f, (float16_t)-6.649017334e-03f, (float16_t)-0.201904297f, (float16_t)0.165527344f, (float16_t)-0.16809082f,
    (float16_t)-0.194213867f, (float16_t)-0.321289062f, (float16_t)0.060791016f, (float16_t)0.249267578f, (float16_t)0.261474609f, (float16_t)-0.169799805f, (float16_t)-0.190917969f, (float16_t)0.135620117f, (float16_t)-0.01574707f, (float16_t)0.486572266f, (float16_t)-0.317871094f, (float16_t)-0.298828125f, (float16_t)0.515136719f, (float16_t)-0.561035156f, (float16_t)-0.120117188f, (float16_t)-0.509765625f,
    (float16_t)-0.06463623f, (float16_t)-0.436035156f, (float16_t)0.188598633f, (float16_t)0.308837891f, (float16_t)0.522460938f, (float16_t)-0.160400391f, (float16_t)-0.575195312f, (float16_t)-0.414550781f, (float16_t)0.345458984f, (float16_t)-0.257080078f, (float16_t)0.180419922f, (float16_t)0.286376953f, (float16_t)-0.160522461f, (float16_t)-0.497314453f, (float16_t)-0.185302734f, (float16_t)0.179321289f,
    (float16_t)0.575195312f, (float16_t)0.395263672f, (float16_t)0.447021484f, (float16_t)0.059143066f, (float16_t)-0.486816406f, (float16_t)0.424072266f, (float16_t)0.078125f, (float16_t)-0.154663086f, (float16_t)-0.09375f, (float16_t)0.459960938f, (float16_t)-0.536621094f, (float16_t)-0.015472412f
};

// Biases
static const float16_t fully_connected_float_default_f16_biases[] = {
    (float16_t)-0.003479004f, (float16_t)-0.027450562f, (float16_t)0.10760498f, (float16_t)0.244995117f, (float16_t)-0.12310791f
};

// Input data (for testing)
static const float16_t fully_connected_float_default_f16_input[] = {
    (float16_t)0.506835938f, (float16_t)0.921875f, (float16_t)0.248779297f, (float16_t)0.33203125f, (float16_t)-0.759765625f, (float16_t)0.065246582f, (float16_t)0.7265625f, (float16_t)-0.000869751f, (float16_t)-0.673339844f, (float16_t)-0.840332031f, (float16_t)-0.015823364f, (float16_t)-0.926269531f
};

// Expected output (golden)
static const float16_t fully_connected_float_default_f16_expected_output[] = {
    (float16_t)0.213256836f, (float16_t)-0.217041016f, (float16_t)-0.024047852f, (float16_t)0.295654297f, (float16_t)0.817382812f
};

#endif