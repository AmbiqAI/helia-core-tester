#ifndef FULLY_CONNECTED_FLOAT_DEFAULT_F16_FULLY_CONNECTED_H
#define FULLY_CONNECTED_FLOAT_DEFAULT_F16_FULLY_CONNECTED_H

#include <stdint.h>
// Golden arrays may carry NAN/INFINITY, and this header is included ahead of any
// other translation-unit include that would define them.
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
    (float16_t)-0.045379639f, (float16_t)-0.233032227f, (float16_t)0.062103271f, (float16_t)-0.12286377f, (float16_t)0.075561523f, (float16_t)0.400146484f, (float16_t)-0.213623047f, (float16_t)-0.55078125f, (float16_t)0.394042969f, (float16_t)-0.076049805f, (float16_t)-0.126464844f, (float16_t)0.278076172f, (float16_t)-0.12298584f, (float16_t)0.17980957f, (float16_t)0.044128418f, (float16_t)0.274414062f,
    (float16_t)-0.511230469f, (float16_t)-0.104553223f, (float16_t)0.454345703f, (float16_t)0.019256592f, (float16_t)0.072143555f, (float16_t)0.273681641f, (float16_t)0.076904297f, (float16_t)0.112487793f, (float16_t)-0.102172852f, (float16_t)-0.354980469f, (float16_t)0.391845703f, (float16_t)0.019638062f, (float16_t)0.141723633f, (float16_t)-0.219482422f, (float16_t)-0.540039062f, (float16_t)-0.193481445f,
    (float16_t)-0.100341797f, (float16_t)-0.139038086f, (float16_t)-0.082641602f, (float16_t)-0.185058594f, (float16_t)-0.225097656f, (float16_t)0.013877869f, (float16_t)0.331054688f, (float16_t)0.011940002f, (float16_t)0.373535156f, (float16_t)0.107299805f, (float16_t)0.162109375f, (float16_t)0.035949707f, (float16_t)-0.172973633f, (float16_t)0.501464844f, (float16_t)0.317138672f, (float16_t)0.483154297f,
    (float16_t)-0.210327148f, (float16_t)0.017288208f, (float16_t)-0.5625f, (float16_t)0.020462036f, (float16_t)-0.192871094f, (float16_t)-0.574707031f, (float16_t)0.129638672f, (float16_t)0.552246094f, (float16_t)-0.477783203f, (float16_t)0.314941406f, (float16_t)0.436767578f, (float16_t)-0.042572021f
};

// Biases
static const float16_t fully_connected_float_default_f16_biases[] = {
    (float16_t)0.080871582f, (float16_t)-0.224975586f, (float16_t)0.219482422f, (float16_t)0.21472168f, (float16_t)0.221557617f
};


// Input data (for testing)
static const float16_t fully_connected_float_default_f16_input[] = {
    (float16_t)0.506835938f, (float16_t)0.921875f, (float16_t)0.248779297f, (float16_t)0.33203125f, (float16_t)-0.759765625f, (float16_t)0.065246582f, (float16_t)0.7265625f, (float16_t)-0.000869751f, (float16_t)-0.673339844f, (float16_t)-0.840332031f, (float16_t)-0.015823364f, (float16_t)-0.926269531f
};

// Expected output (golden)
static const float16_t fully_connected_float_default_f16_expected_output[] = {
    (float16_t)-0.825195312f, (float16_t)0.308349609f, (float16_t)-0.212768555f, (float16_t)-0.716308594f, (float16_t)0.289794922f
};

#endif