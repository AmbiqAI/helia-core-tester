#ifndef FULLY_CONNECTED_FLOAT_DEFAULT_F32_HARNESS_H
#define FULLY_CONNECTED_FLOAT_DEFAULT_F32_HARNESS_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims fully_connected_float_default_f32_input_dims = {
    .n = 1,
    .h = 1,
    .w = 1,
    .c = 12
};

// Filter dimensions
static const cmsis_nn_dims fully_connected_float_default_f32_filter_dims = {
    .n = 12,
    .h = 1,
    .w = 1,
    .c = 5
};

// Bias dimensions
static const cmsis_nn_dims fully_connected_float_default_f32_bias_dims = {
    .n = 1,
    .h = 1,
    .w = 1,
    .c = 5
};

// Output dimensions
static const cmsis_nn_dims fully_connected_float_default_f32_output_dims = {
    .n = 1,
    .h = 1,
    .w = 1,
    .c = 5
};

// Fully connected parameters
static const cmsis_nn_fc_params_f32 fully_connected_float_default_f32_fc_params = {
    .activation = {.min = -1.0e+30f, .max = 1.0e+30f},
    .weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD
};

// Weights
static const float fully_connected_float_default_f32_weights[] = {
    -0.152828291f, 0.288920015f, -0.104245581f, -0.234368637f, 0.354279339f, 0.478141278f, 0.531910837f, 0.007800495f, 0.445037693f, 0.475066155f, 0.477185011f, -0.494910985f, 0.418134987f, -0.267046183f, 0.24938558f, -0.166729629f,
    -0.476274312f, 0.537248552f, -0.170705676f, 0.4919146f, 0.409184337f, 0.459042996f, -0.18113403f, 0.480753511f, -0.320282429f, 0.110006124f, 0.250927299f, 0.418677568f, -0.508032918f, -0.582383335f, 0.016635234f, 0.221276134f,
    0.361988187f, 0.559093297f, -0.412113249f, -0.282255054f, -0.074866943f, -0.546313822f, 0.006358361f, 0.061942071f, 0.396562785f, -1.494311448e-02f, -0.54507184f, 0.283090562f, -0.223425046f, 0.444775105f, -0.534828842f, -0.505919814f,
    -0.566144347f, -0.230825588f, -0.103041962f, 0.548389077f, 0.243397444f, 0.124327838f, -0.200564519f, -0.569545865f, -0.213492677f, -0.198955536f, -0.433979422f, -0.192043498f
};

// Biases
static const float fully_connected_float_default_f32_biases[] = {
    0.184678569f, -0.198930889f, -0.193914279f, 0.159600317f, -0.092495956f
};

// Input data (for testing)
static const float fully_connected_float_default_f32_input[] = {
    -0.616488993f, -0.503400087f, 0.451576054f, -0.224444553f, 0.103615589f, 0.864267349f, -0.525916755f, 0.370993018f, -0.296346396f, -0.629014671f, -0.98044914f, -0.001461412f
};

// Expected output (golden)
static const float fully_connected_float_default_f32_expected_output[] = {
    -0.585756302f, 0.281898618f, -0.56961298f, 1.201153278f, 0.844153285f
};

#endif