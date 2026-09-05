#ifndef FULLY_CONNECTED_FLOAT_DEFAULT_F32_FULLY_CONNECTED_H
#define FULLY_CONNECTED_FLOAT_DEFAULT_F32_FULLY_CONNECTED_H

#include <stdint.h>
// Golden arrays may carry NAN/INFINITY, and this header is included ahead of any
// other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions: input shape is [N, features] or [N, H, W, C] (flattened to [N, features])
static const cmsis_nn_dims fully_connected_float_default_f32_input_dims = {
    .n = 1, .h = 1,
    .w = 1, .c = 12
};

// Filter dimensions: weights shape is [output_units, input_features]
// CMSIS format: n=input_features (col_dim), c=output_units (row_dim), h=1, w=1
static const cmsis_nn_dims fully_connected_float_default_f32_filter_dims = {
    .n = 12, .h = 1,
    .w = 1, .c = 5
};

// Bias dimensions: bias shape is [output_units]
static const cmsis_nn_dims fully_connected_float_default_f32_bias_dims = {
    .n = 1, .h = 1,
    .w = 1, .c = 5
};

// Output dimensions: output shape is [N, output_units]
static const cmsis_nn_dims fully_connected_float_default_f32_output_dims = {
    .n = 1, .h = 1,
    .w = 1, .c = 5
};

// Fully connected parameters
static const cmsis_nn_fc_params_f32 fully_connected_float_default_f32_fc_params = {
    .activation = {.min = -1.0e+30f, .max = 1.0e+30f},
    .weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD
};


// Weights
static const float fully_connected_float_default_f32_weights[] = {
    -0.473706365f, 0.546016574f, 0.14284724f, 0.259765327f, 0.223671138f, 0.329569042f, 0.327856302f, -0.102112502f, -0.091846168f, 0.052331269f, -0.546234488f, -0.21741271f, -0.443819404f, 0.219174623f, -0.129366517f, -0.298815638f,
    -0.217593163f, -0.019773185f, 0.086775839f, 0.248784542f, 0.103772998f, 0.004890203f, -0.465551078f, -0.065797269f, 0.36142242f, -0.385493457f, 0.347569585f, 0.100618303f, 0.553967834f, 0.546589255f, -0.017306209f, -0.160514534f,
    -0.446270376f, 0.118290126f, 0.153360784f, 0.537892938f, 0.240467727f, -0.202074915f, 0.362178802f, 0.034498394f, -0.221545249f, -0.131926686f, 0.07861048f, -0.411204278f, 0.449667692f, -0.156533808f, -0.479073763f, 0.108982146f,
    -0.068009019f, 0.045535386f, -0.267778546f, -0.056851149f, -0.016631305f, 0.253105462f, 0.395894349f, -0.128010988f, 0.165942252f, 0.467531562f, -0.126936346f, -0.008867621f
};

// Biases
static const float fully_connected_float_default_f32_biases[] = {
    -0.128519893f, 0.028359354f, -0.226798117f, 0.017792165f, -0.151638269f
};


// Input data (for testing)
static const float fully_connected_float_default_f32_input[] = {
    -0.616488993f, -0.503400087f, 0.451576054f, -0.224444553f, 0.103615589f, 0.864267349f, -0.525916755f, 0.370993018f, -0.296346396f, -0.629014671f, -0.98044914f, -0.001461412f
};

// Expected output (golden)
static const float fully_connected_float_default_f32_expected_output[] = {
    0.522731006f, 0.630026996f, 0.264863849f, 0.230960205f, -0.498260587f
};

#endif