#ifndef FULLY_CONNECTED_FLOAT_DEFAULT_F32_FULLY_CONNECTED_H
#define FULLY_CONNECTED_FLOAT_DEFAULT_F32_FULLY_CONNECTED_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
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
    0.287148476f, -0.342412829f, -0.026528358f, -0.272790104f, 0.510272861f, 0.150348067f, 0.467027664f, -0.188357174f, -0.520859063f, -0.51187259f, 0.390116215f, 0.430709243f, 0.560541034f, -0.03737545f, 0.084123135f, 0.140521646f,
    0.347124994f, 0.122952104f, 0.545341015f, -0.418229401f, -0.110372066f, -0.005901337f, -0.254301608f, 0.542200327f, -0.352578461f, 0.579705358f, -0.235002786f, -0.061338961f, 0.212362766f, -0.35408622f, -0.119413614f, 0.294668496f,
    0.585405469f, 0.505181909f, 0.518861055f, 0.258029878f, 0.339790463f, 0.56861949f, 0.368746042f, 0.391123712f, -0.232502669f, 0.247765243f, 0.560053587f, -0.173419505f, 0.052180946f, -0.027092934f, 0.182569146f, 0.366213918f,
    -0.095229f, -0.336113453f, -0.098455459f, 0.308493018f, 0.517726064f, -0.034999371f, 0.044348478f, -0.016274333f, 0.040222168f, 0.09273994f, -0.24884218f, -0.257869989f
};

// Biases. Emitted whenever bias data exists, including when the kernel is
// called with a NULL bias because the bias is folded into the weight sum:
// consumers read the bias out of this decl.
static const float fully_connected_float_default_f32_biases[] = {
    -0.128519893f, 0.028359354f, -0.226798117f, 0.017792165f, -0.151638269f
};


// Input data (for testing)
static const float fully_connected_float_default_f32_input[] = {
    -0.616488993f, -0.503400087f, 0.451576054f, -0.224444553f, 0.103615589f, 0.864267349f, -0.525916755f, 0.370993018f, -0.296346396f, -0.629014671f, -0.98044914f, -0.001461412f
};

// Expected output (golden)
static const float fully_connected_float_default_f32_expected_output[] = {
    -0.123398639f, -0.306719899f, -1.50585866f, -0.745988011f, 0.130703077f
};

#endif