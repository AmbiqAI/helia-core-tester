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
    0.574798822f, 0.135854602f, 0.168082893f, 0.396303356f, -0.253218502f, 0.096061677f, -0.049131304f, -0.542287707f, 0.222916782f, 0.37835747f, -0.462626845f, -0.373373538f, -0.254275978f, -0.543847203f, 0.229503155f, -0.322582304f,
    0.398952186f, -0.574450195f, 0.198953807f, 0.061468977f, 0.407894135f, -0.324926525f, -0.162387565f, 0.47390455f, 0.513452053f, 0.493311822f, 0.142925322f, 0.541062415f, 0.275119066f, -0.081021108f, -0.591278255f, -0.578530788f,
    0.445388943f, -0.475982636f, 0.139583051f, -0.125888392f, 0.451745719f, 0.198738828f, 0.190689206f, -0.023054685f, -0.458765805f, -0.481220126f, -0.046975367f, -0.432054579f, 0.500612617f, -0.022806503f, 0.339775056f, -0.484891474f,
    0.221702307f, -0.347536594f, -0.374843448f, -0.463659644f, -0.163685665f, 0.094202712f, 0.386695117f, 0.002663851f, -0.209291726f, 0.553456128f, -0.232319966f, -0.24841921f
};

// Biases
static const float fully_connected_float_default_f32_biases[] = {
    0.152862832f, 0.177858815f, -0.15475212f, 1.586030266e-04f, 0.086951159f
};

// Input data (for testing)
static const float fully_connected_float_default_f32_input[] = {
    -0.616488993f, -0.503400087f, 0.451576054f, -0.224444553f, 0.103615589f, 0.864267349f, -0.525916755f, 0.370993018f, -0.296346396f, -0.629014671f, -0.98044914f, -0.001461412f
};

// Expected output (golden)
static const float fully_connected_float_default_f32_expected_output[] = {
    -0.251415074f, 0.489486158f, -0.690964222f, -1.352552533f, -0.135873809f
};

#endif