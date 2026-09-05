#ifndef BATCH_MATMUL_FLOAT_DEFAULT_F32_MATMUL_BATCH_H
#define BATCH_MATMUL_FLOAT_DEFAULT_F32_MATMUL_BATCH_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input LHS dimensions
static const cmsis_nn_dims batch_matmul_float_default_f32_input_lhs_dims = {
    .n = 1, .h = 1,
    .w = 3, .c = 4
};

// Input RHS dimensions
static const cmsis_nn_dims batch_matmul_float_default_f32_input_rhs_dims = {
    .n = 1, .h = 1,
    .w = 3, .c = 2
};

// Output dimensions
static const cmsis_nn_dims batch_matmul_float_default_f32_output_dims = {
    .n = 1, .h = 1,
    .w = 4, .c = 2
};

// Batch matmul parameters
static const cmsis_nn_bmm_params_f32 batch_matmul_float_default_f32_bmm_params = {
    .adj_x = false,
    .adj_y = true,
    .activation = {
        .min = -1.0e+30f,
        .max = 1.0e+30f
    },
    .rhs_format = ARM_NN_WEIGHT_FORMAT_STANDARD
};


// Input LHS data (for testing)
static const float batch_matmul_float_default_f32_input_lhs[] = {
    -0.790458202f, -0.821159065f, -0.329297036f, 0.790937662f, 0.951711059f, 0.579429448f, 0.363695621f, 0.795291424f, -0.305035621f, -0.728619695f, -0.075332284f, 0.876095355f
};

// Input RHS data (for testing)
static const float batch_matmul_float_default_f32_input_rhs[] = {
    -0.39882797f, 0.208587572f, 0.667111039f, -0.053960856f, -0.819526732f, 0.608264506f
};

// Expected output (golden)
static const float batch_matmul_float_default_f32_expected_output[] = {
    0.037320275f, -0.320868999f, -0.155409038f, 0.46607098f, 0.635480523f, -0.152594566f, -0.477644652f, 0.384981692f
};

#endif