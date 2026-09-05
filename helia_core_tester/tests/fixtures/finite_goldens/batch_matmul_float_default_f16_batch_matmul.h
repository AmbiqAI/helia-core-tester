#ifndef BATCH_MATMUL_FLOAT_DEFAULT_F16_MATMUL_BATCH_H
#define BATCH_MATMUL_FLOAT_DEFAULT_F16_MATMUL_BATCH_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input LHS dimensions
static const cmsis_nn_dims batch_matmul_float_default_f16_input_lhs_dims = {
    .n = 1, .h = 1,
    .w = 3, .c = 4
};

// Input RHS dimensions
static const cmsis_nn_dims batch_matmul_float_default_f16_input_rhs_dims = {
    .n = 1, .h = 1,
    .w = 3, .c = 2
};

// Output dimensions
static const cmsis_nn_dims batch_matmul_float_default_f16_output_dims = {
    .n = 1, .h = 1,
    .w = 4, .c = 2
};

// Batch matmul parameters
static const cmsis_nn_bmm_params_f16 batch_matmul_float_default_f16_bmm_params = {
    .adj_x = false,
    .adj_y = true,
    .activation = {
        .min = -1.0e+30f,
        .max = 1.0e+30f
    },
    .rhs_format = ARM_NN_WEIGHT_FORMAT_STANDARD
};


// Input LHS data (for testing)
static const float16_t batch_matmul_float_default_f16_input_lhs[] = {
    (float16_t)-0.495117188f, (float16_t)0.647949219f, (float16_t)-0.516113281f, (float16_t)0.324462891f, (float16_t)-0.060394287f, (float16_t)-0.091369629f, (float16_t)-0.444091797f, (float16_t)0.476318359f, (float16_t)0.719726562f, (float16_t)0.971191406f, (float16_t)-0.258544922f, (float16_t)0.290527344f
};

// Input RHS data (for testing)
static const float16_t batch_matmul_float_default_f16_input_rhs[] = {
    (float16_t)0.935058594f, (float16_t)-0.529785156f, (float16_t)-0.968261719f, (float16_t)-0.116027832f, (float16_t)-0.039489746f, (float16_t)0.617675781f
};

// Expected output (golden)
static const float16_t batch_matmul_float_default_f16_expected_output[] = {
    (float16_t)-1.0703125f, (float16_t)-0.131713867f, (float16_t)0.365478516f, (float16_t)-0.221313477f, (float16_t)-0.904785156f, (float16_t)0.624511719f, (float16_t)1.147460938f, (float16_t)-0.305175781f
};

#endif