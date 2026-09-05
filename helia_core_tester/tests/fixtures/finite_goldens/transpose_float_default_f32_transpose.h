#ifndef TRANSPOSE_FLOAT_DEFAULT_F32_TRANSPOSE_H
#define TRANSPOSE_FLOAT_DEFAULT_F32_TRANSPOSE_H

#include <stdint.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims transpose_float_default_f32_input_dims = {
    .n = 1, .h = 2,
    .w = 3, .c = 4
};

// Output dimensions
static const cmsis_nn_dims transpose_float_default_f32_output_dims = {
    .n = 1, .h = 3,
    .w = 2, .c = 4
};

// Transpose parameters
static const int32_t transpose_float_default_f32_perm[] = {
    0, 2, 1, 3
};

static const cmsis_nn_transpose_params_f32 transpose_float_default_f32_transpose_params = {
    .num_dims = 4,
    .perm = {
        (int32_t)transpose_float_default_f32_perm[0],        (int32_t)transpose_float_default_f32_perm[1],        (int32_t)transpose_float_default_f32_perm[2],        (int32_t)transpose_float_default_f32_perm[3]    }
};

// Input data (for testing)
static const float transpose_float_default_f32_input[] = {
    0.546203792f, 0.887750924f, -0.726016164f, 0.312738121f, -0.354798645f, 0.740014613f, -0.392774284f, 0.054122373f, 0.249613672f, 0.446382105f, -0.296612948f, 0.931174457f, 0.452431142f, 0.333167583f, -0.013223344f, -0.03673185f,
    -0.455131799f, -0.26184687f, -0.878052831f, -0.325571388f, -0.778586507f, 0.330902904f, -0.955840766f, -0.306494087f
};

// Expected output (golden)
static const float transpose_float_default_f32_expected_output[] = {
    0.546203792f, 0.887750924f, -0.726016164f, 0.312738121f, 0.452431142f, 0.333167583f, -0.013223344f, -0.03673185f, -0.354798645f, 0.740014613f, -0.392774284f, 0.054122373f, -0.455131799f, -0.26184687f, -0.878052831f, -0.325571388f,
    0.249613672f, 0.446382105f, -0.296612948f, 0.931174457f, -0.778586507f, 0.330902904f, -0.955840766f, -0.306494087f
};

#endif