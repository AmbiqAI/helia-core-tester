#ifndef AVG_POOL_FLOAT_DEFAULT_F32_POOLING_H
#define AVG_POOL_FLOAT_DEFAULT_F32_POOLING_H

#include <stdint.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions: input shape is [N, H_IN, W_IN, C_IN]
static const cmsis_nn_dims avg_pool_float_default_f32_input_dims = {
    .n = 1, .h = 6,
    .w = 6, .c = 3
};

// Filter dimensions: pool_size is [H_FILT, W_FILT]
static const cmsis_nn_dims avg_pool_float_default_f32_filter_dims = {
    .n = 1, .h = 2,
    .w = 2, .c = 1
};

// Output dimensions: output shape is [N, H_OUT, W_OUT, C_OUT]. C_OUT = C_IN
static const cmsis_nn_dims avg_pool_float_default_f32_output_dims = {
    .n = 1, .h = 3,
    .w = 3, .c = 3
};

// Pooling parameters
static const cmsis_nn_pool_params_f32 avg_pool_float_default_f32_pool_params = {
    .stride.w = 2,
    .stride.h = 2,
    .padding.w = 0,
    .padding.h = 0,
    .activation.min = -1.0e+30f,
    .activation.max = 1.0e+30f};

// Input data (for testing)
static const float avg_pool_float_default_f32_input[] = {
    -31.0f, 21.0f, 26.0f, -9.0f, -31.0f, 8.0f, -14.0f, -6.0f, 15.0f, 16.0f, -30.0f, 31.0f, 15.0f, -30.0f, 11.0f, -30.0f,
    -2.0f, 26.0f, 0.0f, 31.0f, 8.0f, 16.0f, -21.0f, 29.0f, -31.0f, -9.0f, -10.0f, 11.0f, 22.0f, -14.0f, -2.0f, -8.0f,
    -17.0f, 22.0f, -29.0f, 29.0f, 31.0f, 26.0f, 6.0f, -32.0f, -26.0f, -29.0f, -11.0f, 4.0f, -17.0f, 19.0f, 14.0f, 10.0f,
    23.0f, 18.0f, -24.0f, -4.0f, -12.0f, -31.0f, -3.0f, -15.0f, 28.0f, 26.0f, -12.0f, -14.0f, -29.0f, -8.0f, -16.0f, 19.0f,
    17.0f, 7.0f, 28.0f, -8.0f, -2.0f, 29.0f, -12.0f, 22.0f, -4.0f, -16.0f, 12.0f, -31.0f, 2.0f, 6.0f, 29.0f, -24.0f,
    -18.0f, 7.0f, 9.0f, 22.0f, 12.0f, -13.0f, -5.0f, -17.0f, -16.0f, -26.0f, 17.0f, 22.0f, -13.0f, -18.0f, 30.0f, 15.0f,
    6.0f, 10.0f, 14.0f, 7.0f, 15.0f, -14.0f, -17.0f, 3.0f, -5.0f, -23.0f, -18.0f, 31.0f
};

// Expected output (golden)
static const float avg_pool_float_default_f32_expected_output[] = {
    -6.0f, 0.0f, 17.75f, -4.5f, -5.75f, 5.5f, 1.25f, -17.25f, 12.25f, 5.5f, -6.75f, -2.25f, -0.5f, 6.75f, -4.0f, 19.0f,
    -3.5f, -8.75f, -9.0f, 9.5f, 5.0f, 12.25f, 2.5f, 1.0f, -11.25f, -11.0f, -1.25f
};

#endif