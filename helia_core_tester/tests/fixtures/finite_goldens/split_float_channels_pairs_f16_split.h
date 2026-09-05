#ifndef SPLIT_FLOAT_CHANNELS_PAIRS_F16_SPLIT_H
#define SPLIT_FLOAT_CHANNELS_PAIRS_F16_SPLIT_H

#include <stdint.h>
// Golden arrays may carry NAN/INFINITY, and this header is included ahead of any
// other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims split_float_channels_pairs_f16_input_dims = {
    .n = 1, .h = 2,
    .w = 2, .c = 4
};

// Input shape
static const int32_t split_float_channels_pairs_f16_input_shape[] = {
    1, 2, 2, 4
};

// Split dimensions
static const int32_t split_float_channels_pairs_f16_split_dims[] = {
    2, 2
};

// Input data (for testing)
static const float16_t split_float_channels_pairs_f16_input[] = {
    (float16_t)0.153808594f, (float16_t)-0.706054688f, (float16_t)0.362548828f, (float16_t)-0.864746094f, (float16_t)-0.700683594f, (float16_t)0.323242188f, (float16_t)-0.991210938f, (float16_t)-0.13659668f, (float16_t)0.995605469f, (float16_t)0.668945312f, (float16_t)-0.104980469f, (float16_t)0.232177734f, (float16_t)-0.465820312f, (float16_t)-0.659667969f, (float16_t)-0.55078125f, (float16_t)0.701171875f,
};

// Expected output 0
static const float16_t split_float_channels_pairs_f16_out_0_expected_output[] = {
    (float16_t)0.153808594f, (float16_t)-0.706054688f, (float16_t)-0.700683594f, (float16_t)0.323242188f, (float16_t)0.995605469f, (float16_t)0.668945312f, (float16_t)-0.465820312f, (float16_t)-0.659667969f
};
// Expected output 1
static const float16_t split_float_channels_pairs_f16_out_1_expected_output[] = {
    (float16_t)0.362548828f, (float16_t)-0.864746094f, (float16_t)-0.991210938f, (float16_t)-0.13659668f, (float16_t)-0.104980469f, (float16_t)0.232177734f, (float16_t)-0.55078125f, (float16_t)0.701171875f
};

#endif