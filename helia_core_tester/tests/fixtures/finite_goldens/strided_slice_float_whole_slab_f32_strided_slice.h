#ifndef STRIDED_SLICE_FLOAT_WHOLE_SLAB_F32_STRIDEDSLICE_H
#define STRIDED_SLICE_FLOAT_WHOLE_SLAB_F32_STRIDEDSLICE_H

#include <stdint.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims strided_slice_float_whole_slab_f32_input_dims = {
    .n = 2, .h = 3,
    .w = 4, .c = 2
};

// Output dimensions
static const cmsis_nn_dims strided_slice_float_whole_slab_f32_output_dims = {
    .n = 1, .h = 3,
    .w = 4, .c = 2
};

// Begin dimensions
static const cmsis_nn_dims strided_slice_float_whole_slab_f32_begin_dims = {
    .n = 1, .h = 0,
    .w = 0, .c = 0
};

// Stride dimensions
static const cmsis_nn_dims strided_slice_float_whole_slab_f32_stride_dims = {
    .n = 1, .h = 1,
    .w = 1, .c = 1
};

// Input data (for testing)
static const float strided_slice_float_whole_slab_f32_input[] = {
    -0.630430043f, -0.983700752f, -0.555756152f, 0.734207511f, -0.43221128f, -0.184611082f, 0.262213171f, -0.483360231f, -0.489226341f, 0.02630941f, -0.998422742f, -0.251193672f, -0.084335133f, -0.475626647f, -0.318900019f, -0.440728337f,
    -0.234663233f, -0.325091958f, 0.869400144f, 0.552218854f, 0.979093969f, 0.299266547f, -0.082704417f, -0.277553856f, -0.172994137f, 0.99201411f, 0.8755216f, 0.455936432f, -0.014547333f, -0.664488614f, -0.476195902f, -0.474020541f,
    -0.95756644f, 0.406363726f, -0.977509618f, 0.140295148f, 0.841403842f, 0.386315852f, -0.53752315f, 0.596441686f, -0.674152553f, 0.365210384f, -0.921377003f, -0.091131993f, -0.458041102f, -0.324986994f, -0.462538004f, 0.184023887f,
};

// Expected output (golden)
static const float strided_slice_float_whole_slab_f32_expected_output[] = {
    -0.172994137f, 0.99201411f, 0.8755216f, 0.455936432f, -0.014547333f, -0.664488614f, -0.476195902f, -0.474020541f, -0.95756644f, 0.406363726f, -0.977509618f, 0.140295148f, 0.841403842f, 0.386315852f, -0.53752315f, 0.596441686f,
    -0.674152553f, 0.365210384f, -0.921377003f, -0.091131993f, -0.458041102f, -0.324986994f, -0.462538004f, 0.184023887f
};

#endif