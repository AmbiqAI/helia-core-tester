#ifndef CONCATENATION_AXIS_X_F32_CONCATENATION_H
#define CONCATENATION_AXIS_X_F32_CONCATENATION_H

#include <stdint.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Output dimensions
static const cmsis_nn_dims concatenation_axis_x_f32_output_dims = {
    .n = 1, .h = 2,
    .w = 6, .c = 3
};

// Input concatenation dimensions (along axis)
static const int32_t concatenation_axis_x_f32_input_concat_dims[] = {
    2, 1, 3
};

// Output shape
static const int32_t concatenation_axis_x_f32_output_shape[] = {
    1, 2, 6, 3
};

// Per-input dimensions (NHWC)
static const int32_t concatenation_axis_x_f32_input_x[] = {  // width
    2, 1, 3
};
static const int32_t concatenation_axis_x_f32_input_y[] = {  // height
    2, 2, 2
};
static const int32_t concatenation_axis_x_f32_input_z[] = {  // channels
    3, 3, 3
};
static const int32_t concatenation_axis_x_f32_input_w[] = {  // batch
    1, 1, 1
};

// Per-input offsets along concat axis
static const int32_t concatenation_axis_x_f32_offsets[] = {
    0, 2, 3
};

// Input 1 data (for testing)
static const float concatenation_axis_x_f32_input1[] = {
    -1.551459193f, 0.354537129f, -1.475263834f, 1.551109195f, 1.026612997f, 0.836554527f, 1.73639369f, 1.514538646f, -1.673038602f, -0.84045136f, 0.46366778f, -0.462945849f
};
// Input 2 data (for testing)
static const float concatenation_axis_x_f32_input2[] = {
    1.231376171f, -0.271517634f, 0.15827699f, -1.125284553f, 0.123243593f, -1.212194324f
};
// Input 3 data (for testing)
static const float concatenation_axis_x_f32_input3[] = {
    -1.179903388f, -0.172626495f, -1.410054684f, 1.195246935f, 0.126863673f, -0.173435926f, 0.683098078f, -0.862756252f, 1.041602135f, -1.748399973f, -0.527730644f, 1.467175961f, 0.803250492f, -0.017642166f, 1.045714855f, -0.59537816f,
    -1.160194159f, 1.562226534f
};

// Expected output (golden)
static const float concatenation_axis_x_f32_expected_output[] = {
    -1.551459193f, 0.354537129f, 1.231376171f, -1.179903388f, -0.172626495f, -1.410054684f, -1.475263834f, 1.551109195f, -0.271517634f, 1.195246935f, 0.126863673f, -0.173435926f, 1.026612997f, 0.836554527f, 0.15827699f, 0.683098078f,
    -0.862756252f, 1.041602135f, 1.73639369f, 1.514538646f, -1.125284553f, -1.748399973f, -0.527730644f, 1.467175961f, -1.673038602f, -0.84045136f, 0.123243593f, 0.803250492f, -0.017642166f, 1.045714855f, 0.46366778f, -0.462945849f,
    -1.212194324f, -0.59537816f, -1.160194159f, 1.562226534f
};

#endif