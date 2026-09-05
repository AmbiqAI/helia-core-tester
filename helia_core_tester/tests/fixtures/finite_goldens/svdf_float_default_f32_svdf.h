#ifndef SVDF_FLOAT_DEFAULT_F32_SVDF_H
#define SVDF_FLOAT_DEFAULT_F32_SVDF_H

#include <stdint.h>
// Golden arrays may carry NAN/INFINITY, and this header is included ahead of any
// other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"

static const cmsis_nn_dims svdf_float_default_f32_input_dims = {
    .n = 1, .h = 4, .w = 1, .c = 1
};

static const cmsis_nn_dims svdf_float_default_f32_state_dims = {
    .n = 1, .h = 4, .w = 3, .c = 1
};

static const cmsis_nn_dims svdf_float_default_f32_weights_feature_dims = {
    .n = 4, .h = 4, .w = 1, .c = 1
};

static const cmsis_nn_dims svdf_float_default_f32_weights_time_dims = {
    .n = 4, .h = 3, .w = 1, .c = 1
};

static const cmsis_nn_dims svdf_float_default_f32_bias_dims = {
    .n = 4, .h = 1, .w = 1, .c = 1
};

static const cmsis_nn_dims svdf_float_default_f32_output_dims = {
    .n = 1, .h = 4, .w = 1, .c = 1
};

static const cmsis_nn_svdf_params_f32 svdf_float_default_f32_svdf_params = {
    .rank = 1,
    .input_activation = {.min = -1.0e+30f, .max = 1.0e+30f},
    .output_activation = {.min = -1.0e+30f, .max = 1.0e+30f}
};

static const float svdf_float_default_f32_input_sequence[] = {
    -0.855112135f, -0.952575743f, 0.550172865f, -0.962845147f, -0.94311744f, 0.92508626f, 0.127314314f, 0.243604049f
};

static const float svdf_float_default_f32_weights_feature[] = {
    0.123945676f, -0.109201357f, 0.829697371f, -0.669368982f, -0.539382815f, 0.213675633f, -0.898388326f, 0.015709581f, -0.658328116f, 0.813904464f, 0.814987481f, 0.424194068f, -0.114826135f, -0.181280702f, -0.788436472f, -0.378994048f,
};

static const float svdf_float_default_f32_weights_time[] = {
    -0.601521969f, -0.541270137f, 0.953217745f, -0.149808988f, -0.173734844f, 0.827832937f, -0.532364786f, -0.746871591f, -0.209843099f, -0.586137235f, -0.722408533f, 0.079431668f
};

static const float svdf_float_default_f32_initial_state[] = {
    0.008138136f, -0.251509517f, 0.271143585f, 0.196268514f, 0.031145267f, 0.010821174f, 0.0727382f, 0.194062039f, 0.491680175f, 0.238616332f, -0.488429874f, -0.156072319f
};

static const float svdf_float_default_f32_bias[] = {
    0.43134445f, 0.000509068f, -0.19906047f, -0.354343981f
};

static const float svdf_float_default_f32_expected_output[] = {
    -0.58907944f, 0.535856307f, -0.663787723f, -0.428822726f
};

#endif