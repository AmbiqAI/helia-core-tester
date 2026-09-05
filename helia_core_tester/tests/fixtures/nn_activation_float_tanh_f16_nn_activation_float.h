#ifndef NN_ACTIVATION_FLOAT_TANH_F16_NN_ACTIVATION_FLOAT_H
#define NN_ACTIVATION_FLOAT_TANH_F16_NN_ACTIVATION_FLOAT_H

#include <stdint.h>
// Golden arrays may carry NAN/INFINITY, and this header is included ahead of any
// other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

static const float16_t nn_activation_float_tanh_f16_input[] = {
    (float16_t)-0.41796875f, (float16_t)-0.750976562f, (float16_t)0.887207031f, (float16_t)0.459960938f, (float16_t)-0.94140625f, (float16_t)-0.467285156f, (float16_t)0.847167969f, (float16_t)0.322265625f, (float16_t)-0.248046875f, (float16_t)0.799316406f, (float16_t)-0.771972656f, (float16_t)-0.649902344f, (float16_t)-0.687011719f, (float16_t)-0.118469238f, (float16_t)0.909667969f, (float16_t)0.942871094f,
};

static const float16_t nn_activation_float_tanh_f16_expected_output[] = {
    (float16_t)-0.395019531f, (float16_t)-0.635742188f, (float16_t)0.709960938f, (float16_t)0.430175781f, (float16_t)-0.735839844f, (float16_t)-0.436035156f, (float16_t)0.689453125f, (float16_t)0.311523438f, (float16_t)-0.243041992f, (float16_t)0.663574219f, (float16_t)-0.647949219f, (float16_t)-0.571289062f, (float16_t)-0.595703125f, (float16_t)-0.117919922f, (float16_t)0.720703125f, (float16_t)0.736328125f,
};

#endif