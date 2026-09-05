#ifndef REDUCE_SUM_FLOAT_AXIS_C_F32_REDUCESUM_H
#define REDUCE_SUM_FLOAT_AXIS_C_F32_REDUCESUM_H

#include <stdint.h>
// Golden arrays may carry NAN/INFINITY, and this header is included ahead of any
// other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims reduce_sum_float_axis_c_f32_input_dims = {
    .n = 1, .h = 3,
    .w = 4, .c = 5
};

// Output dimensions
static const cmsis_nn_dims reduce_sum_float_axis_c_f32_output_dims = {
    .n = 1, .h = 3,
    .w = 4, .c = 1
};

// Axis dimensions
static const cmsis_nn_dims reduce_sum_float_axis_c_f32_axis_dims = {
    .n = 0, .h = 0,
    .w = 0, .c = 1
};

// Input data (for testing)
static const float reduce_sum_float_axis_c_f32_input[] = {
    -0.784942091f, 0.293971896f, -0.473547965f, 0.135935664f, -0.97969085f, 0.986676574f, 0.677757919f, 0.0001229f, -0.96663481f, 0.548669457f, 0.695767999f, 0.343006253f, -0.158346519f, 0.3354913f, 0.225863978f, -0.440424353f,
    -0.769464493f, -0.797136009f, -0.684970498f, 0.486699611f, -0.269791037f, -0.311104357f, -0.895334661f, -0.988561392f, -0.451411813f, -0.275301903f, -0.323385715f, 0.048329514f, -0.440989316f, 0.388321787f, -0.532255769f, -0.521213949f,
    0.03159323f, -0.611594856f, 0.282745659f, 0.385577589f, 0.705968678f, 0.864809871f, -0.398514539f, -0.428739429f, -0.568201661f, -0.190557152f, 0.97429049f, -0.052965675f, -0.353934526f, 0.287794471f, -0.535157979f, -0.769671142f,
    0.626171052f, 0.615984499f, -0.366151154f, 0.656705379f, -0.273309767f, -0.640766382f, -0.055236008f, -0.260664552f, 0.987345517f, 0.207679212f, -0.868555188f, -0.809247971f
};

// Expected output (golden)
static const float reduce_sum_float_axis_c_f32_expected_output[] = {
    -1.808273315f, 1.246592045f, 1.441782951f, -2.205295563f, -2.91620326f, -0.603025675f, -1.350725651f, 1.12910223f, -0.19136849f, 0.225120902f, -0.678757906f, -0.743442953f
};

#endif