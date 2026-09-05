#ifndef TRANSPOSE_CONV_FLOAT_DEFAULT_F16_TRANSPOSE_CONV_H
#define TRANSPOSE_CONV_FLOAT_DEFAULT_F16_TRANSPOSE_CONV_H

#include <stdint.h>
// Golden arrays may carry NAN/INFINITY, and this header is included ahead of any
// other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims transpose_conv_float_default_f16_input_dims = {
    .n = 1, .h = 4,
    .w = 4, .c = 2
};

// Filter dimensions (C_OUT, HK, WK, C_IN)
static const cmsis_nn_dims transpose_conv_float_default_f16_filter_dims = {
    .n = 3, .h = 3,
    .w = 3, .c = 2
};

// Output dimensions
static const cmsis_nn_dims transpose_conv_float_default_f16_output_dims = {
    .n = 1, .h = 8,
    .w = 8, .c = 3
};

// Bias dimensions
static const cmsis_nn_dims transpose_conv_float_default_f16_bias_dims = {
    .n = 1, .h = 1,
    .w = 1, .c = 3
};

// Transpose convolution parameters
static const cmsis_nn_transpose_conv_params_f16 transpose_conv_float_default_f16_transpose_conv_params = {
    .stride = {.w = 2, .h = 2},
    .dilation = {.w = 1, .h = 1},
    .padding = {.w = 0, .h = 0},
    .padding_offsets = {.w = 1, .h = 1},
    .activation = {.min = -1.0e+30f, .max = 1.0e+30f}
};


// Weights
static const float16_t transpose_conv_float_default_f16_weights[] = {
    (float16_t)-0.01751709f, (float16_t)-0.081176758f, (float16_t)-0.19909668f, (float16_t)-0.054412842f, (float16_t)-0.071472168f, (float16_t)0.244018555f, (float16_t)-0.293945312f, (float16_t)0.121398926f, (float16_t)-0.322021484f, (float16_t)0.288574219f, (float16_t)0.197753906f, (float16_t)-0.121826172f, (float16_t)0.20300293f, (float16_t)0.04574585f, (float16_t)-0.06427002f, (float16_t)-0.318847656f,
    (float16_t)-0.082092285f, (float16_t)-0.28125f, (float16_t)-0.358642578f, (float16_t)-0.209594727f, (float16_t)0.080871582f, (float16_t)-0.018218994f, (float16_t)-0.106994629f, (float16_t)-0.049530029f, (float16_t)0.030044556f, (float16_t)0.363525391f, (float16_t)-0.354248047f, (float16_t)0.322753906f, (float16_t)-0.186767578f, (float16_t)0.105773926f, (float16_t)0.280273438f, (float16_t)0.360839844f,
    (float16_t)0.198730469f, (float16_t)0.13659668f, (float16_t)0.042541504f, (float16_t)0.268554688f, (float16_t)-0.288818359f, (float16_t)-0.245483398f, (float16_t)0.280517578f, (float16_t)0.312744141f, (float16_t)0.159545898f, (float16_t)-0.196166992f, (float16_t)0.154418945f, (float16_t)0.212646484f, (float16_t)0.270507812f, (float16_t)-0.09765625f, (float16_t)-0.207641602f, (float16_t)-0.331054688f,
    (float16_t)0.300048828f, (float16_t)-0.306396484f, (float16_t)0.246826172f, (float16_t)0.282958984f, (float16_t)-0.298583984f, (float16_t)0.292480469f
};

// Biases
static const float16_t transpose_conv_float_default_f16_biases[] = {
    (float16_t)-0.126586914f, (float16_t)-0.315917969f, (float16_t)0.147216797f
};

// Input data (for testing)
static const float16_t transpose_conv_float_default_f16_input[] = {
    (float16_t)-0.000503063f, (float16_t)-0.462402344f, (float16_t)-0.077148438f, (float16_t)0.088745117f, (float16_t)-0.980957031f, (float16_t)-0.090209961f, (float16_t)0.307373047f, (float16_t)-0.876464844f, (float16_t)0.381347656f, (float16_t)-0.694824219f, (float16_t)0.674316406f, (float16_t)-0.402099609f, (float16_t)0.736328125f, (float16_t)0.472167969f, (float16_t)0.131103516f, (float16_t)0.641113281f,
    (float16_t)0.191040039f, (float16_t)0.594726562f, (float16_t)-0.978027344f, (float16_t)-0.512207031f, (float16_t)0.752929688f, (float16_t)-0.399169922f, (float16_t)-0.295410156f, (float16_t)-0.313232422f, (float16_t)0.334716797f, (float16_t)0.583984375f, (float16_t)-0.039550781f, (float16_t)0.096618652f, (float16_t)0.439453125f, (float16_t)0.768554688f, (float16_t)-0.251464844f, (float16_t)0.142822266f,
};

// Expected output (golden)
static const float16_t transpose_conv_float_default_f16_expected_output[] = {
    (float16_t)-0.089111328f, (float16_t)-0.21887207f, (float16_t)0.260986328f, (float16_t)-0.101379395f, (float16_t)-0.307617188f, (float16_t)0.002450943f, (float16_t)-0.245239258f, (float16_t)-0.283935547f, (float16_t)0.23840332f, (float16_t)-0.116088867f, (float16_t)-0.323730469f, (float16_t)0.153320312f, (float16_t)-0.074951172f, (float16_t)0.05871582f, (float16_t)0.422851562f, (float16_t)0.073608398f,
    (float16_t)-0.393554688f, (float16_t)-0.15625f, (float16_t)-0.012771606f, (float16_t)-0.133056641f, (float16_t)0.134887695f, (float16_t)-0.140136719f, (float16_t)-0.275146484f, (float16_t)-0.040679932f, (float16_t)-0.182617188f, (float16_t)-0.484130859f, (float16_t)0.048828125f, (float16_t)-0.260009766f, (float16_t)-0.465087891f, (float16_t)0.192260742f, (float16_t)-0.036956787f, (float16_t)-0.334716797f,
    (float16_t)0.307373047f, (float16_t)-0.076171875f, (float16_t)-0.260009766f, (float16_t)0.117736816f, (float16_t)0.124755859f, (float16_t)-0.354492188f, (float16_t)-0.036743164f, (float16_t)0.163330078f, (float16_t)0.002334595f, (float16_t)-0.109191895f, (float16_t)-0.506347656f, (float16_t)-0.451660156f, (float16_t)0.241943359f, (float16_t)-0.478515625f, (float16_t)-0.707519531f, (float16_t)0.315917969f,
    (float16_t)-0.098205566f, (float16_t)-0.474121094f, (float16_t)0.349365234f, (float16_t)-0.01725769f, (float16_t)-0.335693359f, (float16_t)-0.094055176f, (float16_t)-0.184082031f, (float16_t)-0.59375f, (float16_t)0.062988281f, (float16_t)-0.262451172f, (float16_t)-0.257324219f, (float16_t)0.216796875f, (float16_t)-0.545898438f, (float16_t)-1.018554688f, (float16_t)-0.212524414f, (float16_t)-0.20715332f,
    (float16_t)-0.472167969f, (float16_t)0.233886719f, (float16_t)0.009857178f, (float16_t)-0.895507812f, (float16_t)0.604003906f, (float16_t)0.072143555f, (float16_t)-0.375732422f, (float16_t)0.212524414f, (float16_t)-0.322998047f, (float16_t)-0.557128906f, (float16_t)0.058349609f, (float16_t)-0.449951172f, (float16_t)-0.675292969f, (float16_t)0.318115234f, (float16_t)-0.213623047f, (float16_t)-0.586425781f,
    (float16_t)0.316650391f, (float16_t)-0.459716797f, (float16_t)-0.684570312f, (float16_t)0.368896484f, (float16_t)-0.103393555f, (float16_t)-0.290771484f, (float16_t)0.354492188f, (float16_t)-0.227539062f, (float16_t)-0.424316406f, (float16_t)0.300292969f, (float16_t)0.000790119f, (float16_t)-0.166503906f, (float16_t)-0.005355835f, (float16_t)0.016143799f, (float16_t)-0.155517578f, (float16_t)0.120117188f,
    (float16_t)-0.132568359f, (float16_t)-0.652832031f, (float16_t)0.2734375f, (float16_t)3.331899643e-05f, (float16_t)-0.330566406f, (float16_t)0.284423828f, (float16_t)0.346191406f, (float16_t)-0.034088135f, (float16_t)0.477783203f, (float16_t)0.180908203f, (float16_t)-0.306640625f, (float16_t)-0.234741211f, (float16_t)0.06628418f, (float16_t)-0.074890137f, (float16_t)-0.270507812f, (float16_t)-0.452636719f,
    (float16_t)-0.036987305f, (float16_t)0.548828125f, (float16_t)-0.384521484f, (float16_t)0.221069336f, (float16_t)0.268798828f, (float16_t)-0.263671875f, (float16_t)-0.220581055f, (float16_t)0.180053711f, (float16_t)-0.110595703f, (float16_t)-0.093994141f, (float16_t)0.303222656f, (float16_t)-0.016555786f, (float16_t)-0.191650391f, (float16_t)0.140869141f, (float16_t)0.064025879f, (float16_t)-0.504394531f,
    (float16_t)-0.349121094f, (float16_t)0.040557861f, (float16_t)-0.134887695f, (float16_t)-0.067199707f, (float16_t)-0.52734375f, (float16_t)-0.310058594f, (float16_t)0.551269531f, (float16_t)-0.484375f, (float16_t)-0.711425781f, (float16_t)0.389892578f, (float16_t)0.119750977f, (float16_t)-0.621582031f, (float16_t)0.010757446f, (float16_t)-0.121887207f, (float16_t)-0.3125f, (float16_t)0.097961426f,
    (float16_t)-0.113891602f, (float16_t)-0.290283203f, (float16_t)-0.217773438f, (float16_t)-0.427001953f, (float16_t)-0.180297852f, (float16_t)0.639160156f, (float16_t)-0.420166016f, (float16_t)-0.677734375f, (float16_t)0.054199219f, (float16_t)0.102233887f, (float16_t)-0.584960938f, (float16_t)-0.219970703f, (float16_t)0.188720703f, (float16_t)-0.747558594f, (float16_t)0.296875f, (float16_t)-0.177124023f,
    (float16_t)-0.19934082f, (float16_t)0.583984375f, (float16_t)-0.001490593f, (float16_t)-0.611816406f, (float16_t)-0.230102539f, (float16_t)0.034545898f, (float16_t)-0.440429688f, (float16_t)-0.040161133f, (float16_t)-0.154174805f, (float16_t)-0.09362793f, (float16_t)0.323242188f, (float16_t)-0.065917969f, (float16_t)-0.24597168f, (float16_t)0.180786133f, (float16_t)-0.108215332f, (float16_t)-0.282714844f,
    (float16_t)-0.101135254f, (float16_t)-0.085998535f, (float16_t)-0.270751953f, (float16_t)0.127075195f, (float16_t)-0.182128906f, (float16_t)-0.005760193f, (float16_t)0.354736328f, (float16_t)-0.046386719f, (float16_t)-0.223510742f, (float16_t)0.191040039f, (float16_t)-0.042053223f, (float16_t)-0.272460938f, (float16_t)-0.206787109f, (float16_t)-0.004428864f, (float16_t)-0.180786133f, (float16_t)0.065307617f,
};

#endif