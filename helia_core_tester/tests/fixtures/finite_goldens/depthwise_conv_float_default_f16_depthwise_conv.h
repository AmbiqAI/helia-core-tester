#ifndef DEPTHWISE_CONV_FLOAT_DEFAULT_F16_HARNESS_H
#define DEPTHWISE_CONV_FLOAT_DEFAULT_F16_HARNESS_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims depthwise_conv_float_default_f16_input_dims = {
    .n = 1,
    .h = 6,
    .w = 6,
    .c = 3
};

// Filter dimensions
static const cmsis_nn_dims depthwise_conv_float_default_f16_filter_dims = {
    .n = 1,
    .h = 3,
    .w = 3,
    .c = 3
};

// Output dimensions
static const cmsis_nn_dims depthwise_conv_float_default_f16_output_dims = {
    .n = 1,
    .h = 6,
    .w = 6,
    .c = 3
};

// Depthwise convolution parameters
static const cmsis_nn_dw_conv_params_f16 depthwise_conv_float_default_f16_dw_conv_params = {
    .ch_mult = 1,
    .stride = {.w = 1, .h = 1},
    .dilation = {.w = 1, .h = 1},
    .padding = {.w = 1, .h = 1},
    .activation = {.min = -1.0e+30f, .max = 1.0e+30f}
};

// Weights
static const float16_t depthwise_conv_float_default_f16_weights[] = {
    (float16_t)-0.067321777f, (float16_t)-0.074829102f, (float16_t)-0.056365967f, (float16_t)0.052734375f, (float16_t)0.059020996f, (float16_t)-0.094238281f, (float16_t)-0.053955078f, (float16_t)-0.004058838f, (float16_t)0.404541016f, (float16_t)-0.102966309f, (float16_t)0.175170898f, (float16_t)-0.063354492f, (float16_t)0.400390625f, (float16_t)0.184204102f, (float16_t)-0.299560547f, (float16_t)0.04598999f,
    (float16_t)0.275146484f, (float16_t)-0.118591309f, (float16_t)0.050170898f, (float16_t)0.335693359f, (float16_t)0.135620117f, (float16_t)0.311279297f, (float16_t)-0.333740234f, (float16_t)0.165893555f, (float16_t)0.053009033f, (float16_t)0.250244141f, (float16_t)-0.187988281f
};

// Biases
static const float16_t depthwise_conv_float_default_f16_biases[] = {
    (float16_t)0.09564209f, (float16_t)0.097045898f, (float16_t)0.097351074f
};

// Input data (for testing)
static const float16_t depthwise_conv_float_default_f16_input[] = {
    (float16_t)-0.575683594f, (float16_t)0.365722656f, (float16_t)0.121643066f, (float16_t)0.383056641f, (float16_t)0.814941406f, (float16_t)0.4453125f, (float16_t)-0.716308594f, (float16_t)0.478515625f, (float16_t)-0.223999023f, (float16_t)0.364746094f, (float16_t)-0.628417969f, (float16_t)-0.595703125f, (float16_t)-0.909179688f, (float16_t)-0.184082031f, (float16_t)0.60546875f, (float16_t)0.655273438f,
    (float16_t)0.602050781f, (float16_t)0.825195312f, (float16_t)0.674316406f, (float16_t)-0.177612305f, (float16_t)0.967773438f, (float16_t)0.479980469f, (float16_t)0.935546875f, (float16_t)-0.5234375f, (float16_t)-0.979492188f, (float16_t)-0.776367188f, (float16_t)-0.028961182f, (float16_t)0.87109375f, (float16_t)0.691894531f, (float16_t)0.748535156f, (float16_t)0.978515625f, (float16_t)0.114257812f,
    (float16_t)-0.502929688f, (float16_t)-0.196044922f, (float16_t)-0.6484375f, (float16_t)0.44140625f, (float16_t)0.854492188f, (float16_t)0.901367188f, (float16_t)-0.884765625f, (float16_t)-0.735839844f, (float16_t)-0.463134766f, (float16_t)0.280029297f, (float16_t)-0.526855469f, (float16_t)-0.541015625f, (float16_t)0.976074219f, (float16_t)0.299560547f, (float16_t)-0.233886719f, (float16_t)-0.141235352f,
    (float16_t)-0.803222656f, (float16_t)0.360351562f, (float16_t)-0.56640625f, (float16_t)0.476806641f, (float16_t)0.319580078f, (float16_t)0.776855469f, (float16_t)0.370361328f, (float16_t)-0.877441406f, (float16_t)-0.068359375f, (float16_t)-0.671875f, (float16_t)-0.026779175f, (float16_t)-0.24230957f, (float16_t)0.34765625f, (float16_t)0.603027344f, (float16_t)0.888671875f, (float16_t)0.215087891f,
    (float16_t)-0.211669922f, (float16_t)0.630371094f, (float16_t)-0.879394531f, (float16_t)-0.099853516f, (float16_t)-0.359375f, (float16_t)0.399169922f, (float16_t)-0.799804688f, (float16_t)-0.070983887f, (float16_t)0.261962891f, (float16_t)-0.340576172f, (float16_t)-0.573730469f, (float16_t)-0.409179688f, (float16_t)-0.062347412f, (float16_t)-0.187866211f, (float16_t)-0.214233398f, (float16_t)0.210571289f,
    (float16_t)-0.580078125f, (float16_t)-0.01184845f, (float16_t)-0.363037109f, (float16_t)0.654296875f, (float16_t)-0.836914062f, (float16_t)-0.884277344f, (float16_t)0.60546875f, (float16_t)0.276855469f, (float16_t)0.515625f, (float16_t)0.170532227f, (float16_t)0.9765625f, (float16_t)0.357666016f, (float16_t)-0.315673828f, (float16_t)-0.281005859f, (float16_t)0.635742188f, (float16_t)0.838378906f,
    (float16_t)0.119934082f, (float16_t)0.187866211f, (float16_t)-0.029556274f, (float16_t)-0.875f, (float16_t)-0.868652344f, (float16_t)0.696289062f, (float16_t)-0.208862305f, (float16_t)0.173950195f, (float16_t)0.635742188f, (float16_t)0.548828125f, (float16_t)-0.5f, (float16_t)0.993164062f
};

// Expected output (golden)
static const float16_t depthwise_conv_float_default_f16_expected_output[] = {
    (float16_t)0.118103027f, (float16_t)0.682128906f, (float16_t)0.267089844f, (float16_t)0.406738281f, (float16_t)-0.12322998f, (float16_t)0.032684326f, (float16_t)-0.448486328f, (float16_t)0.901367188f, (float16_t)-9.620666504e-03f, (float16_t)0.547363281f, (float16_t)-0.448486328f, (float16_t)0.432861328f, (float16_t)0.062103271f, (float16_t)0.150512695f, (float16_t)-0.208984375f, (float16_t)0.439697266f,
    (float16_t)0.430419922f, (float16_t)-0.183227539f, (float16_t)0.563476562f, (float16_t)-0.076721191f, (float16_t)-0.161254883f, (float16_t)0.05682373f, (float16_t)0.365234375f, (float16_t)-0.200195312f, (float16_t)-0.57421875f, (float16_t)0.244628906f, (float16_t)0.031890869f, (float16_t)0.730957031f, (float16_t)0.034393311f, (float16_t)0.463623047f, (float16_t)0.071166992f, (float16_t)-0.024215698f,
    (float16_t)0.199462891f, (float16_t)0.120300293f, (float16_t)0.061248779f, (float16_t)-0.062866211f, (float16_t)0.493164062f, (float16_t)0.407470703f, (float16_t)0.060455322f, (float16_t)-0.450683594f, (float16_t)-0.042266846f, (float16_t)-0.279785156f, (float16_t)-0.070800781f, (float16_t)-0.529785156f, (float16_t)0.135131836f, (float16_t)0.329833984f, (float16_t)0.404785156f, (float16_t)0.165283203f,
    (float16_t)-0.473144531f, (float16_t)-0.069885254f, (float16_t)0.406738281f, (float16_t)0.373291016f, (float16_t)0.405517578f, (float16_t)-0.173217773f, (float16_t)0.357666016f, (float16_t)0.081176758f, (float16_t)0.283447266f, (float16_t)-0.388916016f, (float16_t)-0.029083252f, (float16_t)0.487304688f, (float16_t)0.231689453f, (float16_t)-0.033172607f, (float16_t)-0.637695312f, (float16_t)0.141357422f,
    (float16_t)0.131958008f, (float16_t)-0.459960938f, (float16_t)-0.594726562f, (float16_t)0.161254883f, (float16_t)0.706054688f, (float16_t)0.469482422f, (float16_t)-0.544921875f, (float16_t)0.210449219f, (float16_t)0.526367188f, (float16_t)5.199432373e-03f, (float16_t)-0.010055542f, (float16_t)-0.216308594f, (float16_t)0.100402832f, (float16_t)0.746582031f, (float16_t)0.08026123f, (float16_t)-3.18145752e-03f,
    (float16_t)0.368164062f, (float16_t)-0.167602539f, (float16_t)0.163085938f, (float16_t)-0.396484375f, (float16_t)-0.387695312f, (float16_t)-0.44921875f, (float16_t)-0.162841797f, (float16_t)0.533203125f, (float16_t)0.22265625f, (float16_t)0.285888672f, (float16_t)0.509765625f, (float16_t)0.318115234f, (float16_t)0.070556641f, (float16_t)-0.139526367f, (float16_t)0.349365234f, (float16_t)-0.314941406f,
    (float16_t)0.149291992f, (float16_t)0.022583008f, (float16_t)0.300537109f, (float16_t)-0.217651367f, (float16_t)-0.015792847f, (float16_t)0.031219482f, (float16_t)0.069091797f, (float16_t)-0.187744141f, (float16_t)-0.280029297f, (float16_t)0.407714844f, (float16_t)0.131958008f, (float16_t)-0.290527344f
};

#endif