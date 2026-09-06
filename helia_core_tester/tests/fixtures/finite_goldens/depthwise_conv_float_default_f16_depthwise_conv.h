#ifndef DEPTHWISE_CONV_FLOAT_DEFAULT_F16_DEPTHWISE_CONV2D_H
#define DEPTHWISE_CONV_FLOAT_DEFAULT_F16_DEPTHWISE_CONV2D_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims depthwise_conv_float_default_f16_input_dims = {
    .n = 1, .h = 6,
    .w = 6, .c = 3
};

// Filter dimensions  
static const cmsis_nn_dims depthwise_conv_float_default_f16_filter_dims = {
    .n = 1, .h = 3,
    .w = 3, .c = 3
};

// Output dimensions
static const cmsis_nn_dims depthwise_conv_float_default_f16_output_dims = {
    .n = 1, .h = 6,
    .w = 6, .c = 3
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
    (float16_t)0.197265625f, (float16_t)0.385253906f, (float16_t)-0.24230957f, (float16_t)0.233520508f, (float16_t)-0.065429688f, (float16_t)-0.235351562f, (float16_t)-0.025680542f, (float16_t)0.3984375f, (float16_t)0.390625f, (float16_t)-0.230957031f, (float16_t)-0.018234253f, (float16_t)0.057800293f, (float16_t)-0.161499023f, (float16_t)0.253417969f, (float16_t)-0.067626953f, (float16_t)-0.1875f,
    (float16_t)0.096557617f, (float16_t)-0.042144775f, (float16_t)0.268798828f, (float16_t)0.212036133f, (float16_t)0.350585938f, (float16_t)0.238525391f, (float16_t)0.145874023f, (float16_t)-0.159790039f, (float16_t)0.355712891f, (float16_t)0.10333252f, (float16_t)0.084472656f
};

// Biases
static const float16_t depthwise_conv_float_default_f16_biases[] = {
    (float16_t)-0.842773438f, (float16_t)-0.380371094f, (float16_t)0.952148438f
};

// Weight sum (precomputed for S8 depthwise convolutions)

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
    (float16_t)-0.489990234f, (float16_t)-0.138305664f, (float16_t)0.726074219f, (float16_t)-0.689941406f, (float16_t)-0.115722656f, (float16_t)1.358398438f, (float16_t)-0.678710938f, (float16_t)-0.178222656f, (float16_t)0.90234375f, (float16_t)-0.273193359f, (float16_t)-0.618164062f, (float16_t)0.781738281f, (float16_t)-0.505371094f, (float16_t)-0.261230469f, (float16_t)1.221679688f, (float16_t)-0.522460938f,
    (float16_t)-0.294921875f, (float16_t)0.684082031f, (float16_t)-1.244140625f, (float16_t)0.049316406f, (float16_t)1.21875f, (float16_t)-1.03125f, (float16_t)0.130737305f, (float16_t)0.550292969f, (float16_t)-1.276367188f, (float16_t)-0.696289062f, (float16_t)0.534667969f, (float16_t)-1.329101562f, (float16_t)-0.139404297f, (float16_t)1.668945312f, (float16_t)-1.263671875f, (float16_t)-0.380859375f,
    (float16_t)1.44140625f, (float16_t)-1.166015625f, (float16_t)-0.534179688f, (float16_t)0.229370117f, (float16_t)-0.848632812f, (float16_t)0.056762695f, (float16_t)0.558105469f, (float16_t)-0.489257812f, (float16_t)-1.1328125f, (float16_t)0.808105469f, (float16_t)-0.821289062f, (float16_t)0.215576172f, (float16_t)1.16015625f, (float16_t)-0.802246094f, (float16_t)-0.607421875f, (float16_t)0.856933594f,
    (float16_t)-0.476318359f, (float16_t)-0.395263672f, (float16_t)1.331054688f, (float16_t)-0.728027344f, (float16_t)-0.357421875f, (float16_t)0.770019531f, (float16_t)-0.641113281f, (float16_t)-0.904785156f, (float16_t)1.360351562f, (float16_t)-0.978027344f, (float16_t)-0.210571289f, (float16_t)1.236328125f, (float16_t)-1.225585938f, (float16_t)-0.50390625f, (float16_t)0.580566406f, (float16_t)-1.1640625f,
    (float16_t)-0.604003906f, (float16_t)0.294433594f, (float16_t)-1.0703125f, (float16_t)-0.618164062f, (float16_t)1.633789062f, (float16_t)-0.91015625f, (float16_t)-0.575683594f, (float16_t)1.075195312f, (float16_t)-0.571777344f, (float16_t)-0.308105469f, (float16_t)1.041015625f, (float16_t)-0.651855469f, (float16_t)-0.277587891f, (float16_t)1.129882812f, (float16_t)-1.126953125f, (float16_t)-0.422607422f,
    (float16_t)1.40625f, (float16_t)-0.744140625f, (float16_t)-0.424316406f, (float16_t)0.276611328f, (float16_t)-1.01953125f, (float16_t)-1.15234375f, (float16_t)1.072265625f, (float16_t)-0.699707031f, (float16_t)-0.255859375f, (float16_t)1.143554688f, (float16_t)-0.875976562f, (float16_t)-0.230957031f, (float16_t)0.999511719f, (float16_t)-1.083984375f, (float16_t)-0.250976562f, (float16_t)0.834960938f,
    (float16_t)-0.763671875f, (float16_t)-0.610839844f, (float16_t)1.411132812f, (float16_t)-0.713378906f, (float16_t)-0.834472656f, (float16_t)1.099609375f, (float16_t)-0.914550781f, (float16_t)-0.245361328f, (float16_t)0.672851562f, (float16_t)-0.983886719f, (float16_t)-0.884765625f, (float16_t)0.734863281f
};

#endif