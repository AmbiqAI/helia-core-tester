#ifndef DEPTHWISE_CONV_FLOAT_DEFAULT_F16_DEPTHWISE_CONV2D_H
#define DEPTHWISE_CONV_FLOAT_DEFAULT_F16_DEPTHWISE_CONV2D_H

#include <stdint.h>
// Golden arrays may carry NAN/INFINITY, and this header is included ahead of any
// other translation-unit include that would define them.
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
    (float16_t)-0.239379883f, (float16_t)-0.148193359f, (float16_t)-0.092590332f, (float16_t)0.104797363f, (float16_t)-0.288818359f, (float16_t)0.270507812f, (float16_t)-0.3203125f, (float16_t)0.051422119f, (float16_t)-0.213134766f, (float16_t)-0.339355469f, (float16_t)-0.041320801f, (float16_t)0.376953125f, (float16_t)-0.213134766f, (float16_t)0.22265625f, (float16_t)-0.229980469f, (float16_t)0.157348633f,
    (float16_t)0.055664062f, (float16_t)-0.209350586f, (float16_t)-0.117553711f, (float16_t)0.017913818f, (float16_t)-0.369628906f, (float16_t)-0.098999023f, (float16_t)-0.29296875f, (float16_t)0.052337646f, (float16_t)0.120300293f, (float16_t)-0.280029297f, (float16_t)0.171142578f
};

// Biases
static const float16_t depthwise_conv_float_default_f16_biases[] = {
    (float16_t)0.201171875f, (float16_t)-0.391357422f, (float16_t)-0.619140625f
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
    (float16_t)0.375244141f, (float16_t)-0.474609375f, (float16_t)-0.779785156f, (float16_t)-0.042419434f, (float16_t)-0.258544922f, (float16_t)-1.01953125f, (float16_t)0.426513672f, (float16_t)-0.302978516f, (float16_t)0.044799805f, (float16_t)0.370117188f, (float16_t)-0.810058594f, (float16_t)-0.729980469f, (float16_t)0.151489258f, (float16_t)-0.212524414f, (float16_t)-1.383789062f, (float16_t)0.274414062f,
    (float16_t)-0.057739258f, (float16_t)-0.372070312f, (float16_t)-0.223144531f, (float16_t)-0.577148438f, (float16_t)-0.79296875f, (float16_t)0.03237915f, (float16_t)-0.180786133f, (float16_t)0.537597656f, (float16_t)0.275146484f, (float16_t)-0.640136719f, (float16_t)-1.018554688f, (float16_t)0.938476562f, (float16_t)-0.139892578f, (float16_t)-1.431640625f, (float16_t)-0.625f, (float16_t)-0.452636719f,
    (float16_t)-0.115478516f, (float16_t)0.244506836f, (float16_t)-0.774414062f, (float16_t)-0.493164062f, (float16_t)-0.297363281f, (float16_t)0.147460938f, (float16_t)-0.146240234f, (float16_t)0.252685547f, (float16_t)-1.022460938f, (float16_t)-1.282226562f, (float16_t)0.184082031f, (float16_t)-0.502441406f, (float16_t)-0.583496094f, (float16_t)0.034118652f, (float16_t)-0.379394531f, (float16_t)-0.145141602f,
    (float16_t)0.412353516f, (float16_t)-0.203369141f, (float16_t)-1.268554688f, (float16_t)0.181152344f, (float16_t)0.067749023f, (float16_t)-0.716308594f, (float16_t)0.266601562f, (float16_t)-0.755371094f, (float16_t)-0.9140625f, (float16_t)0.14440918f, (float16_t)-0.402099609f, (float16_t)-0.723144531f, (float16_t)0.481689453f, (float16_t)-0.016174316f, (float16_t)-0.627929688f, (float16_t)0.239501953f,
    (float16_t)0.054870605f, (float16_t)-0.009742737f, (float16_t)0.187255859f, (float16_t)-0.394287109f, (float16_t)-0.770996094f, (float16_t)0.728027344f, (float16_t)-0.877929688f, (float16_t)-0.690917969f, (float16_t)0.204589844f, (float16_t)-0.501464844f, (float16_t)-0.287841797f, (float16_t)-0.177246094f, (float16_t)-0.443359375f, (float16_t)-0.764160156f, (float16_t)0.428222656f, (float16_t)-0.34375f,
    (float16_t)-0.7578125f, (float16_t)0.413085938f, (float16_t)-0.354492188f, (float16_t)-0.794433594f, (float16_t)0.345214844f, (float16_t)-0.452148438f, (float16_t)-0.7421875f, (float16_t)0.648925781f, (float16_t)0.155395508f, (float16_t)-0.599609375f, (float16_t)0.10736084f, (float16_t)-0.181274414f, (float16_t)-0.837402344f, (float16_t)-0.088439941f, (float16_t)-0.174926758f, (float16_t)-0.798828125f,
    (float16_t)0.212646484f, (float16_t)-0.494628906f, (float16_t)-0.721191406f, (float16_t)0.632324219f, (float16_t)-0.5546875f, (float16_t)-0.822265625f, (float16_t)0.455322266f, (float16_t)-0.008918762f, (float16_t)-0.64453125f, (float16_t)0.384521484f, (float16_t)-0.527832031f, (float16_t)-0.618164062f
};

#endif