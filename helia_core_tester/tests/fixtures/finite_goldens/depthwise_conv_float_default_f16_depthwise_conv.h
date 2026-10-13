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
    (float16_t)0.316650391f, (float16_t)0.139648438f, (float16_t)-0.152099609f, (float16_t)0.1796875f, (float16_t)0.2421875f, (float16_t)-0.284912109f, (float16_t)0.262207031f, (float16_t)-0.04196167f, (float16_t)-0.3125f, (float16_t)-0.313720703f, (float16_t)0.283447266f, (float16_t)0.390136719f, (float16_t)-0.024581909f, (float16_t)0.30078125f, (float16_t)-0.282470703f, (float16_t)0.401611328f,
    (float16_t)-0.04498291f, (float16_t)-0.374511719f, (float16_t)-0.285888672f, (float16_t)-0.145263672f, (float16_t)-0.181518555f, (float16_t)0.158203125f, (float16_t)-0.257324219f, (float16_t)-0.026580811f, (float16_t)0.388183594f, (float16_t)-0.043426514f, (float16_t)-0.170288086f
};

// Biases
static const float16_t depthwise_conv_float_default_f16_biases[] = {
    (float16_t)-0.305419922f, (float16_t)-0.482421875f, (float16_t)0.616210938f
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
    (float16_t)0.155517578f, (float16_t)-0.404052734f, (float16_t)0.478515625f, (float16_t)-0.918945312f, (float16_t)-0.336425781f, (float16_t)0.46484375f, (float16_t)-0.215576172f, (float16_t)-0.045410156f, (float16_t)1.044921875f, (float16_t)0.342773438f, (float16_t)-0.59765625f, (float16_t)0.541503906f, (float16_t)-0.3046875f, (float16_t)-0.844726562f, (float16_t)-0.293945312f, (float16_t)-0.347167969f,
    (float16_t)-0.20324707f, (float16_t)0.698730469f, (float16_t)-0.282714844f, (float16_t)-0.735351562f, (float16_t)0.340820312f, (float16_t)-1.7890625f, (float16_t)0.023651123f, (float16_t)1.064453125f, (float16_t)0.249389648f, (float16_t)-0.009170532f, (float16_t)0.269287109f, (float16_t)-0.140014648f, (float16_t)-0.454101562f, (float16_t)0.51953125f, (float16_t)-0.584960938f, (float16_t)-0.453125f,
    (float16_t)0.453613281f, (float16_t)-0.47265625f, (float16_t)-0.659667969f, (float16_t)0.050292969f, (float16_t)-0.577148438f, (float16_t)-0.045806885f, (float16_t)0.692382812f, (float16_t)-0.801269531f, (float16_t)6.184577942e-04f, (float16_t)-0.295166016f, (float16_t)0.59375f, (float16_t)-0.994628906f, (float16_t)0.26953125f, (float16_t)-0.773925781f, (float16_t)-0.696777344f, (float16_t)1.080078125f,
    (float16_t)0.166381836f, (float16_t)-0.212036133f, (float16_t)0.228881836f, (float16_t)0.523925781f, (float16_t)-0.204956055f, (float16_t)0.193603516f, (float16_t)-0.741210938f, (float16_t)-0.416992188f, (float16_t)0.937988281f, (float16_t)-0.488037109f, (float16_t)-0.673339844f, (float16_t)0.282958984f, (float16_t)-0.1875f, (float16_t)-0.514160156f, (float16_t)-0.303955078f, (float16_t)-1.362304688f,
    (float16_t)-0.416748047f, (float16_t)0.973144531f, (float16_t)-0.136962891f, (float16_t)-0.237426758f, (float16_t)0.766601562f, (float16_t)0.075073242f, (float16_t)-0.627929688f, (float16_t)0.246459961f, (float16_t)-0.540527344f, (float16_t)-0.913085938f, (float16_t)0.809570312f, (float16_t)-0.652832031f, (float16_t)-0.985351562f, (float16_t)0.504394531f, (float16_t)-0.510742188f, (float16_t)-0.372314453f,
    (float16_t)-0.221557617f, (float16_t)-0.909667969f, (float16_t)-0.266357422f, (float16_t)-0.345458984f, (float16_t)0.274902344f, (float16_t)-0.791503906f, (float16_t)0.352783203f, (float16_t)-0.109863281f, (float16_t)-0.682128906f, (float16_t)0.737304688f, (float16_t)-0.502441406f, (float16_t)-0.483398438f, (float16_t)0.613769531f, (float16_t)-0.603515625f, (float16_t)-0.269775391f, (float16_t)0.589355469f,
    (float16_t)-0.742675781f, (float16_t)-0.149169922f, (float16_t)0.680175781f, (float16_t)-0.694824219f, (float16_t)-0.719726562f, (float16_t)-0.117492676f, (float16_t)0.113098145f, (float16_t)-0.940429688f, (float16_t)0.011016846f, (float16_t)-0.46875f, (float16_t)-0.58203125f, (float16_t)0.443115234f
};

#endif