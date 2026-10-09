#ifndef TRANSPOSE_CONV_FLOAT_DEFAULT_F16_HARNESS_H
#define TRANSPOSE_CONV_FLOAT_DEFAULT_F16_HARNESS_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims transpose_conv_float_default_f16_input_dims = {
    .n = 1,
    .h = 4,
    .w = 4,
    .c = 2
};

// Filter dimensions (C_OUT, HK, WK, C_IN)
static const cmsis_nn_dims transpose_conv_float_default_f16_filter_dims = {
    .n = 3,
    .h = 3,
    .w = 3,
    .c = 2
};

// Output dimensions
static const cmsis_nn_dims transpose_conv_float_default_f16_output_dims = {
    .n = 1,
    .h = 8,
    .w = 8,
    .c = 3
};

// Bias dimensions
static const cmsis_nn_dims transpose_conv_float_default_f16_bias_dims = {
    .n = 1,
    .h = 1,
    .w = 1,
    .c = 3
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
    (float16_t)0.085571289f, (float16_t)-0.076965332f, (float16_t)0.168579102f, (float16_t)0.089233398f, (float16_t)0.033996582f, (float16_t)-0.272949219f, (float16_t)0.049438477f, (float16_t)0.140869141f, (float16_t)-0.006340027f, (float16_t)0.136352539f, (float16_t)-0.294921875f, (float16_t)-0.276855469f, (float16_t)-0.092041016f, (float16_t)0.06842041f, (float16_t)-0.291015625f, (float16_t)-0.322021484f,
    (float16_t)-0.189941406f, (float16_t)0.141723633f, (float16_t)0.345947266f, (float16_t)-0.143310547f, (float16_t)-0.140625f, (float16_t)0.257568359f, (float16_t)-0.345703125f, (float16_t)-0.153442383f, (float16_t)-0.2578125f, (float16_t)0.245361328f, (float16_t)0.24621582f, (float16_t)-0.144287109f, (float16_t)0.315429688f, (float16_t)0.215698242f, (float16_t)-0.307861328f, (float16_t)-0.016769409f,
    (float16_t)0.078186035f, (float16_t)-0.090148926f, (float16_t)-0.052429199f, (float16_t)-2.466201782e-03f, (float16_t)0.013542175f, (float16_t)-0.363037109f, (float16_t)-0.212768555f, (float16_t)-0.055084229f, (float16_t)-0.185424805f, (float16_t)-0.081604004f, (float16_t)0.02355957f, (float16_t)-0.293212891f, (float16_t)0.152709961f, (float16_t)-0.037139893f, (float16_t)0.111694336f, (float16_t)0.176879883f,
    (float16_t)-0.005329132f, (float16_t)-0.085083008f, (float16_t)-0.087768555f, (float16_t)-0.084228516f, (float16_t)0.058898926f, (float16_t)0.080322266f
};

// Biases
static const float16_t transpose_conv_float_default_f16_biases[] = {
    (float16_t)-0.08215332f, (float16_t)0.022369385f, (float16_t)0.152587891f
};

// Input data (for testing)
static const float16_t transpose_conv_float_default_f16_input[] = {
    (float16_t)-5.030632019e-04f, (float16_t)-0.462402344f, (float16_t)-0.077148438f, (float16_t)0.088745117f, (float16_t)-0.980957031f, (float16_t)-0.090209961f, (float16_t)0.307373047f, (float16_t)-0.876464844f, (float16_t)0.381347656f, (float16_t)-0.694824219f, (float16_t)0.674316406f, (float16_t)-0.402099609f, (float16_t)0.736328125f, (float16_t)0.472167969f, (float16_t)0.131103516f, (float16_t)0.641113281f,
    (float16_t)0.191040039f, (float16_t)0.594726562f, (float16_t)-0.978027344f, (float16_t)-0.512207031f, (float16_t)0.752929688f, (float16_t)-0.399169922f, (float16_t)-0.295410156f, (float16_t)-0.313232422f, (float16_t)0.334716797f, (float16_t)0.583984375f, (float16_t)-0.039550781f, (float16_t)0.096618652f, (float16_t)0.439453125f, (float16_t)0.768554688f, (float16_t)-0.251464844f, (float16_t)0.142822266f,
};

// Expected output (golden)
static const float16_t transpose_conv_float_default_f16_expected_output[] = {
    (float16_t)-0.046600342f, (float16_t)0.088439941f, (float16_t)0.320556641f, (float16_t)-0.123474121f, (float16_t)-0.096679688f, (float16_t)0.178222656f, (float16_t)0.030609131f, (float16_t)0.054077148f, (float16_t)0.157104492f, (float16_t)-0.087219238f, (float16_t)0.056091309f, (float16_t)0.1640625f, (float16_t)-0.186035156f, (float16_t)-0.291015625f, (float16_t)0.179077148f, (float16_t)-0.255615234f,
    (float16_t)0.137084961f, (float16_t)0.366210938f, (float16_t)2.880096436e-03f, (float16_t)0.607421875f, (float16_t)0.6640625f, (float16_t)-0.108520508f, (float16_t)-0.246582031f, (float16_t)0.135498047f, (float16_t)-0.147338867f, (float16_t)-0.090942383f, (float16_t)0.288085938f, (float16_t)-0.145141602f, (float16_t)0.088989258f, (float16_t)0.169677734f, (float16_t)0.0546875f, (float16_t)-0.035858154f,
    (float16_t)0.042907715f, (float16_t)-0.069580078f, (float16_t)-0.009429932f, (float16_t)0.137451172f, (float16_t)-0.145141602f, (float16_t)0.247924805f, (float16_t)0.162963867f, (float16_t)-0.088256836f, (float16_t)-0.206176758f, (float16_t)0.006137848f, (float16_t)0.123840332f, (float16_t)-0.600585938f, (float16_t)0.291259766f, (float16_t)-0.203613281f, (float16_t)0.224487305f, (float16_t)0.232055664f,
    (float16_t)-0.027633667f, (float16_t)0.26171875f, (float16_t)0.449462891f, (float16_t)0.069152832f, (float16_t)-0.168579102f, (float16_t)0.148681641f, (float16_t)0.156860352f, (float16_t)0.311523438f, (float16_t)0.249389648f, (float16_t)-0.010482788f, (float16_t)-0.190063477f, (float16_t)0.030563354f, (float16_t)0.188598633f, (float16_t)0.345458984f, (float16_t)-0.085571289f, (float16_t)0.398681641f,
    (float16_t)-0.028121948f, (float16_t)0.063598633f, (float16_t)-0.138793945f, (float16_t)-0.379394531f, (float16_t)-0.245483398f, (float16_t)0.189941406f, (float16_t)0.272216797f, (float16_t)0.136230469f, (float16_t)-0.161132812f, (float16_t)-0.246459961f, (float16_t)0.365234375f, (float16_t)-0.179321289f, (float16_t)0.216552734f, (float16_t)0.236572266f, (float16_t)-0.025558472f, (float16_t)-0.279785156f,
    (float16_t)0.206054688f, (float16_t)-0.141235352f, (float16_t)0.246459961f, (float16_t)0.270507812f, (float16_t)-0.066772461f, (float16_t)0.07434082f, (float16_t)0.035675049f, (float16_t)-0.022445679f, (float16_t)0.135498047f, (float16_t)0.247436523f, (float16_t)-0.333251953f, (float16_t)0.479980469f, (float16_t)0.133422852f, (float16_t)0.004432678f, (float16_t)-0.037841797f, (float16_t)0.148803711f,
    (float16_t)-0.194213867f, (float16_t)-0.102539062f, (float16_t)-3.646850586e-03f, (float16_t)0.115905762f, (float16_t)0.241088867f, (float16_t)0.104248047f, (float16_t)-0.54296875f, (float16_t)-0.619140625f, (float16_t)0.238647461f, (float16_t)-0.359375f, (float16_t)0.116943359f, (float16_t)0.363525391f, (float16_t)-0.100952148f, (float16_t)0.487792969f, (float16_t)0.494140625f, (float16_t)-0.357177734f,
    (float16_t)-0.171264648f, (float16_t)-0.090026855f, (float16_t)0.01008606f, (float16_t)-0.324951172f, (float16_t)0.181274414f, (float16_t)-0.404541016f, (float16_t)-0.064331055f, (float16_t)0.167236328f, (float16_t)0.011070251f, (float16_t)0.119018555f, (float16_t)-0.017288208f, (float16_t)-2.271652222e-03f, (float16_t)-0.016403198f, (float16_t)0.159667969f, (float16_t)-0.423583984f, (float16_t)0.337402344f,
    (float16_t)0.40625f, (float16_t)-0.145751953f, (float16_t)-0.14453125f, (float16_t)0.022262573f, (float16_t)0.329101562f, (float16_t)-0.688476562f, (float16_t)0.087524414f, (float16_t)-0.141357422f, (float16_t)0.265380859f, (float16_t)0.282470703f, (float16_t)-0.252441406f, (float16_t)0.173095703f, (float16_t)0.250976562f, (float16_t)-0.12298584f, (float16_t)-0.005168915f, (float16_t)0.11907959f,
    (float16_t)-0.075378418f, (float16_t)-0.014312744f, (float16_t)-0.106506348f, (float16_t)-0.220703125f, (float16_t)0.087036133f, (float16_t)-0.017654419f, (float16_t)-0.138061523f, (float16_t)0.08770752f, (float16_t)0.11505127f, (float16_t)0.369384766f, (float16_t)0.022521973f, (float16_t)0.284667969f, (float16_t)-0.114868164f, (float16_t)-0.109436035f, (float16_t)-0.189819336f, (float16_t)-0.030059814f,
    (float16_t)0.253417969f, (float16_t)-0.015716553f, (float16_t)-0.503417969f, (float16_t)-0.297119141f, (float16_t)-6.359100342e-03f, (float16_t)0.075012207f, (float16_t)0.09967041f, (float16_t)0.250488281f, (float16_t)0.016662598f, (float16_t)0.079345703f, (float16_t)-0.010757446f, (float16_t)-4.64630127e-03f, (float16_t)0.020523071f, (float16_t)0.182006836f, (float16_t)-0.330810547f, (float16_t)0.287841797f,
    (float16_t)0.263916016f, (float16_t)-0.068725586f, (float16_t)-1.309394836e-03f, (float16_t)0.142944336f, (float16_t)0.032745361f, (float16_t)0.106018066f, (float16_t)-0.049743652f, (float16_t)0.019851685f, (float16_t)0.019683838f, (float16_t)0.191162109f, (float16_t)-0.416748047f, (float16_t)0.426513672f, (float16_t)0.289794922f, (float16_t)-0.061096191f, (float16_t)-0.060150146f, (float16_t)0.108886719f,
};

#endif