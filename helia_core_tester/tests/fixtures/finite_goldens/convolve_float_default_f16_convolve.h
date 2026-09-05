#ifndef CONVOLVE_FLOAT_DEFAULT_F16_CONV2D_H
#define CONVOLVE_FLOAT_DEFAULT_F16_CONV2D_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims convolve_float_default_f16_input_dims = {
    .n = 1, .h = 6,
    .w = 6, .c = 3
};

// Filter dimensions  
static const cmsis_nn_dims convolve_float_default_f16_filter_dims = {
    .n = 5, .h = 3,
    .w = 3, .c = 3
};

// Output dimensions
static const cmsis_nn_dims convolve_float_default_f16_output_dims = {
    .n = 1, .h = 6,
    .w = 6, .c = 5
};

// Convolution parameters
static const cmsis_nn_conv_params_f16 convolve_float_default_f16_conv_params = {
    .stride = {.w = 1, .h = 1},
    .dilation = {.w = 1, .h = 1},
    .padding = {.w = 1, .h = 1},
    .activation = {.min = -1.0e+30f, .max = 1.0e+30f},
    .weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD
};


// Weights
static const float16_t convolve_float_default_f16_weights[] = {
    (float16_t)0.139526367f, (float16_t)-0.166381836f, (float16_t)-0.012893677f, (float16_t)-0.132568359f, (float16_t)0.247924805f, (float16_t)0.073059082f, (float16_t)0.226928711f, (float16_t)-0.091552734f, (float16_t)-0.253173828f, (float16_t)-0.248779297f, (float16_t)0.189575195f, (float16_t)0.209228516f, (float16_t)-0.016311646f, (float16_t)-0.1328125f, (float16_t)-0.160888672f, (float16_t)-0.051757812f,
    (float16_t)-0.22644043f, (float16_t)0.155151367f, (float16_t)0.232910156f, (float16_t)0.227416992f, (float16_t)-0.201049805f, (float16_t)-0.113342285f, (float16_t)0.058044434f, (float16_t)-0.107055664f, (float16_t)-0.276367188f, (float16_t)-0.005050659f, (float16_t)-0.250976562f, (float16_t)0.272460938f, (float16_t)-0.018157959f, (float16_t)0.040863037f, (float16_t)0.06829834f, (float16_t)0.168701172f,
    (float16_t)0.059753418f, (float16_t)0.264892578f, (float16_t)-0.20324707f, (float16_t)-0.053619385f, (float16_t)-0.002866745f, (float16_t)-0.123596191f, (float16_t)0.263427734f, (float16_t)-0.112365723f, (float16_t)-0.187988281f, (float16_t)-0.225952148f, (float16_t)-0.160644531f, (float16_t)0.232666016f, (float16_t)0.239746094f, (float16_t)0.08807373f, (float16_t)-0.108642578f, (float16_t)0.024230957f,
    (float16_t)-0.098999023f, (float16_t)0.166625977f, (float16_t)0.283935547f, (float16_t)-0.229492188f, (float16_t)-0.069091797f, (float16_t)-0.009941101f, (float16_t)-0.171264648f, (float16_t)0.281738281f, (float16_t)-0.114196777f, (float16_t)-0.029800415f, (float16_t)0.103210449f, (float16_t)-0.17199707f, (float16_t)-0.058013916f, (float16_t)0.143188477f, (float16_t)0.284423828f, (float16_t)0.245483398f,
    (float16_t)0.252197266f, (float16_t)0.125366211f, (float16_t)-0.072631836f, (float16_t)0.274658203f, (float16_t)0.251708984f, (float16_t)0.213012695f, (float16_t)0.083068848f, (float16_t)0.04095459f, (float16_t)-0.088256836f, (float16_t)0.0859375f, (float16_t)0.112609863f, (float16_t)-0.138916016f, (float16_t)0.019226074f, (float16_t)0.264892578f, (float16_t)0.193481445f, (float16_t)-0.1484375f,
    (float16_t)0.053100586f, (float16_t)0.165161133f, (float16_t)0.276367188f, (float16_t)0.179199219f, (float16_t)0.190063477f, (float16_t)-0.112976074f, (float16_t)0.120422363f, (float16_t)0.272216797f, (float16_t)-0.084289551f, (float16_t)0.025360107f, (float16_t)-0.013168335f, (float16_t)0.088684082f, (float16_t)0.177978516f, (float16_t)-0.151611328f, (float16_t)-0.232055664f, (float16_t)0.048522949f,
    (float16_t)-0.112365723f, (float16_t)-0.257080078f, (float16_t)-0.123962402f, (float16_t)-0.070983887f, (float16_t)-0.281738281f, (float16_t)0.149536133f, (float16_t)-0.019851685f, (float16_t)-0.160888672f, (float16_t)-0.153808594f, (float16_t)0.014129639f, (float16_t)-0.096130371f, (float16_t)-0.224853516f, (float16_t)-0.046264648f, (float16_t)-0.163330078f, (float16_t)-0.047851562f, (float16_t)0.149902344f,
    (float16_t)0.251464844f, (float16_t)-0.01701355f, (float16_t)0.02154541f, (float16_t)-0.007911682f, (float16_t)0.019546509f, (float16_t)0.045074463f, (float16_t)-0.120910645f, (float16_t)-0.125244141f, (float16_t)-0.17175293f, (float16_t)-0.263427734f, (float16_t)0.090026855f, (float16_t)0.086853027f, (float16_t)-0.156005859f, (float16_t)0.266357422f, (float16_t)-0.054779053f, (float16_t)0.077880859f,
    (float16_t)-0.054382324f, (float16_t)-0.037506104f, (float16_t)-0.200805664f, (float16_t)-0.09161377f, (float16_t)0.151367188f, (float16_t)0.016952515f, (float16_t)0.237426758f
};

// Biases
static const float16_t convolve_float_default_f16_biases[] = {
    (float16_t)0.062561035f, (float16_t)-0.051025391f, (float16_t)0.204711914f, (float16_t)-0.001818657f, (float16_t)-0.025878906f
};

// Input data (for testing)
static const float16_t convolve_float_default_f16_input[] = {
    (float16_t)-0.925292969f, (float16_t)-0.945800781f, (float16_t)-0.49609375f, (float16_t)-0.189331055f, (float16_t)-0.146362305f, (float16_t)0.953125f, (float16_t)0.818359375f, (float16_t)0.630371094f, (float16_t)-0.124023438f, (float16_t)0.503417969f, (float16_t)-0.328369141f, (float16_t)-0.698730469f, (float16_t)-0.94140625f, (float16_t)-0.13671875f, (float16_t)-0.168457031f, (float16_t)0.779296875f,
    (float16_t)-0.84375f, (float16_t)0.361572266f, (float16_t)-0.029724121f, (float16_t)-0.361816406f, (float16_t)0.728515625f, (float16_t)0.100341797f, (float16_t)0.418945312f, (float16_t)-0.598144531f, (float16_t)-0.438232422f, (float16_t)-0.4296875f, (float16_t)-0.439453125f, (float16_t)-0.747070312f, (float16_t)-0.508300781f, (float16_t)0.659667969f, (float16_t)-0.052764893f, (float16_t)0.370849609f,
    (float16_t)0.092224121f, (float16_t)0.1015625f, (float16_t)0.852050781f, (float16_t)0.796875f, (float16_t)-0.699707031f, (float16_t)-0.827636719f, (float16_t)0.688476562f, (float16_t)0.668945312f, (float16_t)-0.008201599f, (float16_t)-0.680664062f, (float16_t)0.152099609f, (float16_t)0.730957031f, (float16_t)0.607421875f, (float16_t)0.832519531f, (float16_t)0.399658203f, (float16_t)-0.823730469f,
    (float16_t)0.844726562f, (float16_t)0.145996094f, (float16_t)0.476318359f, (float16_t)-0.621582031f, (float16_t)-0.456542969f, (float16_t)-0.834960938f, (float16_t)0.433105469f, (float16_t)0.067810059f, (float16_t)0.515625f, (float16_t)0.565429688f, (float16_t)0.321533203f, (float16_t)0.777832031f, (float16_t)0.567382812f, (float16_t)-0.575683594f, (float16_t)0.608886719f, (float16_t)0.588378906f,
    (float16_t)0.790527344f, (float16_t)-0.916015625f, (float16_t)-0.448974609f, (float16_t)0.897460938f, (float16_t)-0.974609375f, (float16_t)0.553710938f, (float16_t)-0.210327148f, (float16_t)0.712402344f, (float16_t)0.590332031f, (float16_t)0.308105469f, (float16_t)-0.704101562f, (float16_t)0.870117188f, (float16_t)0.769042969f, (float16_t)0.118713379f, (float16_t)0.935058594f, (float16_t)-0.548828125f,
    (float16_t)-0.644042969f, (float16_t)0.02961731f, (float16_t)-0.521972656f, (float16_t)0.424804688f, (float16_t)-0.392578125f, (float16_t)0.611328125f, (float16_t)-0.166015625f, (float16_t)-0.448730469f, (float16_t)-0.272460938f, (float16_t)0.522949219f, (float16_t)0.959960938f, (float16_t)-0.494140625f, (float16_t)0.450195312f, (float16_t)0.967773438f, (float16_t)-0.569335938f, (float16_t)-0.758300781f,
    (float16_t)0.026870728f, (float16_t)0.019958496f, (float16_t)0.615234375f, (float16_t)-0.107177734f, (float16_t)-0.848632812f, (float16_t)0.721191406f, (float16_t)-0.237670898f, (float16_t)0.386230469f, (float16_t)0.586425781f, (float16_t)-0.701171875f, (float16_t)0.522460938f, (float16_t)0.060180664f
};

// Expected output (golden)
static const float16_t convolve_float_default_f16_expected_output[] = {
    (float16_t)0.498535156f, (float16_t)0.671386719f, (float16_t)-0.010452271f, (float16_t)0.317138672f, (float16_t)0.485107422f, (float16_t)-0.250732422f, (float16_t)-0.164306641f, (float16_t)-0.043121338f, (float16_t)0.077880859f, (float16_t)-0.071716309f, (float16_t)0.498291016f, (float16_t)-0.292724609f, (float16_t)0.236572266f, (float16_t)-0.18371582f, (float16_t)-0.307617188f, (float16_t)0.016479492f,
    (float16_t)0.237670898f, (float16_t)0.188232422f, (float16_t)0.173461914f, (float16_t)-0.113647461f, (float16_t)-0.640625f, (float16_t)-0.252685547f, (float16_t)0.327880859f, (float16_t)0.051361084f, (float16_t)0.705566406f, (float16_t)0.282958984f, (float16_t)0.228881836f, (float16_t)-0.01890564f, (float16_t)-0.284912109f, (float16_t)-0.130859375f, (float16_t)-0.674316406f, (float16_t)-0.549804688f,
    (float16_t)0.957519531f, (float16_t)0.137817383f, (float16_t)-0.359375f, (float16_t)-0.194213867f, (float16_t)-0.351806641f, (float16_t)-0.461669922f, (float16_t)0.201293945f, (float16_t)-0.039489746f, (float16_t)0.954101562f, (float16_t)0.623046875f, (float16_t)-0.366210938f, (float16_t)0.318359375f, (float16_t)0.341064453f, (float16_t)-0.69921875f, (float16_t)-0.543945312f, (float16_t)0.106384277f,
    (float16_t)-0.052429199f, (float16_t)0.415527344f, (float16_t)1.223632812f, (float16_t)1.120117188f, (float16_t)-0.106323242f, (float16_t)-0.532226562f, (float16_t)-0.525390625f, (float16_t)-0.202636719f, (float16_t)-0.934082031f, (float16_t)0.541503906f, (float16_t)0.121582031f, (float16_t)-0.151733398f, (float16_t)-0.415527344f, (float16_t)-0.3359375f, (float16_t)0.218505859f, (float16_t)0.144287109f,
    (float16_t)0.330322266f, (float16_t)-0.033782959f, (float16_t)0.8359375f, (float16_t)-0.00447464f, (float16_t)-0.75390625f, (float16_t)0.002220154f, (float16_t)-1.17578125f, (float16_t)-1.150390625f, (float16_t)1.256835938f, (float16_t)-0.484619141f, (float16_t)-0.617675781f, (float16_t)0.641601562f, (float16_t)-0.281494141f, (float16_t)-0.113098145f, (float16_t)-0.258300781f, (float16_t)-0.960449219f,
    (float16_t)-0.146484375f, (float16_t)-1.306640625f, (float16_t)0.487304688f, (float16_t)-0.701660156f, (float16_t)0.156005859f, (float16_t)0.514648438f, (float16_t)0.530761719f, (float16_t)0.383789062f, (float16_t)-0.079589844f, (float16_t)0.34375f, (float16_t)-0.002954483f, (float16_t)-0.450195312f, (float16_t)-0.118041992f, (float16_t)-0.133300781f, (float16_t)-0.035644531f, (float16_t)-0.012863159f,
    (float16_t)-0.681152344f, (float16_t)0.955566406f, (float16_t)-0.385498047f, (float16_t)0.175537109f, (float16_t)0.605957031f, (float16_t)0.051422119f, (float16_t)0.12878418f, (float16_t)0.119873047f, (float16_t)0.090454102f, (float16_t)-0.190185547f, (float16_t)0.721679688f, (float16_t)0.254638672f, (float16_t)0.233886719f, (float16_t)-0.809570312f, (float16_t)0.074707031f, (float16_t)0.292480469f,
    (float16_t)0.306152344f, (float16_t)-0.190795898f, (float16_t)0.001652718f, (float16_t)0.100341797f, (float16_t)-0.481201172f, (float16_t)0.500976562f, (float16_t)-0.189086914f, (float16_t)-0.185791016f, (float16_t)-0.436279297f, (float16_t)-0.058685303f, (float16_t)0.671386719f, (float16_t)0.006145477f, (float16_t)-0.207641602f, (float16_t)-0.404052734f, (float16_t)-0.736328125f, (float16_t)0.082702637f,
    (float16_t)0.45703125f, (float16_t)0.013343811f, (float16_t)0.528808594f, (float16_t)0.584472656f, (float16_t)-0.147460938f, (float16_t)0.544433594f, (float16_t)-0.145874023f, (float16_t)-0.58203125f, (float16_t)0.179443359f, (float16_t)0.093139648f, (float16_t)-0.459228516f, (float16_t)0.650390625f, (float16_t)0.100524902f, (float16_t)1.02734375f, (float16_t)0.835449219f, (float16_t)0.159423828f,
    (float16_t)-0.19519043f, (float16_t)-0.054260254f, (float16_t)-0.194458008f, (float16_t)0.854980469f, (float16_t)0.314941406f, (float16_t)-0.034881592f, (float16_t)0.04498291f, (float16_t)-0.51953125f, (float16_t)0.469238281f, (float16_t)0.288330078f, (float16_t)0.143676758f, (float16_t)0.645019531f, (float16_t)1.16015625f, (float16_t)-0.229736328f, (float16_t)0.319580078f, (float16_t)0.389160156f,
    (float16_t)-0.60546875f, (float16_t)-0.084533691f, (float16_t)0.422851562f, (float16_t)0.564453125f, (float16_t)0.396972656f, (float16_t)0.227416992f, (float16_t)0.343994141f, (float16_t)-0.05078125f, (float16_t)0.052490234f, (float16_t)0.204589844f, (float16_t)-0.067382812f, (float16_t)0.323974609f, (float16_t)0.260742188f, (float16_t)-0.348388672f, (float16_t)0.011108398f, (float16_t)0.125732422f,
    (float16_t)-0.146728516f, (float16_t)0.680664062f, (float16_t)0.210693359f, (float16_t)-0.386962891f
};

#endif