#ifndef CONVOLVE_FLOAT_DEFAULT_F16_HARNESS_H
#define CONVOLVE_FLOAT_DEFAULT_F16_HARNESS_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

// Input dimensions
static const cmsis_nn_dims convolve_float_default_f16_input_dims = {
    .n = 1,
    .h = 6,
    .w = 6,
    .c = 3
};

// Filter dimensions
static const cmsis_nn_dims convolve_float_default_f16_filter_dims = {
    .n = 5,
    .h = 3,
    .w = 3,
    .c = 3
};

// Output dimensions
static const cmsis_nn_dims convolve_float_default_f16_output_dims = {
    .n = 1,
    .h = 6,
    .w = 6,
    .c = 5
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
    (float16_t)0.146728516f, (float16_t)-0.088562012f, (float16_t)0.131103516f, (float16_t)0.173461914f, (float16_t)-0.270263672f, (float16_t)0.116638184f, (float16_t)-0.038696289f, (float16_t)0.191040039f, (float16_t)-0.243041992f, (float16_t)0.202880859f, (float16_t)-0.058502197f, (float16_t)-0.197998047f, (float16_t)0.163574219f, (float16_t)0.038543701f, (float16_t)-0.01651001f, (float16_t)0.13293457f,
    (float16_t)0.256591797f, (float16_t)0.19909668f, (float16_t)0.085998535f, (float16_t)0.022293091f, (float16_t)-0.139892578f, (float16_t)0.027069092f, (float16_t)-0.147338867f, (float16_t)-0.22644043f, (float16_t)-0.136108398f, (float16_t)-0.137573242f, (float16_t)-0.17175293f, (float16_t)0.198852539f, (float16_t)-0.151855469f, (float16_t)-0.285400391f, (float16_t)0.151123047f, (float16_t)-0.086975098f,
    (float16_t)0.022232056f, (float16_t)0.203369141f, (float16_t)-0.212646484f, (float16_t)0.130371094f, (float16_t)-0.168334961f, (float16_t)-0.046051025f, (float16_t)0.27734375f, (float16_t)-0.228759766f, (float16_t)0.174072266f, (float16_t)-0.268554688f, (float16_t)0.282958984f, (float16_t)0.114135742f, (float16_t)-0.266357422f, (float16_t)-0.201171875f, (float16_t)0.19909668f, (float16_t)-0.236572266f,
    (float16_t)-0.230224609f, (float16_t)0.140625f, (float16_t)0.226928711f, (float16_t)-0.034820557f, (float16_t)0.065368652f, (float16_t)0.275878906f, (float16_t)0.272460938f, (float16_t)0.153564453f, (float16_t)-0.033172607f, (float16_t)0.144775391f, (float16_t)-0.16003418f, (float16_t)-0.244018555f, (float16_t)-0.077209473f, (float16_t)-0.211547852f, (float16_t)-0.084472656f, (float16_t)0.198852539f,
    (float16_t)-0.151123047f, (float16_t)0.013961792f, (float16_t)-0.212890625f, (float16_t)0.108764648f, (float16_t)0.202392578f, (float16_t)0.286132812f, (float16_t)-0.192993164f, (float16_t)-0.252929688f, (float16_t)-0.251953125f, (float16_t)0.245361328f, (float16_t)0.135742188f, (float16_t)-0.066711426f, (float16_t)-5.855560303e-03f, (float16_t)0.056976318f, (float16_t)-0.158813477f, (float16_t)-0.219604492f,
    (float16_t)-0.261474609f, (float16_t)-0.053588867f, (float16_t)0.180908203f, (float16_t)-0.228759766f, (float16_t)0.052032471f, (float16_t)0.192382812f, (float16_t)-0.23034668f, (float16_t)0.058624268f, (float16_t)-0.11126709f, (float16_t)-0.049041748f, (float16_t)-0.16809082f, (float16_t)-0.215698242f, (float16_t)0.095153809f, (float16_t)-0.273925781f, (float16_t)-0.204101562f, (float16_t)0.218994141f,
    (float16_t)-0.033508301f, (float16_t)-0.097167969f, (float16_t)0.130249023f, (float16_t)0.216918945f, (float16_t)0.101501465f, (float16_t)0.234375f, (float16_t)-0.061981201f, (float16_t)-0.02545166f, (float16_t)-0.15234375f, (float16_t)0.254882812f, (float16_t)-0.217773438f, (float16_t)-0.04107666f, (float16_t)-0.228149414f, (float16_t)-0.268798828f, (float16_t)-0.059234619f, (float16_t)-0.198974609f,
    (float16_t)3.662109375e-03f, (float16_t)-0.098449707f, (float16_t)0.138427734f, (float16_t)-0.223999023f, (float16_t)-0.087768555f, (float16_t)-0.176635742f, (float16_t)-0.170043945f, (float16_t)0.174926758f, (float16_t)-0.264404297f, (float16_t)-0.245849609f, (float16_t)-0.036834717f, (float16_t)0.189208984f, (float16_t)0.164428711f, (float16_t)0.109680176f, (float16_t)-0.262451172f, (float16_t)0.120788574f,
    (float16_t)0.194580078f, (float16_t)0.029403687f, (float16_t)0.09197998f, (float16_t)-0.102172852f, (float16_t)0.246582031f, (float16_t)0.036376953f, (float16_t)-0.018157959f
};

// Biases
static const float16_t convolve_float_default_f16_biases[] = {
    (float16_t)-0.002653122f, (float16_t)0.161132812f, (float16_t)-0.019943237f, (float16_t)-0.130981445f, (float16_t)-0.109558105f
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
    (float16_t)-0.136230469f, (float16_t)-2.595901489e-03f, (float16_t)-0.19921875f, (float16_t)0.210571289f, (float16_t)0.372802734f, (float16_t)0.314697266f, (float16_t)-0.151123047f, (float16_t)0.546386719f, (float16_t)0.568847656f, (float16_t)0.516601562f, (float16_t)0.093444824f, (float16_t)1.03125f, (float16_t)0.291992188f, (float16_t)-0.510742188f, (float16_t)-0.601074219f, (float16_t)-0.10168457f,
    (float16_t)0.146484375f, (float16_t)-0.500488281f, (float16_t)-0.998046875f, (float16_t)-0.788574219f, (float16_t)-0.453125f, (float16_t)0.412597656f, (float16_t)0.313964844f, (float16_t)-0.131103516f, (float16_t)0.399902344f, (float16_t)-0.375732422f, (float16_t)0.196777344f, (float16_t)-0.222900391f, (float16_t)-0.022918701f, (float16_t)0.115112305f, (float16_t)-0.272705078f, (float16_t)0.182495117f,
    (float16_t)0.461425781f, (float16_t)-0.009017944f, (float16_t)0.145019531f, (float16_t)-0.452636719f, (float16_t)0.375732422f, (float16_t)-1.153320312f, (float16_t)-0.662597656f, (float16_t)0.555664062f, (float16_t)-0.028274536f, (float16_t)-0.482421875f, (float16_t)-0.612304688f, (float16_t)0.058258057f, (float16_t)-0.200317383f, (float16_t)0.045928955f, (float16_t)-0.19152832f, (float16_t)0.642578125f,
    (float16_t)1.091796875f, (float16_t)0.374023438f, (float16_t)0.106018066f, (float16_t)0.861816406f, (float16_t)-0.061065674f, (float16_t)0.177124023f, (float16_t)0.377929688f, (float16_t)0.489013672f, (float16_t)-0.193725586f, (float16_t)-0.080932617f, (float16_t)0.125732422f, (float16_t)-0.431396484f, (float16_t)-0.17980957f, (float16_t)0.500976562f, (float16_t)0.009773254f, (float16_t)-0.045227051f,
    (float16_t)0.312744141f, (float16_t)-0.128295898f, (float16_t)0.259033203f, (float16_t)-0.529785156f, (float16_t)0.244750977f, (float16_t)0.5859375f, (float16_t)-0.102111816f, (float16_t)-0.108032227f, (float16_t)0.992675781f, (float16_t)-0.115112305f, (float16_t)-0.339599609f, (float16_t)0.489013672f, (float16_t)-0.328125f, (float16_t)-0.908203125f, (float16_t)-0.997558594f, (float16_t)0.041748047f,
    (float16_t)0.068237305f, (float16_t)-0.454101562f, (float16_t)-0.858398438f, (float16_t)-0.615234375f, (float16_t)-0.958007812f, (float16_t)-0.185424805f, (float16_t)0.801269531f, (float16_t)-0.024383545f, (float16_t)-0.516113281f, (float16_t)-0.120849609f, (float16_t)0.581054688f, (float16_t)-0.244140625f, (float16_t)-0.645996094f, (float16_t)-0.267822266f, (float16_t)0.565917969f, (float16_t)0.304443359f,
    (float16_t)-0.4921875f, (float16_t)-0.123474121f, (float16_t)1.825332642e-03f, (float16_t)-0.154174805f, (float16_t)0.57421875f, (float16_t)0.245849609f, (float16_t)0.011795044f, (float16_t)0.282470703f, (float16_t)-0.206542969f, (float16_t)0.138549805f, (float16_t)0.645507812f, (float16_t)-0.297607422f, (float16_t)-0.745117188f, (float16_t)-1.272460938f, (float16_t)0.548828125f, (float16_t)0.6328125f,
    (float16_t)0.422607422f, (float16_t)-0.422119141f, (float16_t)-0.855957031f, (float16_t)0.121765137f, (float16_t)-0.068786621f, (float16_t)0.456298828f, (float16_t)-0.457275391f, (float16_t)-0.477294922f, (float16_t)0.495117188f, (float16_t)0.321777344f, (float16_t)-0.301269531f, (float16_t)-0.435791016f, (float16_t)-0.041595459f, (float16_t)0.389160156f, (float16_t)-0.253417969f, (float16_t)-0.09765625f,
    (float16_t)-0.633300781f, (float16_t)-1.15234375f, (float16_t)1.077148438f, (float16_t)-0.310791016f, (float16_t)-0.719238281f, (float16_t)-0.985839844f, (float16_t)-1.547851562f, (float16_t)0.571289062f, (float16_t)-0.624023438f, (float16_t)0.164306641f, (float16_t)-0.164306641f, (float16_t)-0.488525391f, (float16_t)-1.198242188f, (float16_t)0.581542969f, (float16_t)0.206542969f, (float16_t)0.468994141f,
    (float16_t)-0.227539062f, (float16_t)-0.418212891f, (float16_t)0.47265625f, (float16_t)0.248168945f, (float16_t)0.460205078f, (float16_t)-0.02456665f, (float16_t)-0.019119263f, (float16_t)0.221069336f, (float16_t)0.359619141f, (float16_t)-0.058563232f, (float16_t)-0.361328125f, (float16_t)0.351318359f, (float16_t)0.437988281f, (float16_t)-0.017532349f, (float16_t)0.117919922f, (float16_t)-0.177368164f,
    (float16_t)0.390136719f, (float16_t)-0.303466797f, (float16_t)1.123046875f, (float16_t)0.256103516f, (float16_t)-0.884765625f, (float16_t)0.471435547f, (float16_t)0.111022949f, (float16_t)-0.132446289f, (float16_t)0.044372559f, (float16_t)0.029190063f, (float16_t)-0.406494141f, (float16_t)0.110168457f, (float16_t)-0.160522461f, (float16_t)0.154541016f, (float16_t)0.244018555f, (float16_t)-0.361816406f,
    (float16_t)0.425292969f, (float16_t)-0.055664062f, (float16_t)-0.046539307f, (float16_t)-4.062652588e-03f
};

#endif