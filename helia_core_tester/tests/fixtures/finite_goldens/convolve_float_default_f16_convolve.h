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
    (float16_t)0.065917969f, (float16_t)-0.117736816f, (float16_t)-0.224975586f, (float16_t)0.135620117f, (float16_t)0.088378906f, (float16_t)-0.279296875f, (float16_t)-0.233886719f, (float16_t)0.060852051f, (float16_t)-0.231079102f, (float16_t)-0.061859131f, (float16_t)-0.13671875f, (float16_t)-0.023864746f, (float16_t)0.108215332f, (float16_t)-0.198852539f, (float16_t)0.206420898f, (float16_t)0.119506836f,
    (float16_t)-0.01448822f, (float16_t)-0.065979004f, (float16_t)-0.215454102f, (float16_t)0.227050781f, (float16_t)-0.108032227f, (float16_t)0.240478516f, (float16_t)-0.149780273f, (float16_t)-0.055847168f, (float16_t)0.098144531f, (float16_t)0.203491211f, (float16_t)0.114135742f, (float16_t)0.014022827f, (float16_t)0.149780273f, (float16_t)0.03729248f, (float16_t)0.061950684f, (float16_t)-0.26171875f,
    (float16_t)-0.198242188f, (float16_t)-0.153686523f, (float16_t)-0.274658203f, (float16_t)0.104858398f, (float16_t)-0.074523926f, (float16_t)1.389503479e-03f, (float16_t)0.234741211f, (float16_t)-0.071594238f, (float16_t)-8.209228516e-03f, (float16_t)0.210571289f, (float16_t)0.167480469f, (float16_t)-0.079833984f, (float16_t)0.080383301f, (float16_t)0.019378662f, (float16_t)0.083862305f, (float16_t)-0.164306641f,
    (float16_t)-0.112121582f, (float16_t)0.024276733f, (float16_t)-0.256835938f, (float16_t)0.265380859f, (float16_t)-0.040161133f, (float16_t)0.202270508f, (float16_t)0.102111816f, (float16_t)0.179199219f, (float16_t)0.163696289f, (float16_t)-0.270751953f, (float16_t)0.220092773f, (float16_t)-0.094543457f, (float16_t)0.092529297f, (float16_t)-0.288574219f, (float16_t)-0.256835938f, (float16_t)-0.045379639f,
    (float16_t)0.26171875f, (float16_t)0.115539551f, (float16_t)0.019943237f, (float16_t)0.173095703f, (float16_t)0.184814453f, (float16_t)0.002527237f, (float16_t)-0.052215576f, (float16_t)-0.159545898f, (float16_t)0.079650879f, (float16_t)-0.112426758f, (float16_t)-0.207275391f, (float16_t)-0.277832031f, (float16_t)-0.176269531f, (float16_t)0.259033203f, (float16_t)0.249633789f, (float16_t)-0.152709961f,
    (float16_t)-0.071044922f, (float16_t)0.025253296f, (float16_t)-0.165283203f, (float16_t)0.224365234f, (float16_t)0.218139648f, (float16_t)0.118041992f, (float16_t)-0.215698242f, (float16_t)-0.119018555f, (float16_t)0.121887207f, (float16_t)-0.220214844f, (float16_t)-0.112609863f, (float16_t)-0.040466309f, (float16_t)-0.086975098f, (float16_t)-0.081054688f, (float16_t)-0.072692871f, (float16_t)6.328582764e-03f,
    (float16_t)0.123474121f, (float16_t)0.277832031f, (float16_t)-0.126586914f, (float16_t)0.115844727f, (float16_t)-0.269287109f, (float16_t)0.280761719f, (float16_t)0.147094727f, (float16_t)0.268066406f, (float16_t)-0.121582031f, (float16_t)-0.23840332f, (float16_t)-0.061401367f, (float16_t)-0.217529297f, (float16_t)-0.186401367f, (float16_t)-0.202636719f, (float16_t)0.107910156f, (float16_t)-0.120422363f,
    (float16_t)0.208129883f, (float16_t)0.121704102f, (float16_t)-0.101867676f, (float16_t)0.079101562f, (float16_t)0.009750366f, (float16_t)-0.011962891f, (float16_t)0.030715942f, (float16_t)-0.263183594f, (float16_t)0.030410767f, (float16_t)-0.090332031f, (float16_t)0.041992188f, (float16_t)-0.119750977f, (float16_t)-0.132324219f, (float16_t)-0.262207031f, (float16_t)0.259521484f, (float16_t)0.254638672f,
    (float16_t)0.153808594f, (float16_t)-0.129760742f, (float16_t)-0.130615234f, (float16_t)0.072692871f, (float16_t)-0.234863281f, (float16_t)-0.231079102f, (float16_t)0.04107666f
};

// Biases
static const float16_t convolve_float_default_f16_biases[] = {
    (float16_t)0.075622559f, (float16_t)-0.172119141f, (float16_t)0.1484375f, (float16_t)-0.148681641f, (float16_t)0.161499023f
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
    (float16_t)0.01096344f, (float16_t)-0.449707031f, (float16_t)-0.006008148f, (float16_t)-0.302246094f, (float16_t)-0.05078125f, (float16_t)0.235961914f, (float16_t)-0.114685059f, (float16_t)-0.360351562f, (float16_t)1.0859375f, (float16_t)0.2578125f, (float16_t)0.153442383f, (float16_t)0.272460938f, (float16_t)0.474121094f, (float16_t)-0.587402344f, (float16_t)0.479248047f, (float16_t)-0.197265625f,
    (float16_t)-0.682617188f, (float16_t)0.613769531f, (float16_t)-0.824707031f, (float16_t)0.245361328f, (float16_t)0.253173828f, (float16_t)-0.134521484f, (float16_t)-0.453613281f, (float16_t)-0.16394043f, (float16_t)-0.215332031f, (float16_t)0.421142578f, (float16_t)-0.294921875f, (float16_t)0.035217285f, (float16_t)0.044342041f, (float16_t)0.356445312f, (float16_t)-0.011726379f, (float16_t)0.301269531f,
    (float16_t)0.897949219f, (float16_t)-0.741699219f, (float16_t)0.250244141f, (float16_t)-0.049194336f, (float16_t)-0.8203125f, (float16_t)-0.87890625f, (float16_t)-0.289306641f, (float16_t)-0.093444824f, (float16_t)-0.246582031f, (float16_t)-0.540527344f, (float16_t)0.686523438f, (float16_t)0.178222656f, (float16_t)0.096740723f, (float16_t)1.224609375f, (float16_t)0.850585938f, (float16_t)-0.537109375f,
    (float16_t)0.392822266f, (float16_t)-0.284667969f, (float16_t)-0.212646484f, (float16_t)-0.180541992f, (float16_t)0.405029297f, (float16_t)-0.304443359f, (float16_t)-0.250488281f, (float16_t)-0.280761719f, (float16_t)0.388427734f, (float16_t)-0.034729004f, (float16_t)-0.197509766f, (float16_t)0.477539062f, (float16_t)0.607421875f, (float16_t)-0.041931152f, (float16_t)0.168823242f, (float16_t)-0.201049805f,
    (float16_t)0.174682617f, (float16_t)0.307861328f, (float16_t)-0.036865234f, (float16_t)0.2578125f, (float16_t)0.529296875f, (float16_t)0.022628784f, (float16_t)0.474609375f, (float16_t)-0.076293945f, (float16_t)0.393310547f, (float16_t)-0.215942383f, (float16_t)0.17956543f, (float16_t)-0.553710938f, (float16_t)-0.624023438f, (float16_t)-0.782226562f, (float16_t)0.427246094f, (float16_t)-0.340820312f,
    (float16_t)-0.100158691f, (float16_t)-0.019668579f, (float16_t)-0.0259552f, (float16_t)-0.602050781f, (float16_t)1.271484375f, (float16_t)0.172485352f, (float16_t)-0.590332031f, (float16_t)0.251708984f, (float16_t)-0.932617188f, (float16_t)0.194091797f, (float16_t)0.257568359f, (float16_t)0.247436523f, (float16_t)-3.442764282e-03f, (float16_t)-0.479736328f, (float16_t)-0.749023438f, (float16_t)0.279052734f,
    (float16_t)0.320800781f, (float16_t)-0.052612305f, (float16_t)0.084655762f, (float16_t)-0.052947998f, (float16_t)0.795410156f, (float16_t)-0.396972656f, (float16_t)0.303710938f, (float16_t)0.028274536f, (float16_t)0.544433594f, (float16_t)-0.44140625f, (float16_t)-0.624023438f, (float16_t)0.513183594f, (float16_t)0.182495117f, (float16_t)-0.17956543f, (float16_t)-0.342285156f, (float16_t)-0.363525391f,
    (float16_t)-0.019622803f, (float16_t)0.188110352f, (float16_t)-0.171508789f, (float16_t)0.437744141f, (float16_t)0.024200439f, (float16_t)0.977050781f, (float16_t)-0.439453125f, (float16_t)0.419433594f, (float16_t)-0.181518555f, (float16_t)-0.566894531f, (float16_t)-0.045410156f, (float16_t)-0.202270508f, (float16_t)-0.269042969f, (float16_t)-0.3046875f, (float16_t)-0.202636719f, (float16_t)0.107971191f,
    (float16_t)-0.099304199f, (float16_t)0.498535156f, (float16_t)-0.610839844f, (float16_t)-0.4921875f, (float16_t)0.552734375f, (float16_t)-0.327148438f, (float16_t)0.220703125f, (float16_t)1.131835938f, (float16_t)-1.006835938f, (float16_t)0.054382324f, (float16_t)0.792480469f, (float16_t)0.818847656f, (float16_t)-0.362792969f, (float16_t)-0.529296875f, (float16_t)0.276611328f, (float16_t)-0.192504883f,
    (float16_t)-0.464355469f, (float16_t)-0.099304199f, (float16_t)0.010673523f, (float16_t)-0.026123047f, (float16_t)-0.428710938f, (float16_t)0.188476562f, (float16_t)0.664550781f, (float16_t)-0.232421875f, (float16_t)0.121032715f, (float16_t)0.150634766f, (float16_t)0.293212891f, (float16_t)0.307373047f, (float16_t)-0.513671875f, (float16_t)0.042022705f, (float16_t)-0.362304688f, (float16_t)-0.425537109f,
    (float16_t)0.250976562f, (float16_t)0.456542969f, (float16_t)-0.108154297f, (float16_t)-0.469726562f, (float16_t)-0.357666016f, (float16_t)0.567871094f, (float16_t)-0.050537109f, (float16_t)-0.347900391f, (float16_t)-0.146484375f, (float16_t)-0.179199219f, (float16_t)0.086791992f, (float16_t)-0.027694702f, (float16_t)0.284667969f, (float16_t)-0.009429932f, (float16_t)0.258056641f, (float16_t)-0.435302734f,
    (float16_t)0.062072754f, (float16_t)0.469970703f, (float16_t)-0.560546875f, (float16_t)-0.051757812f
};

#endif