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
    (float16_t)-0.331298828f, (float16_t)0.304199219f, (float16_t)0.170043945f, (float16_t)-0.199584961f, (float16_t)-0.164672852f, (float16_t)0.329589844f, (float16_t)0.244140625f, (float16_t)0.15625f, (float16_t)0.273193359f, (float16_t)-0.093933105f, (float16_t)0.163208008f, (float16_t)-0.071289062f, (float16_t)0.278808594f, (float16_t)-0.269775391f, (float16_t)-0.135864258f, (float16_t)0.020828247f,
    (float16_t)0.163085938f, (float16_t)0.217041016f, (float16_t)0.055358887f, (float16_t)0.231933594f, (float16_t)0.133300781f, (float16_t)-0.24206543f, (float16_t)0.234619141f, (float16_t)-0.313720703f, (float16_t)-0.217651367f, (float16_t)-0.143798828f, (float16_t)0.218994141f, (float16_t)0.36328125f, (float16_t)-0.160400391f, (float16_t)0.069091797f, (float16_t)0.048980713f, (float16_t)-0.086975098f,
    (float16_t)-0.132202148f, (float16_t)-0.322265625f, (float16_t)0.339599609f, (float16_t)-0.299316406f, (float16_t)-0.031311035f, (float16_t)-0.347412109f, (float16_t)-0.196777344f, (float16_t)-0.307617188f, (float16_t)0.333007812f, (float16_t)-0.021957397f, (float16_t)-0.047637939f, (float16_t)0.01651001f, (float16_t)-0.287841797f, (float16_t)0.122497559f, (float16_t)0.288574219f, (float16_t)-0.262207031f,
    (float16_t)-0.139160156f, (float16_t)-0.274902344f, (float16_t)-0.158935547f, (float16_t)-0.041290283f, (float16_t)-0.216918945f, (float16_t)0.317871094f
};

// Biases
static const float16_t transpose_conv_float_default_f16_biases[] = {
    (float16_t)0.319335938f, (float16_t)-0.336181641f, (float16_t)0.297607422f
};

// Input data (for testing)
static const float16_t transpose_conv_float_default_f16_input[] = {
    (float16_t)-5.030632019e-04f, (float16_t)-0.462402344f, (float16_t)-0.077148438f, (float16_t)0.088745117f, (float16_t)-0.980957031f, (float16_t)-0.090209961f, (float16_t)0.307373047f, (float16_t)-0.876464844f, (float16_t)0.381347656f, (float16_t)-0.694824219f, (float16_t)0.674316406f, (float16_t)-0.402099609f, (float16_t)0.736328125f, (float16_t)0.472167969f, (float16_t)0.131103516f, (float16_t)0.641113281f,
    (float16_t)0.191040039f, (float16_t)0.594726562f, (float16_t)-0.978027344f, (float16_t)-0.512207031f, (float16_t)0.752929688f, (float16_t)-0.399169922f, (float16_t)-0.295410156f, (float16_t)-0.313232422f, (float16_t)0.334716797f, (float16_t)0.583984375f, (float16_t)-0.039550781f, (float16_t)0.096618652f, (float16_t)0.439453125f, (float16_t)0.768554688f, (float16_t)-0.251464844f, (float16_t)0.142822266f,
};

// Expected output (golden)
static const float16_t transpose_conv_float_default_f16_expected_output[] = {
    (float16_t)0.178955078f, (float16_t)-0.443359375f, (float16_t)0.458251953f, (float16_t)0.411621094f, (float16_t)-0.224243164f, (float16_t)0.439941406f, (float16_t)0.219482422f, (float16_t)-0.174804688f, (float16_t)0.279296875f, (float16_t)0.288330078f, (float16_t)-0.367919922f, (float16_t)0.285400391f, (float16_t)0.659179688f, (float16_t)-0.457275391f, (float16_t)0.33203125f, (float16_t)0.170532227f,
    (float16_t)-0.4453125f, (float16_t)0.518554688f, (float16_t)0.082763672f, (float16_t)-0.724121094f, (float16_t)0.267822266f, (float16_t)0.546386719f, (float16_t)-0.083007812f, (float16_t)0.506835938f, (float16_t)0.246826172f, (float16_t)-0.269775391f, (float16_t)0.290039062f, (float16_t)0.362548828f, (float16_t)-0.504394531f, (float16_t)0.241210938f, (float16_t)0.347412109f, (float16_t)-0.364013672f,
    (float16_t)0.423828125f, (float16_t)0.290039062f, (float16_t)-0.320800781f, (float16_t)0.330810547f, (float16_t)0.046691895f, (float16_t)-0.091125488f, (float16_t)0.297363281f, (float16_t)0.059814453f, (float16_t)-0.583496094f, (float16_t)0.568847656f, (float16_t)0.103759766f, (float16_t)-0.125976562f, (float16_t)0.009056091f, (float16_t)0.485595703f, (float16_t)-0.587402344f, (float16_t)0.101745605f,
    (float16_t)0.106262207f, (float16_t)-0.436035156f, (float16_t)0.654296875f, (float16_t)0.513671875f, (float16_t)0.031829834f, (float16_t)0.455322266f, (float16_t)-0.463867188f, (float16_t)0.041992188f, (float16_t)0.397705078f, (float16_t)0.526855469f, (float16_t)-0.167358398f, (float16_t)0.297119141f, (float16_t)-0.267089844f, (float16_t)5.386352539e-03f, (float16_t)0.550292969f, (float16_t)0.481445312f,
    (float16_t)-0.193603516f, (float16_t)0.167236328f, (float16_t)0.647460938f, (float16_t)-0.370361328f, (float16_t)0.687988281f, (float16_t)0.153564453f, (float16_t)-0.231933594f, (float16_t)0.061828613f, (float16_t)0.303710938f, (float16_t)-0.319335938f, (float16_t)0.268066406f, (float16_t)0.488769531f, (float16_t)-0.504882812f, (float16_t)0.102722168f, (float16_t)0.532714844f, (float16_t)-0.534667969f,
    (float16_t)0.55078125f, (float16_t)0.541015625f, (float16_t)-0.334472656f, (float16_t)0.054260254f, (float16_t)0.711425781f, (float16_t)-0.700195312f, (float16_t)0.5703125f, (float16_t)0.476074219f, (float16_t)-0.003396988f, (float16_t)0.143432617f, (float16_t)0.538085938f, (float16_t)-0.542480469f, (float16_t)0.390625f, (float16_t)0.294921875f, (float16_t)-0.074462891f, (float16_t)0.338378906f,
    (float16_t)0.73046875f, (float16_t)-0.108276367f, (float16_t)0.22277832f, (float16_t)0.166992188f, (float16_t)-0.281005859f, (float16_t)0.045074463f, (float16_t)0.859863281f, (float16_t)-0.245483398f, (float16_t)0.269775391f, (float16_t)0.155151367f, (float16_t)-0.302001953f, (float16_t)0.557128906f, (float16_t)0.041534424f, (float16_t)-0.11151123f, (float16_t)-0.408203125f, (float16_t)0.436767578f,
    (float16_t)-0.388916016f, (float16_t)0.135620117f, (float16_t)0.152709961f, (float16_t)-0.063720703f, (float16_t)0.471191406f, (float16_t)0.327148438f, (float16_t)-0.523925781f, (float16_t)0.405029297f, (float16_t)0.458984375f, (float16_t)-0.463134766f, (float16_t)0.298339844f, (float16_t)0.315673828f, (float16_t)-0.078369141f, (float16_t)0.315429688f, (float16_t)-0.010696411f, (float16_t)-0.039367676f,
    (float16_t)0.235107422f, (float16_t)0.100280762f, (float16_t)-0.736328125f, (float16_t)0.516601562f, (float16_t)0.317626953f, (float16_t)-0.321289062f, (float16_t)0.107177734f, (float16_t)0.5625f, (float16_t)-0.316162109f, (float16_t)0.031982422f, (float16_t)0.349853516f, (float16_t)-0.375488281f, (float16_t)0.627929688f, (float16_t)0.268066406f, (float16_t)-0.514648438f, (float16_t)0.344238281f,
    (float16_t)0.278808594f, (float16_t)-0.224609375f, (float16_t)-0.105895996f, (float16_t)0.24621582f, (float16_t)-0.650390625f, (float16_t)-2.763748169e-03f, (float16_t)0.524902344f, (float16_t)-0.536621094f, (float16_t)0.788085938f, (float16_t)0.415283203f, (float16_t)-0.070495605f, (float16_t)0.452392578f, (float16_t)0.492919922f, (float16_t)-0.280517578f, (float16_t)0.055755615f, (float16_t)0.130004883f,
    (float16_t)-0.434570312f, (float16_t)-0.128417969f, (float16_t)0.665527344f, (float16_t)-0.066955566f, (float16_t)0.222412109f, (float16_t)0.281494141f, (float16_t)-0.264404297f, (float16_t)0.363037109f, (float16_t)0.492431641f, (float16_t)-0.492919922f, (float16_t)0.291259766f, (float16_t)0.355957031f, (float16_t)-0.05078125f, (float16_t)0.272949219f, (float16_t)0.337646484f, (float16_t)-0.354980469f,
    (float16_t)0.244628906f, (float16_t)0.299560547f, (float16_t)-0.309570312f, (float16_t)0.320800781f, (float16_t)0.533203125f, (float16_t)-0.529296875f, (float16_t)0.252441406f, (float16_t)0.3671875f, (float16_t)0.039215088f, (float16_t)0.265380859f, (float16_t)0.297363281f, (float16_t)-0.319335938f, (float16_t)0.237060547f, (float16_t)0.237304688f, (float16_t)-0.339355469f, (float16_t)0.387451172f,
};

#endif