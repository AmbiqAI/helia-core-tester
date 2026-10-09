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
    (float16_t)0.178833008f, (float16_t)-0.443359375f, (float16_t)0.458251953f, (float16_t)0.411621094f, (float16_t)-0.224365234f, (float16_t)0.439941406f, (float16_t)0.219604492f, (float16_t)-0.174926758f, (float16_t)0.279296875f, (float16_t)0.288574219f, (float16_t)-0.367919922f, (float16_t)0.285400391f, (float16_t)0.658691406f, (float16_t)-0.457275391f, (float16_t)0.33203125f, (float16_t)0.170532227f,
    (float16_t)-0.445068359f, (float16_t)0.518554688f, (float16_t)0.082702637f, (float16_t)-0.724121094f, (float16_t)0.267822266f, (float16_t)0.546386719f, (float16_t)-0.083068848f, (float16_t)0.506835938f, (float16_t)0.246948242f, (float16_t)-0.26953125f, (float16_t)0.290039062f, (float16_t)0.362548828f, (float16_t)-0.504394531f, (float16_t)0.241088867f, (float16_t)0.347167969f, (float16_t)-0.364013672f,
    (float16_t)0.423828125f, (float16_t)0.290039062f, (float16_t)-0.320800781f, (float16_t)0.330566406f, (float16_t)0.046844482f, (float16_t)-0.091186523f, (float16_t)0.297363281f, (float16_t)0.059814453f, (float16_t)-0.583984375f, (float16_t)0.568847656f, (float16_t)0.103759766f, (float16_t)-0.125976562f, (float16_t)0.00907135f, (float16_t)0.485595703f, (float16_t)-0.587402344f, (float16_t)0.101745605f,
    (float16_t)0.106262207f, (float16_t)-0.436035156f, (float16_t)0.654296875f, (float16_t)0.513183594f, (float16_t)0.031921387f, (float16_t)0.455566406f, (float16_t)-0.464111328f, (float16_t)0.04208374f, (float16_t)0.397949219f, (float16_t)0.526367188f, (float16_t)-0.167358398f, (float16_t)0.297119141f, (float16_t)-0.267089844f, (float16_t)0.00548172f, (float16_t)0.550292969f, (float16_t)0.481689453f,
    (float16_t)-0.193603516f, (float16_t)0.167114258f, (float16_t)0.647949219f, (float16_t)-0.370361328f, (float16_t)0.687988281f, (float16_t)0.153686523f, (float16_t)-0.232055664f, (float16_t)0.061920166f, (float16_t)0.303955078f, (float16_t)-0.319335938f, (float16_t)0.268066406f, (float16_t)0.488769531f, (float16_t)-0.504882812f, (float16_t)0.102722168f, (float16_t)0.532714844f, (float16_t)-0.534179688f,
    (float16_t)0.551269531f, (float16_t)0.541503906f, (float16_t)-0.334472656f, (float16_t)0.054260254f, (float16_t)0.711425781f, (float16_t)-0.700195312f, (float16_t)0.5703125f, (float16_t)0.476074219f, (float16_t)-3.400802612e-03f, (float16_t)0.143554688f, (float16_t)0.538085938f, (float16_t)-0.542480469f, (float16_t)0.390625f, (float16_t)0.294921875f, (float16_t)-0.074584961f, (float16_t)0.338378906f,
    (float16_t)0.730957031f, (float16_t)-0.108581543f, (float16_t)0.222900391f, (float16_t)0.166870117f, (float16_t)-0.28125f, (float16_t)0.045135498f, (float16_t)0.859863281f, (float16_t)-0.245361328f, (float16_t)0.269775391f, (float16_t)0.155273438f, (float16_t)-0.302246094f, (float16_t)0.557128906f, (float16_t)0.041320801f, (float16_t)-0.11151123f, (float16_t)-0.408203125f, (float16_t)0.436767578f,
    (float16_t)-0.388671875f, (float16_t)0.135742188f, (float16_t)0.152587891f, (float16_t)-0.063903809f, (float16_t)0.470947266f, (float16_t)0.327148438f, (float16_t)-0.5234375f, (float16_t)0.404785156f, (float16_t)0.458984375f, (float16_t)-0.463378906f, (float16_t)0.298339844f, (float16_t)0.315673828f, (float16_t)-0.078308105f, (float16_t)0.315429688f, (float16_t)-0.010688782f, (float16_t)-0.039215088f,
    (float16_t)0.234985352f, (float16_t)0.100280762f, (float16_t)-0.736328125f, (float16_t)0.516601562f, (float16_t)0.317626953f, (float16_t)-0.321289062f, (float16_t)0.10723877f, (float16_t)0.5625f, (float16_t)-0.31640625f, (float16_t)0.031982422f, (float16_t)0.349609375f, (float16_t)-0.375244141f, (float16_t)0.628417969f, (float16_t)0.268066406f, (float16_t)-0.514648438f, (float16_t)0.344238281f,
    (float16_t)0.278808594f, (float16_t)-0.224609375f, (float16_t)-0.105834961f, (float16_t)0.24609375f, (float16_t)-0.649902344f, (float16_t)-2.820968628e-03f, (float16_t)0.524902344f, (float16_t)-0.537109375f, (float16_t)0.788574219f, (float16_t)0.415527344f, (float16_t)-0.070495605f, (float16_t)0.452148438f, (float16_t)0.492919922f, (float16_t)-0.280517578f, (float16_t)0.055847168f, (float16_t)0.130004883f,
    (float16_t)-0.434570312f, (float16_t)-0.128417969f, (float16_t)0.665527344f, (float16_t)-0.067016602f, (float16_t)0.222290039f, (float16_t)0.281738281f, (float16_t)-0.264160156f, (float16_t)0.363037109f, (float16_t)0.4921875f, (float16_t)-0.492919922f, (float16_t)0.291259766f, (float16_t)0.355957031f, (float16_t)-0.050720215f, (float16_t)0.272705078f, (float16_t)0.337890625f, (float16_t)-0.354736328f,
    (float16_t)0.244506836f, (float16_t)0.299560547f, (float16_t)-0.309814453f, (float16_t)0.320800781f, (float16_t)0.533203125f, (float16_t)-0.529296875f, (float16_t)0.252685547f, (float16_t)0.3671875f, (float16_t)0.039245605f, (float16_t)0.265380859f, (float16_t)0.297119141f, (float16_t)-0.319335938f, (float16_t)0.237182617f, (float16_t)0.237182617f, (float16_t)-0.339355469f, (float16_t)0.387451172f,
};

#endif