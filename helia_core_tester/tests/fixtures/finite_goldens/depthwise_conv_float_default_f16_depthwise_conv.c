#include "depthwise_conv_float_default_f16_depthwise_conv.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

// Context for buffer allocation
static cmsis_nn_context depthwise_conv_float_default_f16_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
// Buffer size calculated conservatively to handle MVE and DSP implementations
#define DEPTHWISE_CONV_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX 1024
static uint8_t depthwise_conv_float_default_f16_buffer[DEPTHWISE_CONV_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX];


#define DEPTHWISE_CONV_FLOAT_DEFAULT_F16_OUTPUT_SIZE (1 * 6 * 6 * 3)
static float16_t depthwise_conv_float_default_f16_output[DEPTHWISE_CONV_FLOAT_DEFAULT_F16_OUTPUT_SIZE];

// Bias dimensions: bias shape is [1, 1, 1, C_OUT]
static const cmsis_nn_dims depthwise_conv_float_default_f16_bias_dims = {
    .n = 1, .h = 1,
    .w = 1, .c = 3
};

int32_t depthwise_conv_float_default_f16_run(
    const float16_t* __restrict input,
    float16_t* __restrict output
) {
    
    // Calculate required buffer size
    int32_t required_buffer_size = arm_depthwise_conv_f16_get_buffer_size(
        &depthwise_conv_float_default_f16_dw_conv_params,
        &depthwise_conv_float_default_f16_input_dims,
        &depthwise_conv_float_default_f16_filter_dims,
        &depthwise_conv_float_default_f16_output_dims,
        ARM_NN_LAYOUT_NHWC
    );

    // Sizer contract (issue #69): a negative return is the header's
    // out-of-range sentinel, never a size. The capacity check that follows is
    // the allocation guard for the static above, not a statement about the
    // sizer.
    HELIA_VALIDATE_SIZER("arm_depthwise_conv_f16_get_buffer_size", required_buffer_size);
    HELIA_VALIDATE_SIZER_FITS("arm_depthwise_conv_f16_get_buffer_size", required_buffer_size, DEPTHWISE_CONV_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX);

    // Initialize context buffer
    depthwise_conv_float_default_f16_ctx.buf = depthwise_conv_float_default_f16_buffer;
    depthwise_conv_float_default_f16_ctx.size = required_buffer_size;


    // Call depthwise convolution kernel
    arm_cmsis_nn_status kernel_status = arm_depthwise_conv_f16(
        &depthwise_conv_float_default_f16_ctx,
        &depthwise_conv_float_default_f16_dw_conv_params,
        &depthwise_conv_float_default_f16_input_dims,
        input,
        &depthwise_conv_float_default_f16_filter_dims,
        depthwise_conv_float_default_f16_weights,
        &depthwise_conv_float_default_f16_bias_dims,
        depthwise_conv_float_default_f16_biases,
        &depthwise_conv_float_default_f16_output_dims,
        output,
        ARM_NN_LAYOUT_NHWC
    );

    return kernel_status;
}

int32_t depthwise_conv_float_default_f16_test_case_run(void)
{
    int32_t status = depthwise_conv_float_default_f16_run(depthwise_conv_float_default_f16_input, depthwise_conv_float_default_f16_output);
    HELIA_VALIDATE_STATUS("Depthwiseconv", status);

    int failures = 0;
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        depthwise_conv_float_default_f16_output,
        depthwise_conv_float_default_f16_expected_output,
        DEPTHWISE_CONV_FLOAT_DEFAULT_F16_OUTPUT_SIZE,
        1,
        0.001f,
        0.001f,
        20,
        failures
    );
    HELIA_VALIDATE_RETURN_FAILURES(failures);
}

int main(void)
{
    helia_test_platform_init();
    int32_t failures = depthwise_conv_float_default_f16_test_case_run();
    helia_test_finish(failures);
}