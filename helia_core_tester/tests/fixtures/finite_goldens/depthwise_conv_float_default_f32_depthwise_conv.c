#include "depthwise_conv_float_default_f32_depthwise_conv.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

// Context for buffer allocation
static cmsis_nn_context depthwise_conv_float_default_f32_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
// Buffer size calculated conservatively to handle MVE and DSP implementations
#define DEPTHWISE_CONV_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX 1024
static uint8_t depthwise_conv_float_default_f32_buffer[DEPTHWISE_CONV_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX];


#define DEPTHWISE_CONV_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 6 * 6 * 3)
static float depthwise_conv_float_default_f32_output[DEPTHWISE_CONV_FLOAT_DEFAULT_F32_OUTPUT_SIZE];

// Bias dimensions: bias shape is [1, 1, 1, C_OUT]
static const cmsis_nn_dims depthwise_conv_float_default_f32_bias_dims = {
    .n = 1, .h = 1,
    .w = 1, .c = 3
};

int32_t depthwise_conv_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
    
    // Calculate required buffer size
    int32_t required_buffer_size = arm_depthwise_conv_f32_get_buffer_size(
        &depthwise_conv_float_default_f32_dw_conv_params,
        &depthwise_conv_float_default_f32_input_dims,
        &depthwise_conv_float_default_f32_filter_dims,
        &depthwise_conv_float_default_f32_output_dims,
        ARM_NN_LAYOUT_NHWC
    );

    if (required_buffer_size > DEPTHWISE_CONV_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Initialize context buffer
    depthwise_conv_float_default_f32_ctx.buf = depthwise_conv_float_default_f32_buffer;
    depthwise_conv_float_default_f32_ctx.size = required_buffer_size;


    // Call depthwise convolution kernel
    arm_cmsis_nn_status kernel_status = arm_depthwise_conv_f32(
        &depthwise_conv_float_default_f32_ctx,
        &depthwise_conv_float_default_f32_dw_conv_params,
        &depthwise_conv_float_default_f32_input_dims,
        input,
        &depthwise_conv_float_default_f32_filter_dims,
        depthwise_conv_float_default_f32_weights,
        &depthwise_conv_float_default_f32_bias_dims,
        depthwise_conv_float_default_f32_biases,
        &depthwise_conv_float_default_f32_output_dims,
        output,
        ARM_NN_LAYOUT_NHWC
    );

    return kernel_status;
}

int32_t depthwise_conv_float_default_f32_test_case_run(void)
{
    int32_t status = depthwise_conv_float_default_f32_run(depthwise_conv_float_default_f32_input, depthwise_conv_float_default_f32_output);
    HELIA_VALIDATE_STATUS("Depthwiseconv", status);

    int failures = 0;
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        depthwise_conv_float_default_f32_output,
        depthwise_conv_float_default_f32_expected_output,
        DEPTHWISE_CONV_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
        1,
        5e-05f,
        2e-05f,
        20,
        failures
    );
    HELIA_VALIDATE_RETURN_FAILURES(failures);
}

int main(void)
{
    helia_test_platform_init();
    int32_t failures = depthwise_conv_float_default_f32_test_case_run();
    helia_test_finish(failures);
}