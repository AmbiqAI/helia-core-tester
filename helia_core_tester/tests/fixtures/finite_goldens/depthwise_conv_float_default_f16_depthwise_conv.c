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
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[DEPTHWISE_CONV_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX];
    uint8_t tail[HELIA_GUARD_BYTES];
} depthwise_conv_float_default_f16_buffer_guard;
#define depthwise_conv_float_default_f16_buffer (depthwise_conv_float_default_f16_buffer_guard.body)


#define DEPTHWISE_CONV_FLOAT_DEFAULT_F16_OUTPUT_SIZE (1 * 6 * 6 * 3)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float16_t body[DEPTHWISE_CONV_FLOAT_DEFAULT_F16_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} depthwise_conv_float_default_f16_output_guard;
#define depthwise_conv_float_default_f16_output (depthwise_conv_float_default_f16_output_guard.body)

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

    if (required_buffer_size > DEPTHWISE_CONV_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Initialize context buffer
    // Armed unconditionally: force_no_scratch bypasses depthwise_conv_float_default_f16_buffer entirely,
    // but guarding/poisoning it here regardless is harmless either way.
    HELIA_GUARD_ARM(depthwise_conv_float_default_f16_buffer, true /* pure scratch: poison to catch read-before-write */);
    depthwise_conv_float_default_f16_ctx.buf = depthwise_conv_float_default_f16_buffer;
    depthwise_conv_float_default_f16_ctx.size = required_buffer_size;
    HELIA_GUARD_STAMP_SLACK(depthwise_conv_float_default_f16_buffer, depthwise_conv_float_default_f16_ctx.buf == depthwise_conv_float_default_f16_buffer ? (size_t)depthwise_conv_float_default_f16_ctx.size : 0u);


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
    HELIA_GUARD_ARM(depthwise_conv_float_default_f16_output, false /* real output, not scratch: don't poison */);
    int32_t status = depthwise_conv_float_default_f16_run(depthwise_conv_float_default_f16_input, depthwise_conv_float_default_f16_output);
    int failures = 0;
    HELIA_GUARD_CHECK(depthwise_conv_float_default_f16_buffer, "Depthwiseconv scratch", failures);
    HELIA_GUARD_CHECK_SLACK(depthwise_conv_float_default_f16_buffer, "Depthwiseconv scratch slack", depthwise_conv_float_default_f16_ctx.buf == depthwise_conv_float_default_f16_buffer ? (size_t)depthwise_conv_float_default_f16_ctx.size : 0u, failures);
    HELIA_GUARD_CHECK(depthwise_conv_float_default_f16_output, "Depthwiseconv output", failures);
    HELIA_VALIDATE_STATUS("Depthwiseconv", status);

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