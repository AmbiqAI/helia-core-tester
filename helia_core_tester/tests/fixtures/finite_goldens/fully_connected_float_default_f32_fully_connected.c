#include "fully_connected_float_default_f32_fully_connected.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

// Context for buffer allocation
static cmsis_nn_context fully_connected_float_default_f32_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
// Buffer size calculated conservatively to handle MVE and DSP implementations
#define FULLY_CONNECTED_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX 1024
static uint8_t fully_connected_float_default_f32_buffer[FULLY_CONNECTED_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX];


#define FULLY_CONNECTED_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 1 * 1 * 5)
static float fully_connected_float_default_f32_output[FULLY_CONNECTED_FLOAT_DEFAULT_F32_OUTPUT_SIZE];

int32_t fully_connected_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
    // Calculate required buffer size
    int32_t required_buffer_size = arm_fully_connected_f32_get_buffer_size(
        &fully_connected_float_default_f32_fc_params,
        &fully_connected_float_default_f32_input_dims,
        &fully_connected_float_default_f32_filter_dims,
        &fully_connected_float_default_f32_output_dims,
        ARM_NN_LAYOUT_NHWC
    );

    if (required_buffer_size > FULLY_CONNECTED_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX) {
        printf("Buffer size error: required=%d > max=%d\r\n", required_buffer_size, FULLY_CONNECTED_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX);
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Initialize context buffer
    fully_connected_float_default_f32_ctx.buf = fully_connected_float_default_f32_buffer;
    fully_connected_float_default_f32_ctx.size = required_buffer_size;

    // Call fully connected kernel via wrapper (handles per-channel/per-tensor)
    arm_cmsis_nn_status kernel_status = arm_fully_connected_f32(
        &fully_connected_float_default_f32_ctx,
        &fully_connected_float_default_f32_fc_params,
        &fully_connected_float_default_f32_input_dims,
        input,
        &fully_connected_float_default_f32_filter_dims,
        fully_connected_float_default_f32_weights,
        &fully_connected_float_default_f32_bias_dims,
        fully_connected_float_default_f32_biases,
        &fully_connected_float_default_f32_output_dims,
        output,
        ARM_NN_LAYOUT_NHWC
    );
    
    return kernel_status;
}

int32_t fully_connected_float_default_f32_test_case_run(void)
{
    int32_t status = fully_connected_float_default_f32_run(fully_connected_float_default_f32_input, fully_connected_float_default_f32_output);
    HELIA_VALIDATE_STATUS("Fullyconnected", status);

    int failures = 0;
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        fully_connected_float_default_f32_output,
        fully_connected_float_default_f32_expected_output,
        FULLY_CONNECTED_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = fully_connected_float_default_f32_test_case_run();
    helia_test_finish(failures);
}