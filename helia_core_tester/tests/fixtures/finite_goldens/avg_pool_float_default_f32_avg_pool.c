#include "avg_pool_float_default_f32_avg_pool.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

// Context for buffer allocation
static cmsis_nn_context avg_pool_float_default_f32_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
// Buffer size: 0 for max pooling, input_channels * 4 for average pooling (DSP)
#define AVG_POOL_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX 12
static uint8_t avg_pool_float_default_f32_buffer[AVG_POOL_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX];

#define AVG_POOL_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 3 * 3 * 3)
static float avg_pool_float_default_f32_output[AVG_POOL_FLOAT_DEFAULT_F32_OUTPUT_SIZE];

int32_t avg_pool_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
    // Max pooling doesn't need a buffer
    avg_pool_float_default_f32_ctx.buf = NULL;
    avg_pool_float_default_f32_ctx.size = 0;

    // Call pooling kernel
    arm_cmsis_nn_status kernel_status = arm_avg_pool_f32(
        &avg_pool_float_default_f32_ctx,
        &avg_pool_float_default_f32_pool_params,
        &avg_pool_float_default_f32_input_dims,
        input,
        &avg_pool_float_default_f32_filter_dims,
        &avg_pool_float_default_f32_output_dims,
        output
    );
    
    return kernel_status;
}

int32_t avg_pool_float_default_f32_test_case_run(void)
{
    int32_t status = avg_pool_float_default_f32_run(avg_pool_float_default_f32_input, avg_pool_float_default_f32_output);
    HELIA_VALIDATE_STATUS("Avgpool", status);

    int failures = 0;
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        avg_pool_float_default_f32_output,
        avg_pool_float_default_f32_expected_output,
        AVG_POOL_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = avg_pool_float_default_f32_test_case_run();
    helia_test_finish(failures);
}