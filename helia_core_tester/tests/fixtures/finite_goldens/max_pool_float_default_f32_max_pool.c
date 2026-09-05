#include "max_pool_float_default_f32_max_pool.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

// Context for buffer allocation
static cmsis_nn_context max_pool_float_default_f32_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
// Buffer size: 0 for max pooling, input_channels * 4 for average pooling (DSP)
#define MAX_POOL_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX 0
// No buffer needed for max pooling
static uint8_t* max_pool_float_default_f32_buffer = NULL;

#define MAX_POOL_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 3 * 3 * 3)
static float max_pool_float_default_f32_output[MAX_POOL_FLOAT_DEFAULT_F32_OUTPUT_SIZE];

int32_t max_pool_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
    // Max pooling doesn't need a buffer
    max_pool_float_default_f32_ctx.buf = NULL;
    max_pool_float_default_f32_ctx.size = 0;

    // Call pooling kernel
    arm_cmsis_nn_status kernel_status = arm_max_pool_f32(
        &max_pool_float_default_f32_ctx,
        &max_pool_float_default_f32_pool_params,
        &max_pool_float_default_f32_input_dims,
        input,
        &max_pool_float_default_f32_filter_dims,
        &max_pool_float_default_f32_output_dims,
        output
    );
    
    return kernel_status;
}

int32_t max_pool_float_default_f32_test_case_run(void)
{
    int32_t status = max_pool_float_default_f32_run(max_pool_float_default_f32_input, max_pool_float_default_f32_output);
    HELIA_VALIDATE_STATUS("Maxpool", status);

    int failures = 0;
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        max_pool_float_default_f32_output,
        max_pool_float_default_f32_expected_output,
        MAX_POOL_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = max_pool_float_default_f32_test_case_run();
    helia_test_finish(failures);
}