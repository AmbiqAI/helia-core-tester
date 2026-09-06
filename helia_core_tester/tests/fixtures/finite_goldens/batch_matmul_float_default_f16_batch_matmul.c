#include "batch_matmul_float_default_f16_batch_matmul.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

// Context for buffer allocation
static cmsis_nn_context batch_matmul_float_default_f16_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
// Buffer size calculated conservatively to handle MVE and DSP implementations
#define BATCH_MATMUL_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX 1024
static uint8_t batch_matmul_float_default_f16_buffer[BATCH_MATMUL_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX];

#define BATCH_MATMUL_FLOAT_DEFAULT_F16_OUTPUT_SIZE (1 * 1 * 4 * 2)
static float16_t batch_matmul_float_default_f16_output[BATCH_MATMUL_FLOAT_DEFAULT_F16_OUTPUT_SIZE];

int32_t batch_matmul_float_default_f16_run(
    const float16_t* __restrict input_lhs,
    const float16_t* __restrict input_rhs,
    float16_t* __restrict output
) {
    // Calculate required buffer size
    int32_t required_buffer_size = arm_batch_matmul_f16_get_buffer_size(
        &batch_matmul_float_default_f16_bmm_params,
        &batch_matmul_float_default_f16_input_lhs_dims,
        &batch_matmul_float_default_f16_input_rhs_dims,
        &batch_matmul_float_default_f16_output_dims
    );

    // Sizer contract (issue #69): a negative return is the header's
    // out-of-range sentinel, never a size. The capacity check that follows is
    // the allocation guard for the static above, not a statement about the
    // sizer.
    HELIA_VALIDATE_SIZER("arm_batch_matmul_f16_get_buffer_size", required_buffer_size);
    HELIA_VALIDATE_SIZER_FITS("arm_batch_matmul_f16_get_buffer_size", required_buffer_size, BATCH_MATMUL_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX);

    // Initialize context buffer
    batch_matmul_float_default_f16_ctx.buf = batch_matmul_float_default_f16_buffer;
    batch_matmul_float_default_f16_ctx.size = required_buffer_size;

    // Call batch matmul kernel
    arm_cmsis_nn_status kernel_status = arm_batch_matmul_f16(
        &batch_matmul_float_default_f16_ctx,
        &batch_matmul_float_default_f16_bmm_params,
        &batch_matmul_float_default_f16_input_lhs_dims,
        input_lhs,
        &batch_matmul_float_default_f16_input_rhs_dims,
        input_rhs,
        &batch_matmul_float_default_f16_output_dims,
        output
    );
    
    return kernel_status;
}

int32_t batch_matmul_float_default_f16_test_case_run(void)
{
    int32_t status = batch_matmul_float_default_f16_run(batch_matmul_float_default_f16_input_lhs, batch_matmul_float_default_f16_input_rhs, batch_matmul_float_default_f16_output);
    HELIA_VALIDATE_STATUS("Batchmatmul", status);

    int failures = 0;
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        batch_matmul_float_default_f16_output,
        batch_matmul_float_default_f16_expected_output,
        BATCH_MATMUL_FLOAT_DEFAULT_F16_OUTPUT_SIZE,
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
    int32_t failures = batch_matmul_float_default_f16_test_case_run();
    helia_test_finish(failures);
}