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
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[BATCH_MATMUL_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX];
    uint8_t tail[HELIA_GUARD_BYTES];
} batch_matmul_float_default_f16_buffer_guard;
#define batch_matmul_float_default_f16_buffer (batch_matmul_float_default_f16_buffer_guard.body)

#define BATCH_MATMUL_FLOAT_DEFAULT_F16_OUTPUT_SIZE (1 * 1 * 4 * 2)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float16_t body[BATCH_MATMUL_FLOAT_DEFAULT_F16_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} batch_matmul_float_default_f16_output_guard;
#define batch_matmul_float_default_f16_output (batch_matmul_float_default_f16_output_guard.body)

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

    if (required_buffer_size > BATCH_MATMUL_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Initialize context buffer
    batch_matmul_float_default_f16_ctx.buf = batch_matmul_float_default_f16_buffer;
    batch_matmul_float_default_f16_ctx.size = required_buffer_size;
    HELIA_GUARD_ARM(batch_matmul_float_default_f16_buffer, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_STAMP_SLACK(batch_matmul_float_default_f16_buffer, batch_matmul_float_default_f16_ctx.buf == batch_matmul_float_default_f16_buffer ? (size_t)batch_matmul_float_default_f16_ctx.size : 0u);

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
    HELIA_GUARD_ARM(batch_matmul_float_default_f16_output, false /* real output, not scratch: don't poison */);
    int32_t status = batch_matmul_float_default_f16_run(batch_matmul_float_default_f16_input_lhs, batch_matmul_float_default_f16_input_rhs, batch_matmul_float_default_f16_output);
    int failures = 0;
    HELIA_GUARD_CHECK(batch_matmul_float_default_f16_buffer, "Batchmatmul scratch", failures);
    HELIA_GUARD_CHECK_SLACK(batch_matmul_float_default_f16_buffer, "Batchmatmul scratch slack", batch_matmul_float_default_f16_ctx.buf == batch_matmul_float_default_f16_buffer ? (size_t)batch_matmul_float_default_f16_ctx.size : 0u, failures);
    HELIA_GUARD_CHECK(batch_matmul_float_default_f16_output, "Batchmatmul output", failures);
    HELIA_VALIDATE_STATUS("Batchmatmul", status);

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