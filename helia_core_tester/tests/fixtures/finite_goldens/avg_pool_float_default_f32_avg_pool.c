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
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[AVG_POOL_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX];
    uint8_t tail[HELIA_GUARD_BYTES];
} avg_pool_float_default_f32_buffer_guard;
#define avg_pool_float_default_f32_buffer (avg_pool_float_default_f32_buffer_guard.body)

#define AVG_POOL_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 3 * 3 * 3)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[AVG_POOL_FLOAT_DEFAULT_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} avg_pool_float_default_f32_output_guard;
#define avg_pool_float_default_f32_output (avg_pool_float_default_f32_output_guard.body)

int32_t avg_pool_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
    // Armed unconditionally: declared/checked whenever buffer_size_max > 0
    // (see the DSP buffer-size comment above), even for kernel variants
    // (e.g. the float kernels below) that never query a buffer size and
    // leave ctx.buf NULL -- poisoning an unused buffer here is harmless,
    // and keeps this in lockstep with the CHECK in _test_case_run instead
    // of leaving its canary unstamped whenever kernel_get_buffer_size_fn
    // is absent.
    HELIA_GUARD_ARM(avg_pool_float_default_f32_buffer, true /* pure scratch: poison to catch read-before-write */);
    // Max pooling doesn't need a buffer
    avg_pool_float_default_f32_ctx.buf = NULL;
    avg_pool_float_default_f32_ctx.size = 0;
    HELIA_GUARD_STAMP_SLACK(avg_pool_float_default_f32_buffer, avg_pool_float_default_f32_ctx.buf == avg_pool_float_default_f32_buffer ? (size_t)avg_pool_float_default_f32_ctx.size : 0u);

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
    HELIA_GUARD_ARM(avg_pool_float_default_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = avg_pool_float_default_f32_run(avg_pool_float_default_f32_input, avg_pool_float_default_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(avg_pool_float_default_f32_buffer, "Avgpool scratch", failures);
    HELIA_GUARD_CHECK_SLACK(avg_pool_float_default_f32_buffer, "Avgpool scratch slack", avg_pool_float_default_f32_ctx.buf == avg_pool_float_default_f32_buffer ? (size_t)avg_pool_float_default_f32_ctx.size : 0u, failures);
    HELIA_GUARD_CHECK(avg_pool_float_default_f32_output, "Avgpool output", failures);
    HELIA_VALIDATE_STATUS("Avgpool", status);

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