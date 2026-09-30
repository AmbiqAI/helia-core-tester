#include "avg_pool_float_default_f32_avg_pool.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"



// The kernel this harness links must have the prototype the ns-cmsis-nn export records.
_Static_assert(__builtin_types_compatible_p(__typeof__(arm_avg_pool_f32), arm_cmsis_nn_status (const cmsis_nn_context *, const cmsis_nn_pool_params_f32 *, const cmsis_nn_dims *, const float32_t *, const cmsis_nn_dims *, const cmsis_nn_dims *, float32_t *)),
               "arm_avg_pool_f32: prototype differs from the kernel contract export; rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate");

// Context for buffer allocation
static cmsis_nn_context avg_pool_float_default_f32_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
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
        // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).
    HELIA_GUARD_ARM(avg_pool_float_default_f32_buffer, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_STAMP_SLACK(avg_pool_float_default_f32_buffer, 0u);

    // The sizer's answer is checked before it becomes a context size (#133): a negative
    // answer is the documented out-of-range sentinel, and one above this case's static bound
    // means the generation-time bound and the shipped kernel disagree.

    // Initialize context buffer
    // The kernel gets no scratch buffer.
    avg_pool_float_default_f32_ctx.buf = NULL;
    avg_pool_float_default_f32_ctx.size = 0;
    HELIA_GUARD_STAMP_SLACK(avg_pool_float_default_f32_buffer, avg_pool_float_default_f32_ctx.buf == avg_pool_float_default_f32_buffer ? (size_t)avg_pool_float_default_f32_ctx.size : 0u);

    return arm_avg_pool_f32(
        &avg_pool_float_default_f32_ctx, /* ctx */
        &avg_pool_float_default_f32_pool_params, /* pool_params */
        &avg_pool_float_default_f32_input_dims, /* input_dims */
        input, /* src */
        &avg_pool_float_default_f32_filter_dims, /* filter_dims */
        &avg_pool_float_default_f32_output_dims, /* output_dims */
        output /* dst */
    );
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