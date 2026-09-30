#include "minimum_float_default_f32_minmax.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"



// The kernel this harness links must have the prototype the ns-cmsis-nn export records.
_Static_assert(__builtin_types_compatible_p(__typeof__(arm_minimum_f32), arm_cmsis_nn_status (const cmsis_nn_context *, const float32_t *, const cmsis_nn_dims *, const float32_t *, const cmsis_nn_dims *, float32_t *, const cmsis_nn_dims *)),
               "arm_minimum_f32: prototype differs from the kernel contract export; rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate");



#define MINIMUM_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 4 * 4 * 8)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[MINIMUM_FLOAT_DEFAULT_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} minimum_float_default_f32_output_guard;
#define minimum_float_default_f32_output (minimum_float_default_f32_output_guard.body)

static cmsis_nn_context minimum_float_default_f32_ctx = {
    .buf = NULL,
    .size = 0
};

int32_t minimum_float_default_f32_run(
    const float* __restrict input1,
    const float* __restrict input2,
    float* __restrict output
) {
        // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).

    // The sizer's answer is checked before it becomes a context size (#133): a negative
    // answer is the documented out-of-range sentinel, and one above this case's static bound
    // means the generation-time bound and the shipped kernel disagree.


    return arm_minimum_f32(
        &minimum_float_default_f32_ctx, /* ctx */
        input1, /* input_1_data */
        &minimum_float_default_f32_input1_dims, /* input_1_dims */
        input2, /* input_2_data */
        &minimum_float_default_f32_input2_dims, /* input_2_dims */
        output, /* output_data */
        &minimum_float_default_f32_output_dims /* output_dims */
    );
}


int32_t minimum_float_default_f32_test_case_run(void)
{
    HELIA_GUARD_ARM(minimum_float_default_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = minimum_float_default_f32_run(minimum_float_default_f32_input1, minimum_float_default_f32_input2, minimum_float_default_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(minimum_float_default_f32_output, "Minimum output", failures);
    HELIA_VALIDATE_STATUS("Minimum", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        minimum_float_default_f32_output,
        minimum_float_default_f32_expected_output,
        MINIMUM_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = minimum_float_default_f32_test_case_run();
    helia_test_finish(failures);
}