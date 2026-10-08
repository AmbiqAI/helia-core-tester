#include "transpose_float_default_f32_transpose.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"



// The kernel this harness links must have the prototype the ns-cmsis-nn export records.
_Static_assert(__builtin_types_compatible_p(__typeof__(arm_transpose_f32), arm_cmsis_nn_status (const cmsis_nn_context *, const cmsis_nn_transpose_params_f32 *, const cmsis_nn_dims *, const float32_t *, const cmsis_nn_dims *, float32_t *)),
               "arm_transpose_f32: prototype differs from the kernel contract export; rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate");



#define TRANSPOSE_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 3 * 2 * 4)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[TRANSPOSE_FLOAT_DEFAULT_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} transpose_float_default_f32_output_guard;
#define transpose_float_default_f32_output (transpose_float_default_f32_output_guard.body)

static cmsis_nn_context transpose_float_default_f32_ctx = {
    .buf = NULL,
    .size = 0
};

int32_t transpose_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
        // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).

    // The sizer's answer is checked before it becomes a context size (#133): a negative
    // answer is the documented out-of-range sentinel, and one above this case's static bound
    // means the generation-time bound and the shipped kernel disagree.


    return arm_transpose_f32(
        &transpose_float_default_f32_ctx, /* ctx */
        &transpose_float_default_f32_transpose_params, /* params */
        &transpose_float_default_f32_input_dims, /* input_dims */
        input, /* input */
        &transpose_float_default_f32_output_dims, /* output_dims */
        output /* output */
    );
}


int32_t transpose_float_default_f32_test_case_run(void)
{
    HELIA_GUARD_ARM(transpose_float_default_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = transpose_float_default_f32_run(transpose_float_default_f32_input, transpose_float_default_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(transpose_float_default_f32_output, "Transpose output", failures);
    HELIA_VALIDATE_STATUS("Transpose", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        transpose_float_default_f32_output,
        transpose_float_default_f32_expected_output,
        TRANSPOSE_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = transpose_float_default_f32_test_case_run();
    helia_test_finish(failures);
}