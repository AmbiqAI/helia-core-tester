#include "softmax_float_default_f32_softmax.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"



// The kernel this harness links must have the prototype the ns-cmsis-nn export records.
_Static_assert(__builtin_types_compatible_p(__typeof__(arm_softmax_f32), arm_cmsis_nn_status (const float32_t *, int32_t, int32_t, float32_t *)),
               "arm_softmax_f32: prototype differs from the kernel contract export; rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate");



#define SOFTMAX_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 4 * 4 * 3)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[SOFTMAX_FLOAT_DEFAULT_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} softmax_float_default_f32_output_guard;
#define softmax_float_default_f32_output (softmax_float_default_f32_output_guard.body)


int32_t softmax_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
        // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).

    // The sizer's answer is checked before it becomes a context size (#133): a negative
    // answer is the documented out-of-range sentinel, and one above this case's static bound
    // means the generation-time bound and the shipped kernel disagree.


    return arm_softmax_f32(
        input, /* input */
        16, /* num_rows */
        3, /* row_size */
        output /* output */
    );
}


int32_t softmax_float_default_f32_test_case_run(void)
{
    HELIA_GUARD_ARM(softmax_float_default_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = softmax_float_default_f32_run(softmax_float_default_f32_input, softmax_float_default_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(softmax_float_default_f32_output, "Softmax output", failures);
    HELIA_VALIDATE_STATUS("Softmax", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        softmax_float_default_f32_output,
        softmax_float_default_f32_expected_output,
        SOFTMAX_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = softmax_float_default_f32_test_case_run();
    helia_test_finish(failures);
}