#include "batch_norm_default_f32_batch_norm.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"



// The kernel this harness links must have the prototype the ns-cmsis-nn export records.
_Static_assert(__builtin_types_compatible_p(__typeof__(arm_batch_norm_f32), arm_cmsis_nn_status (const float32_t *, float32_t *, const float32_t *, const float32_t *, const cmsis_nn_dims *, arm_nn_tensor_layout)),
               "arm_batch_norm_f32: prototype differs from the kernel contract export; rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate");



#define BATCH_NORM_DEFAULT_F32_OUTPUT_SIZE (1 * 4 * 4 * 3)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[BATCH_NORM_DEFAULT_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} batch_norm_default_f32_output_guard;
#define batch_norm_default_f32_output (batch_norm_default_f32_output_guard.body)


int32_t batch_norm_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
        // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).

    // The sizer's answer is checked before it becomes a context size (#133): a negative
    // answer is the documented out-of-range sentinel, and one above this case's static bound
    // means the generation-time bound and the shipped kernel disagree.


    return arm_batch_norm_f32(
        input, /* input */
        output, /* output */
        batch_norm_default_f32_scale, /* scale */
        batch_norm_default_f32_bias, /* bias */
        &batch_norm_default_f32_input_dims, /* input_dims */
        ARM_NN_LAYOUT_NHWC /* layout */
    );
}


int32_t batch_norm_default_f32_test_case_run(void)
{
    HELIA_GUARD_ARM(batch_norm_default_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = batch_norm_default_f32_run(batch_norm_default_f32_input, batch_norm_default_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(batch_norm_default_f32_output, "Batchnorm output", failures);
    HELIA_VALIDATE_STATUS("Batchnorm", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        batch_norm_default_f32_output,
        batch_norm_default_f32_expected_output,
        BATCH_NORM_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = batch_norm_default_f32_test_case_run();
    helia_test_finish(failures);
}