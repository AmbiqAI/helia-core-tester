#include "reduce_sum_float_axis_c_f32_reduce_sum.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"



// The kernel this harness links must have the prototype the ns-cmsis-nn export records.
_Static_assert(__builtin_types_compatible_p(__typeof__(arm_reduce_sum_f32), arm_cmsis_nn_status (const float32_t *, const cmsis_nn_dims *, const cmsis_nn_dims *, float32_t *, const cmsis_nn_dims *)),
               "arm_reduce_sum_f32: prototype differs from the kernel contract export; rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate");



#define REDUCE_SUM_FLOAT_AXIS_C_F32_OUTPUT_SIZE (1 * 3 * 4 * 1)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[REDUCE_SUM_FLOAT_AXIS_C_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} reduce_sum_float_axis_c_f32_output_guard;
#define reduce_sum_float_axis_c_f32_output (reduce_sum_float_axis_c_f32_output_guard.body)


int32_t reduce_sum_float_axis_c_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
        // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).

    // The sizer's answer is checked before it becomes a context size (#133): a negative
    // answer is the documented out-of-range sentinel, and one above this case's static bound
    // means the generation-time bound and the shipped kernel disagree.


    return arm_reduce_sum_f32(
        input, /* input_data */
        &reduce_sum_float_axis_c_f32_input_dims, /* input_dims */
        &reduce_sum_float_axis_c_f32_axis_dims, /* axis_dims */
        output, /* output_data */
        &reduce_sum_float_axis_c_f32_output_dims /* output_dims */
    );
}


int32_t reduce_sum_float_axis_c_f32_test_case_run(void)
{
    HELIA_GUARD_ARM(reduce_sum_float_axis_c_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = reduce_sum_float_axis_c_f32_run(reduce_sum_float_axis_c_f32_input, reduce_sum_float_axis_c_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(reduce_sum_float_axis_c_f32_output, "Reducesum output", failures);
    HELIA_VALIDATE_STATUS("Reducesum", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        reduce_sum_float_axis_c_f32_output,
        reduce_sum_float_axis_c_f32_expected_output,
        REDUCE_SUM_FLOAT_AXIS_C_F32_OUTPUT_SIZE,
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
    int32_t failures = reduce_sum_float_axis_c_f32_test_case_run();
    helia_test_finish(failures);
}