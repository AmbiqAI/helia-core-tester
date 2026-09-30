#include "concatenation_axis_x_f32_concatenation.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"



// The kernel this harness links must have the prototype the ns-cmsis-nn export records.
_Static_assert(__builtin_types_compatible_p(__typeof__(arm_concatenation_f32_x), void (const float32_t *, int32_t, int32_t, int32_t, int32_t, float32_t *, int32_t, uint32_t)),
               "arm_concatenation_f32_x: prototype differs from the kernel contract export; rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate");



#define CONCATENATION_AXIS_X_F32_OUTPUT_SIZE (1 * 2 * 6 * 3)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[CONCATENATION_AXIS_X_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} concatenation_axis_x_f32_output_guard;
#define concatenation_axis_x_f32_output (concatenation_axis_x_f32_output_guard.body)

// Array of input pointers
static const float* concatenation_axis_x_f32_input_ptrs[] = {
    concatenation_axis_x_f32_input1,
    concatenation_axis_x_f32_input2,
    concatenation_axis_x_f32_input3,
};

int32_t concatenation_axis_x_f32_run(
    const float* const* __restrict input_ptrs,
    float* __restrict output
) {
        // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).

    // The sizer's answer is checked before it becomes a context size (#133): a negative
    // answer is the documented out-of-range sentinel, and one above this case's static bound
    // means the generation-time bound and the shipped kernel disagree.


    arm_concatenation_f32_x(
        input_ptrs[0], /* input */
        (uint16_t)concatenation_axis_x_f32_input_x[0], /* input_x */
        (uint16_t)concatenation_axis_x_f32_input_y[0], /* input_y */
        (uint16_t)concatenation_axis_x_f32_input_z[0], /* input_z */
        (uint16_t)concatenation_axis_x_f32_input_w[0], /* input_w */
        output, /* output */
        (uint16_t)6, /* output_x */
        (uint32_t)concatenation_axis_x_f32_offsets[0] /* offset_x */
    );
    arm_concatenation_f32_x(
        input_ptrs[1], /* input */
        (uint16_t)concatenation_axis_x_f32_input_x[1], /* input_x */
        (uint16_t)concatenation_axis_x_f32_input_y[1], /* input_y */
        (uint16_t)concatenation_axis_x_f32_input_z[1], /* input_z */
        (uint16_t)concatenation_axis_x_f32_input_w[1], /* input_w */
        output, /* output */
        (uint16_t)6, /* output_x */
        (uint32_t)concatenation_axis_x_f32_offsets[1] /* offset_x */
    );
    arm_concatenation_f32_x(
        input_ptrs[2], /* input */
        (uint16_t)concatenation_axis_x_f32_input_x[2], /* input_x */
        (uint16_t)concatenation_axis_x_f32_input_y[2], /* input_y */
        (uint16_t)concatenation_axis_x_f32_input_z[2], /* input_z */
        (uint16_t)concatenation_axis_x_f32_input_w[2], /* input_w */
        output, /* output */
        (uint16_t)6, /* output_x */
        (uint32_t)concatenation_axis_x_f32_offsets[2] /* offset_x */
    );
    return ARM_CMSIS_NN_SUCCESS;
}


int32_t concatenation_axis_x_f32_test_case_run(void)
{
    HELIA_GUARD_ARM(concatenation_axis_x_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = concatenation_axis_x_f32_run(concatenation_axis_x_f32_input_ptrs, concatenation_axis_x_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(concatenation_axis_x_f32_output, "Concatenation output", failures);
    HELIA_VALIDATE_STATUS("Concatenation", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        concatenation_axis_x_f32_output,
        concatenation_axis_x_f32_expected_output,
        CONCATENATION_AXIS_X_F32_OUTPUT_SIZE,
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
    int32_t failures = concatenation_axis_x_f32_test_case_run();
    helia_test_finish(failures);
}