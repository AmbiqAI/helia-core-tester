#include "strided_slice_float_whole_slab_f32_strided_slice.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"



// The kernel this harness links must have the prototype the ns-cmsis-nn export records.
_Static_assert(__builtin_types_compatible_p(__typeof__(arm_strided_slice_f32), arm_cmsis_nn_status (const float32_t *, float32_t *, const cmsis_nn_dims * const, const cmsis_nn_dims * const, const cmsis_nn_dims * const, const cmsis_nn_dims * const)),
               "arm_strided_slice_f32: prototype differs from the kernel contract export; rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate");



#define STRIDED_SLICE_FLOAT_WHOLE_SLAB_F32_OUTPUT_SIZE (1 * 3 * 4 * 2)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[STRIDED_SLICE_FLOAT_WHOLE_SLAB_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} strided_slice_float_whole_slab_f32_output_guard;
#define strided_slice_float_whole_slab_f32_output (strided_slice_float_whole_slab_f32_output_guard.body)


int32_t strided_slice_float_whole_slab_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
        // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).

    // The sizer's answer is checked before it becomes a context size (#133): a negative
    // answer is the documented out-of-range sentinel, and one above this case's static bound
    // means the generation-time bound and the shipped kernel disagree.


    return arm_strided_slice_f32(
        input, /* input_data */
        output, /* output_data */
        &strided_slice_float_whole_slab_f32_input_dims, /* input_dims */
        &strided_slice_float_whole_slab_f32_begin_dims, /* begin_dims */
        &strided_slice_float_whole_slab_f32_stride_dims, /* stride_dims */
        &strided_slice_float_whole_slab_f32_output_dims /* output_dims */
    );
}


int32_t strided_slice_float_whole_slab_f32_test_case_run(void)
{
    HELIA_GUARD_ARM(strided_slice_float_whole_slab_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = strided_slice_float_whole_slab_f32_run(strided_slice_float_whole_slab_f32_input, strided_slice_float_whole_slab_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(strided_slice_float_whole_slab_f32_output, "StridedSlice output", failures);
    HELIA_VALIDATE_STATUS("StridedSlice", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        strided_slice_float_whole_slab_f32_output,
        strided_slice_float_whole_slab_f32_expected_output,
        STRIDED_SLICE_FLOAT_WHOLE_SLAB_F32_OUTPUT_SIZE,
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
    int32_t failures = strided_slice_float_whole_slab_f32_test_case_run();
    helia_test_finish(failures);
}