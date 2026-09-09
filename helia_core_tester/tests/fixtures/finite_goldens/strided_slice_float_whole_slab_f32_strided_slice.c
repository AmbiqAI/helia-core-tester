#include "strided_slice_float_whole_slab_f32_strided_slice.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"


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
    // Call StridedSlice kernel
    arm_cmsis_nn_status kernel_status = arm_strided_slice_f32(
        input,
        output,
        &strided_slice_float_whole_slab_f32_input_dims,
        &strided_slice_float_whole_slab_f32_begin_dims,
        &strided_slice_float_whole_slab_f32_stride_dims,
        &strided_slice_float_whole_slab_f32_output_dims
    );
    
    return kernel_status;
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