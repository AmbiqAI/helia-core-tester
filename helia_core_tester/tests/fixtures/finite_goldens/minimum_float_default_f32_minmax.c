#include "minimum_float_default_f32_minmax.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

// Context for buffer allocation (min/max operations don't need a buffer, but API requires ctx)
static cmsis_nn_context minimum_float_default_f32_ctx = { .buf = NULL, .size = 0 };

#define MINIMUM_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 4 * 4 * 8)
static float minimum_float_default_f32_output[MINIMUM_FLOAT_DEFAULT_F32_OUTPUT_SIZE];

int32_t minimum_float_default_f32_run(
    const float* __restrict input1,
    const float* __restrict input2,
    float* __restrict output
) {
    // Call min/max kernel (ctx can be NULL or empty for min/max operations)
    arm_cmsis_nn_status kernel_status = arm_minimum_f32(
        &minimum_float_default_f32_ctx,
        input1,
        &minimum_float_default_f32_input1_dims,
        input2,
        &minimum_float_default_f32_input2_dims,
        output,
        &minimum_float_default_f32_output_dims
    );
    
    return kernel_status;
}

int32_t minimum_float_default_f32_test_case_run(void)
{
    int32_t status = minimum_float_default_f32_run(minimum_float_default_f32_input1, minimum_float_default_f32_input2, minimum_float_default_f32_output);
    HELIA_VALIDATE_STATUS("Minimum", status);

    int failures = 0;
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