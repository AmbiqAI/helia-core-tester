#include "abs_float_default_f32_abs.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

#define ABS_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 4 * 4 * 8)
static float abs_float_default_f32_output[ABS_FLOAT_DEFAULT_F32_OUTPUT_SIZE];

int32_t abs_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
    // Call abs kernel
    arm_cmsis_nn_status kernel_status = arm_nn_abs_f32(
        input,
        output,
        128           // block_size
    );

    return kernel_status;
}

int32_t abs_float_default_f32_test_case_run(void)
{
    int32_t status = abs_float_default_f32_run(abs_float_default_f32_input, abs_float_default_f32_output);
    HELIA_VALIDATE_STATUS("Abs", status);

    int failures = 0;
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        abs_float_default_f32_output,
        abs_float_default_f32_expected_output,
        ABS_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = abs_float_default_f32_test_case_run();
    helia_test_finish(failures);
}