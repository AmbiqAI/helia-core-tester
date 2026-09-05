#include "softmax_float_default_f32_softmax.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

#define SOFTMAX_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 4 * 4 * 3)
static float softmax_float_default_f32_output[SOFTMAX_FLOAT_DEFAULT_F32_OUTPUT_SIZE];

int32_t softmax_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
    arm_cmsis_nn_status kernel_status = arm_softmax_f32(
        input,
        16,
        3,
        output
    );
    
    return kernel_status;
}

int32_t softmax_float_default_f32_test_case_run(void)
{
    int32_t status = softmax_float_default_f32_run(softmax_float_default_f32_input, softmax_float_default_f32_output);
    HELIA_VALIDATE_STATUS("Softmax", status);

    int failures = 0;
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