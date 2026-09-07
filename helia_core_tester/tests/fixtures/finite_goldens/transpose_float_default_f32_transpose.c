#include "transpose_float_default_f32_transpose.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

static cmsis_nn_context transpose_float_default_f32_ctx = { .buf = NULL, .size = 0 };

#define TRANSPOSE_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 3 * 2 * 4)
static float transpose_float_default_f32_output[TRANSPOSE_FLOAT_DEFAULT_F32_OUTPUT_SIZE];

int32_t transpose_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
    // Call Transpose kernel
    arm_cmsis_nn_status kernel_status = arm_transpose_f32(
        &transpose_float_default_f32_ctx,
        &transpose_float_default_f32_transpose_params,
        &transpose_float_default_f32_input_dims,
        input,
        &transpose_float_default_f32_output_dims,
        output
    );
    
    return kernel_status;
}

int32_t transpose_float_default_f32_test_case_run(void)
{
    int32_t status = transpose_float_default_f32_run(transpose_float_default_f32_input, transpose_float_default_f32_output);
    HELIA_VALIDATE_EXPECTED_STATUS(
        "Transpose",
        status,
        ARM_CMSIS_NN_SUCCESS
    );

    int failures = 0;
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