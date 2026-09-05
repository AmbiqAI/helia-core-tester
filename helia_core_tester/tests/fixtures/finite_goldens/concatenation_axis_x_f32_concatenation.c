#include "concatenation_axis_x_f32_concatenation.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

#define CONCATENATION_AXIS_X_F32_OUTPUT_SIZE (1 * 2 * 6 * 3)
static float concatenation_axis_x_f32_output[CONCATENATION_AXIS_X_F32_OUTPUT_SIZE];

// Array of input pointers
static const float* concatenation_axis_x_f32_input_ptrs[] = {
    concatenation_axis_x_f32_input1,    concatenation_axis_x_f32_input2,    concatenation_axis_x_f32_input3};

int32_t concatenation_axis_x_f32_run(
    const float* const *input_ptrs,
    float* __restrict output
) {
    // Call Concatenation kernel
    arm_cmsis_nn_status kernel_status = ARM_CMSIS_NN_SUCCESS;
    for (int i = 0; i < 3; i++) {
        arm_concatenation_f32_x(
            input_ptrs[i],
            (uint16_t)concatenation_axis_x_f32_input_x[i],
            (uint16_t)concatenation_axis_x_f32_input_y[i],
            (uint16_t)concatenation_axis_x_f32_input_z[i],
            (uint16_t)concatenation_axis_x_f32_input_w[i],
            output,
            (uint16_t)6,
            (uint32_t)concatenation_axis_x_f32_offsets[i]
        );
    }
    
    return kernel_status;
}

int32_t concatenation_axis_x_f32_test_case_run(void)
{
    int32_t status = concatenation_axis_x_f32_run(concatenation_axis_x_f32_input_ptrs, concatenation_axis_x_f32_output);
    HELIA_VALIDATE_STATUS("Concatenation", status);

    int failures = 0;
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