#include "sub_float_default_f32_sub.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"


#define SUB_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 4 * 4 * 8)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[SUB_FLOAT_DEFAULT_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} sub_float_default_f32_output_guard;
#define sub_float_default_f32_output (sub_float_default_f32_output_guard.body)

int32_t sub_float_default_f32_run(
    const float* __restrict input1,
    const float* __restrict input2,
    float* __restrict output
) {
    // Call subtract kernel
    arm_cmsis_nn_status kernel_status = arm_elementwise_sub_f32(
        input1,
        input2,
        output,
        -1.0e+30f,
        1.0e+30f,
        128
    );
    
    return kernel_status;
}

int32_t sub_float_default_f32_test_case_run(void)
{
    HELIA_GUARD_ARM(sub_float_default_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = sub_float_default_f32_run(sub_float_default_f32_input1, sub_float_default_f32_input2, sub_float_default_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(sub_float_default_f32_output, "Sub output", failures);
    HELIA_VALIDATE_STATUS("Sub", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        sub_float_default_f32_output,
        sub_float_default_f32_expected_output,
        SUB_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = sub_float_default_f32_test_case_run();
    helia_test_finish(failures);
}