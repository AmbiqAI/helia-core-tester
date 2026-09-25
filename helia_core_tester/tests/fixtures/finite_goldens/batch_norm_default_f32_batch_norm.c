#include "batch_norm_default_f32_batch_norm.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"


#define BATCH_NORM_DEFAULT_F32_OUTPUT_SIZE (1 * 4 * 4 * 3)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[BATCH_NORM_DEFAULT_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} batch_norm_default_f32_output_guard;
#define batch_norm_default_f32_output (batch_norm_default_f32_output_guard.body)

int32_t batch_norm_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
    arm_cmsis_nn_status kernel_status = arm_batch_norm_f32(
        input,
        output,
        batch_norm_default_f32_scale,
        batch_norm_default_f32_bias,
        &batch_norm_default_f32_input_dims,
        ARM_NN_LAYOUT_NHWC
    );
    return kernel_status;
}

int32_t batch_norm_default_f32_test_case_run(void)
{
    HELIA_GUARD_ARM(batch_norm_default_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = batch_norm_default_f32_run(batch_norm_default_f32_input, batch_norm_default_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(batch_norm_default_f32_output, "Batchnorm output", failures);
    HELIA_VALIDATE_STATUS("Batchnorm", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        batch_norm_default_f32_output,
        batch_norm_default_f32_expected_output,
        BATCH_NORM_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = batch_norm_default_f32_test_case_run();
    helia_test_finish(failures);
}