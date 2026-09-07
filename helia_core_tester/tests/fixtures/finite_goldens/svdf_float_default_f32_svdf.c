#include "svdf_float_default_f32_svdf.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"


static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[12];
    uint8_t tail[HELIA_GUARD_BYTES];
} svdf_float_default_f32_state_guard;
#define svdf_float_default_f32_state (svdf_float_default_f32_state_guard.body)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[4];
    uint8_t tail[HELIA_GUARD_BYTES];
} svdf_float_default_f32_scratch_input_guard;
#define svdf_float_default_f32_scratch_input (svdf_float_default_f32_scratch_input_guard.body)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[4];
    uint8_t tail[HELIA_GUARD_BYTES];
} svdf_float_default_f32_scratch_output_guard;
#define svdf_float_default_f32_scratch_output (svdf_float_default_f32_scratch_output_guard.body)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[4];
    uint8_t tail[HELIA_GUARD_BYTES];
} svdf_float_default_f32_output_guard;
#define svdf_float_default_f32_output (svdf_float_default_f32_output_guard.body)

static int32_t run_svdf(void)
{
    cmsis_nn_context ctx = {0};
    cmsis_nn_context input_ctx = {.buf = svdf_float_default_f32_scratch_input, .size = sizeof(svdf_float_default_f32_scratch_input)};
    cmsis_nn_context output_ctx = {.buf = svdf_float_default_f32_scratch_output, .size = sizeof(svdf_float_default_f32_scratch_output)};

    HELIA_GUARD_ARM(svdf_float_default_f32_state, false /* recurrent state seeded below: don't poison */);
    HELIA_GUARD_ARM(svdf_float_default_f32_scratch_input, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_ARM(svdf_float_default_f32_scratch_output, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_ARM(svdf_float_default_f32_output, false /* real output, not scratch: don't poison */);

    for (int i = 0; i < 12; ++i)
    {
        svdf_float_default_f32_state[i] = svdf_float_default_f32_initial_state[i];
    }

    arm_cmsis_nn_status status = ARM_CMSIS_NN_SUCCESS;
    for (int step = 0; step < 2; ++step)
    {
        const float *step_input = svdf_float_default_f32_input_sequence + (step * 4);
        status = arm_svdf_f32(
            &ctx,
            &input_ctx,
            &output_ctx,
            &svdf_float_default_f32_svdf_params,
            &svdf_float_default_f32_input_dims,
            step_input,
            &svdf_float_default_f32_state_dims,
            svdf_float_default_f32_state,
            &svdf_float_default_f32_weights_feature_dims,
            svdf_float_default_f32_weights_feature,
            &svdf_float_default_f32_weights_time_dims,
            svdf_float_default_f32_weights_time,
            &svdf_float_default_f32_bias_dims,
svdf_float_default_f32_bias,
            &svdf_float_default_f32_output_dims,
            svdf_float_default_f32_output
        );
        if (status != ARM_CMSIS_NN_SUCCESS)
        {
            return status;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

int32_t svdf_float_default_f32_test_case_run(void)
{
    int32_t status = run_svdf();
    int failures = 0;
    HELIA_GUARD_CHECK(svdf_float_default_f32_state, "Svdf state", failures);
    HELIA_GUARD_CHECK(svdf_float_default_f32_scratch_input, "Svdf scratch_input", failures);
    HELIA_GUARD_CHECK(svdf_float_default_f32_scratch_output, "Svdf scratch_output", failures);
    HELIA_GUARD_CHECK(svdf_float_default_f32_output, "Svdf output", failures);
    HELIA_VALIDATE_STATUS("Svdf", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        svdf_float_default_f32_output,
        svdf_float_default_f32_expected_output,
        4,
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
    int32_t failures = svdf_float_default_f32_test_case_run();
    helia_test_finish(failures);
}