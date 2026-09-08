#include "svdf_float_default_f32_svdf.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"

static float svdf_float_default_f32_state[12];
static float svdf_float_default_f32_scratch_input[4];
static float svdf_float_default_f32_scratch_output[4];
static float svdf_float_default_f32_output[4];

static int32_t run_svdf(void)
{
    cmsis_nn_context ctx = {0};
    // The static arrays are the allocation upper bound; the ctx sizes come from the
    // kernel's own sizers (issue #71).
    const int32_t scratch_input_bytes = arm_svdf_f32_input_ctx_get_buffer_size(&svdf_float_default_f32_input_dims, &svdf_float_default_f32_weights_feature_dims);
    const int32_t scratch_output_bytes = arm_svdf_f32_output_ctx_get_buffer_size(&svdf_float_default_f32_svdf_params, &svdf_float_default_f32_input_dims, &svdf_float_default_f32_weights_feature_dims);
    if (scratch_input_bytes < 0 || scratch_input_bytes > (int32_t)sizeof(svdf_float_default_f32_scratch_input) ||
        scratch_output_bytes < 0 || scratch_output_bytes > (int32_t)sizeof(svdf_float_default_f32_scratch_output))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    cmsis_nn_context input_ctx = {.buf = svdf_float_default_f32_scratch_input, .size = scratch_input_bytes};
    cmsis_nn_context output_ctx = {.buf = svdf_float_default_f32_scratch_output, .size = scratch_output_bytes};

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
    // Exact ctx scratch sizes (issue #71), measured by calling the real sizers on host;
    // target-invariant, so no ISA split.
    HELIA_VALIDATE_SCALAR_EQ_INT("Svdf", "input ctx size", 16,
                                 arm_svdf_f32_input_ctx_get_buffer_size(&svdf_float_default_f32_input_dims, &svdf_float_default_f32_weights_feature_dims));
    HELIA_VALIDATE_SCALAR_EQ_INT("Svdf", "output ctx size", 16,
                                 arm_svdf_f32_output_ctx_get_buffer_size(&svdf_float_default_f32_svdf_params, &svdf_float_default_f32_input_dims, &svdf_float_default_f32_weights_feature_dims));
    int32_t status = run_svdf();
    HELIA_VALIDATE_STATUS("Svdf", status);

    int failures = 0;
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