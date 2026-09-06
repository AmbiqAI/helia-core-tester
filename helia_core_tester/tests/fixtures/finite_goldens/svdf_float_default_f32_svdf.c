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

    /*
     * Sized by the shipped queries rather than by repeating their formulas
     * here (issue #71). The float half of the public header (arm_nnfunctions_flt)
     * documents both as returning "-1 if either pointer is NULL, if
     * input_dims->n or weights_feature_dims->n is negative, or if the product
     * would not fit in an int32_t", with "the figure and the validation ... the
     * same on every build target", so there is no _dsp / _mve variant to pick
     * between here.
     * The statics stay as the allocation; the queried figure is what the
     * kernel is told, and it has to fit inside them.
     */
    const int32_t input_ctx_size = arm_svdf_f32_input_ctx_get_buffer_size(
        &svdf_float_default_f32_input_dims,
        &svdf_float_default_f32_weights_feature_dims);
    const int32_t output_ctx_size = arm_svdf_f32_output_ctx_get_buffer_size(
        &svdf_float_default_f32_svdf_params,
        &svdf_float_default_f32_input_dims,
        &svdf_float_default_f32_weights_feature_dims);

    HELIA_VALIDATE_SIZER("arm_svdf_f32_input_ctx_get_buffer_size", input_ctx_size);
    HELIA_VALIDATE_SIZER_FITS("arm_svdf_f32_input_ctx_get_buffer_size", input_ctx_size, (int32_t)sizeof(svdf_float_default_f32_scratch_input));
    HELIA_VALIDATE_SIZER("arm_svdf_f32_output_ctx_get_buffer_size", output_ctx_size);
    HELIA_VALIDATE_SIZER_FITS("arm_svdf_f32_output_ctx_get_buffer_size", output_ctx_size, (int32_t)sizeof(svdf_float_default_f32_scratch_output));

    /*
     * Non-NULL even when the query answers 0: the same header records that "a
     * 0 here does NOT mean you may pass { NULL, 0 }". A static array is
     * always non-NULL, so the rule is honoured by construction.
     */
    cmsis_nn_context input_ctx = {.buf = svdf_float_default_f32_scratch_input, .size = input_ctx_size};
    cmsis_nn_context output_ctx = {.buf = svdf_float_default_f32_scratch_output, .size = output_ctx_size};

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