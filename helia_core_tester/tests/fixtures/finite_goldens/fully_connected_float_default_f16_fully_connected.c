#include "fully_connected_float_default_f16_fully_connected.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"



// The kernel this harness links must have the prototype the ns-cmsis-nn export records.
_Static_assert(__builtin_types_compatible_p(__typeof__(arm_fully_connected_f16), arm_cmsis_nn_status (const cmsis_nn_context *, const cmsis_nn_fc_params_f16 *, const cmsis_nn_dims *, const float16_t *, const cmsis_nn_dims *, const float16_t *, const cmsis_nn_dims *, const float16_t *, const cmsis_nn_dims *, float16_t *, arm_nn_tensor_layout)),
               "arm_fully_connected_f16: prototype differs from the kernel contract export; rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate");

// Context for buffer allocation
static cmsis_nn_context fully_connected_float_default_f16_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
#define FULLY_CONNECTED_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX 1024
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[FULLY_CONNECTED_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX];
    uint8_t tail[HELIA_GUARD_BYTES];
} fully_connected_float_default_f16_buffer_guard;
#define fully_connected_float_default_f16_buffer (fully_connected_float_default_f16_buffer_guard.body)


#define FULLY_CONNECTED_FLOAT_DEFAULT_F16_OUTPUT_SIZE (1 * 1 * 1 * 5)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float16_t body[FULLY_CONNECTED_FLOAT_DEFAULT_F16_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} fully_connected_float_default_f16_output_guard;
#define fully_connected_float_default_f16_output (fully_connected_float_default_f16_output_guard.body)


int32_t fully_connected_float_default_f16_run(
    const float16_t* __restrict input,
    float16_t* __restrict output
) {
        // Calculate required buffer size
    int32_t required_buffer_size = arm_fully_connected_f16_get_buffer_size(
        &fully_connected_float_default_f16_fc_params, /* fc_params */
        &fully_connected_float_default_f16_input_dims, /* input_dims */
        &fully_connected_float_default_f16_filter_dims, /* filter_dims */
        &fully_connected_float_default_f16_output_dims, /* output_dims */
        ARM_NN_LAYOUT_NHWC /* layout */
    );
    // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).
    HELIA_GUARD_ARM(fully_connected_float_default_f16_buffer, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_STAMP_SLACK(fully_connected_float_default_f16_buffer, 0u);

    // The sizer's answer is checked before it becomes a context size (#133): a negative
    // answer is the documented out-of-range sentinel, and one above this case's static bound
    // means the generation-time bound and the shipped kernel disagree.
    HELIA_VALIDATE_SIZER("arm_fully_connected_f16_get_buffer_size", required_buffer_size);
    HELIA_VALIDATE_SIZER_FITS("arm_fully_connected_f16_get_buffer_size", required_buffer_size, FULLY_CONNECTED_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX);

    // Initialize context buffer
    fully_connected_float_default_f16_ctx.buf = fully_connected_float_default_f16_buffer;
    fully_connected_float_default_f16_ctx.size = required_buffer_size;
    HELIA_GUARD_STAMP_SLACK(fully_connected_float_default_f16_buffer, fully_connected_float_default_f16_ctx.buf == fully_connected_float_default_f16_buffer ? (size_t)fully_connected_float_default_f16_ctx.size : 0u);

    return arm_fully_connected_f16(
        &fully_connected_float_default_f16_ctx, /* ctx */
        &fully_connected_float_default_f16_fc_params, /* fc_params */
        &fully_connected_float_default_f16_input_dims, /* input_dims */
        input, /* input */
        &fully_connected_float_default_f16_filter_dims, /* filter_dims */
        fully_connected_float_default_f16_weights, /* kernel */
        &fully_connected_float_default_f16_bias_dims, /* bias_dims */
        fully_connected_float_default_f16_biases, /* bias */
        &fully_connected_float_default_f16_output_dims, /* output_dims */
        output, /* output */
        ARM_NN_LAYOUT_NHWC /* layout */
    );
}


int32_t fully_connected_float_default_f16_test_case_run(void)
{
    HELIA_GUARD_ARM(fully_connected_float_default_f16_output, false /* real output, not scratch: don't poison */);
    int32_t status = fully_connected_float_default_f16_run(fully_connected_float_default_f16_input, fully_connected_float_default_f16_output);
    int failures = 0;
    HELIA_GUARD_CHECK(fully_connected_float_default_f16_buffer, "Fullyconnected scratch", failures);
    HELIA_GUARD_CHECK_SLACK(fully_connected_float_default_f16_buffer, "Fullyconnected scratch slack", fully_connected_float_default_f16_ctx.buf == fully_connected_float_default_f16_buffer ? (size_t)fully_connected_float_default_f16_ctx.size : 0u, failures);
    HELIA_GUARD_CHECK(fully_connected_float_default_f16_output, "Fullyconnected output", failures);
    HELIA_VALIDATE_STATUS("Fullyconnected", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        fully_connected_float_default_f16_output,
        fully_connected_float_default_f16_expected_output,
        FULLY_CONNECTED_FLOAT_DEFAULT_F16_OUTPUT_SIZE,
        1,
        0.001f,
        0.001f,
        20,
        failures
    );
    HELIA_VALIDATE_RETURN_FAILURES(failures);
}

int main(void)
{
    helia_test_platform_init();
    int32_t failures = fully_connected_float_default_f16_test_case_run();
    helia_test_finish(failures);
}