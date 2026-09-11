#include "fully_connected_float_default_f32_fully_connected.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"


// Context for buffer allocation
static cmsis_nn_context fully_connected_float_default_f32_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
// Buffer size calculated conservatively to handle MVE and DSP implementations
#define FULLY_CONNECTED_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX 1024
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[FULLY_CONNECTED_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX];
    uint8_t tail[HELIA_GUARD_BYTES];
} fully_connected_float_default_f32_buffer_guard;
#define fully_connected_float_default_f32_buffer (fully_connected_float_default_f32_buffer_guard.body)


#define FULLY_CONNECTED_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 1 * 1 * 5)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[FULLY_CONNECTED_FLOAT_DEFAULT_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} fully_connected_float_default_f32_output_guard;
#define fully_connected_float_default_f32_output (fully_connected_float_default_f32_output_guard.body)

int32_t fully_connected_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
    // Calculate required buffer size. Each arm records the sizer it actually calls in
    // sizer_fn, because the checks below name it in their failure line: reporting
    // the dispatch-level name here would attribute a per-channel sizer's answer to the
    // per-tensor one, which is the misattribution #133 exists to remove. The four-bit arm
    // records nothing, since it calls no sizer at all.
    int32_t required_buffer_size = arm_fully_connected_f32_get_buffer_size(
        &fully_connected_float_default_f32_fc_params,
        &fully_connected_float_default_f32_input_dims,
        &fully_connected_float_default_f32_filter_dims,
        &fully_connected_float_default_f32_output_dims,
        ARM_NN_LAYOUT_NHWC
    );
    // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).
    HELIA_GUARD_ARM(fully_connected_float_default_f32_buffer, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_STAMP_SLACK(fully_connected_float_default_f32_buffer, 0u);
    // The slack is stamped as wholly unused here so that an early return from the
    // capacity check below leaves every canary in a checked state; it is re-stamped
    // with the real size once the context is populated (#68).


    // The sizer's answer is checked before it becomes a context size (#133). A negative
    // answer is the documented out-of-range sentinel and never a usable size; an answer
    // above this case's static means our generation-time bound and the shipped kernel
    // disagree. They are separate failures because they have separate owners.
    HELIA_VALIDATE_SIZER("arm_fully_connected_f32_get_buffer_size", required_buffer_size);
    HELIA_VALIDATE_SIZER_FITS("arm_fully_connected_f32_get_buffer_size", required_buffer_size, FULLY_CONNECTED_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX);

    // Armed unconditionally: fully_connected_float_default_f32_buffer is only ctx.buf in some branches
    // below (s4 uses NULL, weight_sum variants point ctx.buf elsewhere), but
    // poisoning/guarding it here regardless is harmless and keeps the check
    // in _test_case_run unconditional too.

    // Initialize context buffer
    fully_connected_float_default_f32_ctx.buf = fully_connected_float_default_f32_buffer;
    fully_connected_float_default_f32_ctx.size = required_buffer_size;
    HELIA_GUARD_STAMP_SLACK(fully_connected_float_default_f32_buffer, fully_connected_float_default_f32_ctx.buf == fully_connected_float_default_f32_buffer ? (size_t)fully_connected_float_default_f32_ctx.size : 0u);

    // Call fully connected kernel via wrapper (handles per-channel/per-tensor)
    arm_cmsis_nn_status kernel_status = arm_fully_connected_f32(
        &fully_connected_float_default_f32_ctx,
        &fully_connected_float_default_f32_fc_params,
        &fully_connected_float_default_f32_input_dims,
        input,
        &fully_connected_float_default_f32_filter_dims,
        fully_connected_float_default_f32_weights,
        &fully_connected_float_default_f32_bias_dims,
        fully_connected_float_default_f32_biases,
        &fully_connected_float_default_f32_output_dims,
        output,
        ARM_NN_LAYOUT_NHWC
    );
    
    return kernel_status;
}

int32_t fully_connected_float_default_f32_test_case_run(void)
{
    HELIA_GUARD_ARM(fully_connected_float_default_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = fully_connected_float_default_f32_run(fully_connected_float_default_f32_input, fully_connected_float_default_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(fully_connected_float_default_f32_buffer, "Fullyconnected scratch", failures);
    HELIA_GUARD_CHECK_SLACK(fully_connected_float_default_f32_buffer, "Fullyconnected scratch slack", fully_connected_float_default_f32_ctx.buf == fully_connected_float_default_f32_buffer ? (size_t)fully_connected_float_default_f32_ctx.size : 0u, failures);
    HELIA_GUARD_CHECK(fully_connected_float_default_f32_output, "Fullyconnected output", failures);
    HELIA_VALIDATE_STATUS("Fullyconnected", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        fully_connected_float_default_f32_output,
        fully_connected_float_default_f32_expected_output,
        FULLY_CONNECTED_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = fully_connected_float_default_f32_test_case_run();
    helia_test_finish(failures);
}