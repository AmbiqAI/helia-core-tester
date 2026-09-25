#include "transpose_conv_float_default_f16_transpose_conv.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"


// Context for buffer allocation
static cmsis_nn_context transpose_conv_float_default_f16_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
// Buffer size calculated conservatively to handle MVE and DSP implementations
#define TRANSPOSE_CONV_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX 1024
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[TRANSPOSE_CONV_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX];
    uint8_t tail[HELIA_GUARD_BYTES];
} transpose_conv_float_default_f16_buffer_guard;
#define transpose_conv_float_default_f16_buffer (transpose_conv_float_default_f16_buffer_guard.body)

// Reverse convolution context buffer (output_ctx parameter in arm_transpose_conv_wrapper_s8)
// Size: output width * output height * output channel * 4
#define TRANSPOSE_CONV_FLOAT_DEFAULT_F16_REVERSE_CONV_CTX_SIZE 1024
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[TRANSPOSE_CONV_FLOAT_DEFAULT_F16_REVERSE_CONV_CTX_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} transpose_conv_float_default_f16_reverse_conv_ctx_buffer_guard;
#define transpose_conv_float_default_f16_reverse_conv_ctx_buffer (transpose_conv_float_default_f16_reverse_conv_ctx_buffer_guard.body)
static cmsis_nn_context transpose_conv_float_default_f16_reverse_conv_ctx;


#define TRANSPOSE_CONV_FLOAT_DEFAULT_F16_OUTPUT_SIZE (1 * 8 * 8 * 3)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float16_t body[TRANSPOSE_CONV_FLOAT_DEFAULT_F16_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} transpose_conv_float_default_f16_output_guard;
#define transpose_conv_float_default_f16_output (transpose_conv_float_default_f16_output_guard.body)

int32_t transpose_conv_float_default_f16_run(
    const float16_t* __restrict input,
    float16_t* __restrict output
) {
    // Calculate required buffer size
    int32_t required_buffer_size = arm_transpose_conv_f16_get_buffer_size(
        &transpose_conv_float_default_f16_transpose_conv_params,
        &transpose_conv_float_default_f16_input_dims,
        &transpose_conv_float_default_f16_filter_dims,
        &transpose_conv_float_default_f16_output_dims
    );
    int32_t reverse_required_buffer_size = arm_transpose_conv_f16_get_reverse_conv_buffer_size(
        &transpose_conv_float_default_f16_transpose_conv_params,
        &transpose_conv_float_default_f16_input_dims,
        &transpose_conv_float_default_f16_filter_dims
    );
    // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).
    HELIA_GUARD_ARM(transpose_conv_float_default_f16_buffer, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_ARM(transpose_conv_float_default_f16_reverse_conv_ctx_buffer, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_STAMP_SLACK(transpose_conv_float_default_f16_buffer, 0u);
    // The slack is stamped as wholly unused here so that an early return from the
    // capacity check below leaves every canary in a checked state; it is re-stamped
    // with the real size once the context is populated (#68).


    // The sizer's answer is checked before it becomes a context size (#133). A negative
    // answer is the documented out-of-range sentinel and never a usable size; an answer
    // above this case's static means our generation-time bound and the shipped kernel
    // disagree. They are separate failures because they have separate owners.
    HELIA_VALIDATE_SIZER("arm_transpose_conv_f16_get_buffer_size", required_buffer_size);
    HELIA_VALIDATE_SIZER_FITS("arm_transpose_conv_f16_get_buffer_size", required_buffer_size, TRANSPOSE_CONV_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX);
    // The reverse sizer's answer is discarded in favour of a compile-time size, so a negative
    // return here was previously invisible (#133).
    HELIA_VALIDATE_SIZER("arm_transpose_conv_f16_get_reverse_conv_buffer_size", reverse_required_buffer_size);
    HELIA_VALIDATE_SIZER_FITS("arm_transpose_conv_f16_get_reverse_conv_buffer_size", reverse_required_buffer_size, TRANSPOSE_CONV_FLOAT_DEFAULT_F16_REVERSE_CONV_CTX_SIZE);

    // Initialize context buffer
    transpose_conv_float_default_f16_ctx.buf = transpose_conv_float_default_f16_buffer;
    transpose_conv_float_default_f16_ctx.size = required_buffer_size;
    HELIA_GUARD_STAMP_SLACK(transpose_conv_float_default_f16_buffer, transpose_conv_float_default_f16_ctx.buf == transpose_conv_float_default_f16_buffer ? (size_t)transpose_conv_float_default_f16_ctx.size : 0u);

    // Initialize reverse convolution context buffer (output_ctx parameter)
    transpose_conv_float_default_f16_reverse_conv_ctx.buf = transpose_conv_float_default_f16_reverse_conv_ctx_buffer;
    transpose_conv_float_default_f16_reverse_conv_ctx.size = TRANSPOSE_CONV_FLOAT_DEFAULT_F16_REVERSE_CONV_CTX_SIZE;


    // Call transpose convolution kernel
    arm_cmsis_nn_status kernel_status = arm_transpose_conv_f16(
        &transpose_conv_float_default_f16_ctx,
        &transpose_conv_float_default_f16_reverse_conv_ctx,
        &transpose_conv_float_default_f16_transpose_conv_params,
        &transpose_conv_float_default_f16_input_dims,
        input,
        &transpose_conv_float_default_f16_filter_dims,
        transpose_conv_float_default_f16_weights,
        &transpose_conv_float_default_f16_bias_dims,
        transpose_conv_float_default_f16_biases,
        &transpose_conv_float_default_f16_output_dims,
        output,
        ARM_NN_LAYOUT_NHWC
    );
    
    return kernel_status;
}

int32_t transpose_conv_float_default_f16_test_case_run(void)
{
    HELIA_GUARD_ARM(transpose_conv_float_default_f16_output, false /* real output, not scratch: don't poison */);
    int32_t status = transpose_conv_float_default_f16_run(transpose_conv_float_default_f16_input, transpose_conv_float_default_f16_output);
    int failures = 0;
    HELIA_GUARD_CHECK(transpose_conv_float_default_f16_buffer, "TransposeConv scratch", failures);
    HELIA_GUARD_CHECK_SLACK(transpose_conv_float_default_f16_buffer, "TransposeConv scratch slack", transpose_conv_float_default_f16_ctx.buf == transpose_conv_float_default_f16_buffer ? (size_t)transpose_conv_float_default_f16_ctx.size : 0u, failures);
    HELIA_GUARD_CHECK(transpose_conv_float_default_f16_reverse_conv_ctx_buffer, "TransposeConv reverse_conv_ctx", failures);
    HELIA_GUARD_CHECK(transpose_conv_float_default_f16_output, "TransposeConv output", failures);
    HELIA_VALIDATE_STATUS("TransposeConv", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        transpose_conv_float_default_f16_output,
        transpose_conv_float_default_f16_expected_output,
        TRANSPOSE_CONV_FLOAT_DEFAULT_F16_OUTPUT_SIZE,
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
    int32_t failures = transpose_conv_float_default_f16_test_case_run();
    helia_test_finish(failures);
}