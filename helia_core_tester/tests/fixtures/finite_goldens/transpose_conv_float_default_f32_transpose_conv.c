#include "transpose_conv_float_default_f32_transpose_conv.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"



// The kernel this harness links must have the prototype the ns-cmsis-nn export records.
_Static_assert(__builtin_types_compatible_p(__typeof__(arm_transpose_conv_f32), arm_cmsis_nn_status (const cmsis_nn_context *, const cmsis_nn_context *, const cmsis_nn_transpose_conv_params_f32 *, const cmsis_nn_dims *, const float32_t *, const cmsis_nn_dims *, const float32_t *, const cmsis_nn_dims *, const float32_t *, const cmsis_nn_dims *, float32_t *, arm_nn_tensor_layout)),
               "arm_transpose_conv_f32: prototype differs from the kernel contract export; rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate");

// Context for buffer allocation
static cmsis_nn_context transpose_conv_float_default_f32_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
#define TRANSPOSE_CONV_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX 1112
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[TRANSPOSE_CONV_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX];
    uint8_t tail[HELIA_GUARD_BYTES];
} transpose_conv_float_default_f32_buffer_guard;
#define transpose_conv_float_default_f32_buffer (transpose_conv_float_default_f32_buffer_guard.body)

// Reverse convolution context (output_ctx of the kernels)
static cmsis_nn_context transpose_conv_float_default_f32_reverse_conv_ctx;
#define TRANSPOSE_CONV_FLOAT_DEFAULT_F32_REVERSE_CONV_CTX_SIZE 1024
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[TRANSPOSE_CONV_FLOAT_DEFAULT_F32_REVERSE_CONV_CTX_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} transpose_conv_float_default_f32_reverse_conv_ctx_buffer_guard;
#define transpose_conv_float_default_f32_reverse_conv_ctx_buffer (transpose_conv_float_default_f32_reverse_conv_ctx_buffer_guard.body)

#define TRANSPOSE_CONV_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 8 * 8 * 3)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[TRANSPOSE_CONV_FLOAT_DEFAULT_F32_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} transpose_conv_float_default_f32_output_guard;
#define transpose_conv_float_default_f32_output (transpose_conv_float_default_f32_output_guard.body)


int32_t transpose_conv_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
        // Calculate required buffer size
    int32_t required_buffer_size = arm_transpose_conv_f32_get_buffer_size(
        &transpose_conv_float_default_f32_transpose_conv_params, /* transpose_conv_params */
        &transpose_conv_float_default_f32_input_dims, /* input_dims */
        &transpose_conv_float_default_f32_filter_dims, /* filter_dims */
        &transpose_conv_float_default_f32_output_dims /* out_dims */
    );
    int32_t reverse_required_buffer_size = arm_transpose_conv_f32_get_reverse_conv_buffer_size(
        &transpose_conv_float_default_f32_transpose_conv_params, /* transpose_conv_params */
        &transpose_conv_float_default_f32_input_dims, /* input_dims */
        &transpose_conv_float_default_f32_filter_dims /* filter_dims */
    );
    // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).
    HELIA_GUARD_ARM(transpose_conv_float_default_f32_buffer, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_ARM(transpose_conv_float_default_f32_reverse_conv_ctx_buffer, true /* fully recomputed below: poison-safe */);
    HELIA_GUARD_STAMP_SLACK(transpose_conv_float_default_f32_buffer, 0u);

    // The sizer's answer is checked before it becomes a context size (#133): a negative
    // answer is the documented out-of-range sentinel, and one above this case's static bound
    // means the generation-time bound and the shipped kernel disagree.
    HELIA_VALIDATE_SIZER("arm_transpose_conv_f32_get_buffer_size", required_buffer_size);
    HELIA_VALIDATE_SIZER_FITS("arm_transpose_conv_f32_get_buffer_size", required_buffer_size, TRANSPOSE_CONV_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX);
    HELIA_VALIDATE_SIZER("arm_transpose_conv_f32_get_reverse_conv_buffer_size", reverse_required_buffer_size);
    HELIA_VALIDATE_SIZER_FITS("arm_transpose_conv_f32_get_reverse_conv_buffer_size", reverse_required_buffer_size, TRANSPOSE_CONV_FLOAT_DEFAULT_F32_REVERSE_CONV_CTX_SIZE);

    // Initialize context buffer
    transpose_conv_float_default_f32_ctx.buf = transpose_conv_float_default_f32_buffer;
    transpose_conv_float_default_f32_ctx.size = required_buffer_size;
    HELIA_GUARD_STAMP_SLACK(transpose_conv_float_default_f32_buffer, transpose_conv_float_default_f32_ctx.buf == transpose_conv_float_default_f32_buffer ? (size_t)transpose_conv_float_default_f32_ctx.size : 0u);

    // Initialize reverse convolution context buffer (output_ctx parameter)
    transpose_conv_float_default_f32_reverse_conv_ctx.buf = transpose_conv_float_default_f32_reverse_conv_ctx_buffer;
    transpose_conv_float_default_f32_reverse_conv_ctx.size = TRANSPOSE_CONV_FLOAT_DEFAULT_F32_REVERSE_CONV_CTX_SIZE;

    return arm_transpose_conv_f32(
        &transpose_conv_float_default_f32_ctx, /* ctx */
        &transpose_conv_float_default_f32_reverse_conv_ctx, /* output_ctx */
        &transpose_conv_float_default_f32_transpose_conv_params, /* transpose_conv_params */
        &transpose_conv_float_default_f32_input_dims, /* input_dims */
        input, /* input_data */
        &transpose_conv_float_default_f32_filter_dims, /* filter_dims */
        transpose_conv_float_default_f32_weights, /* filter_data */
        &transpose_conv_float_default_f32_bias_dims, /* bias_dims */
        transpose_conv_float_default_f32_biases, /* bias_data */
        &transpose_conv_float_default_f32_output_dims, /* output_dims */
        output, /* output_data */
        ARM_NN_LAYOUT_NHWC /* layout */
    );
}


int32_t transpose_conv_float_default_f32_test_case_run(void)
{
    HELIA_GUARD_ARM(transpose_conv_float_default_f32_output, false /* real output, not scratch: don't poison */);
    int32_t status = transpose_conv_float_default_f32_run(transpose_conv_float_default_f32_input, transpose_conv_float_default_f32_output);
    int failures = 0;
    HELIA_GUARD_CHECK(transpose_conv_float_default_f32_buffer, "TransposeConv scratch", failures);
    HELIA_GUARD_CHECK_SLACK(transpose_conv_float_default_f32_buffer, "TransposeConv scratch slack", transpose_conv_float_default_f32_ctx.buf == transpose_conv_float_default_f32_buffer ? (size_t)transpose_conv_float_default_f32_ctx.size : 0u, failures);
    HELIA_GUARD_CHECK(transpose_conv_float_default_f32_reverse_conv_ctx_buffer, "TransposeConv reverse_conv_ctx", failures);
    HELIA_GUARD_CHECK(transpose_conv_float_default_f32_output, "TransposeConv output", failures);
    HELIA_VALIDATE_STATUS("TransposeConv", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        transpose_conv_float_default_f32_output,
        transpose_conv_float_default_f32_expected_output,
        TRANSPOSE_CONV_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
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
    int32_t failures = transpose_conv_float_default_f32_test_case_run();
    helia_test_finish(failures);
}