#include "transpose_conv_float_default_f32_transpose_conv.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"


// Context for buffer allocation
static cmsis_nn_context transpose_conv_float_default_f32_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
// Buffer size calculated conservatively to handle MVE and DSP implementations
#define TRANSPOSE_CONV_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX 1112
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[TRANSPOSE_CONV_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX];
    uint8_t tail[HELIA_GUARD_BYTES];
} transpose_conv_float_default_f32_buffer_guard;
#define transpose_conv_float_default_f32_buffer (transpose_conv_float_default_f32_buffer_guard.body)

// Reverse convolution context buffer (output_ctx parameter in arm_transpose_conv_wrapper_s8)
// Size: output width * output height * output channel * 4
#define TRANSPOSE_CONV_FLOAT_DEFAULT_F32_REVERSE_CONV_CTX_SIZE 1024
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[TRANSPOSE_CONV_FLOAT_DEFAULT_F32_REVERSE_CONV_CTX_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} transpose_conv_float_default_f32_reverse_conv_ctx_buffer_guard;
#define transpose_conv_float_default_f32_reverse_conv_ctx_buffer (transpose_conv_float_default_f32_reverse_conv_ctx_buffer_guard.body)
static cmsis_nn_context transpose_conv_float_default_f32_reverse_conv_ctx;


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
        &transpose_conv_float_default_f32_transpose_conv_params,
        &transpose_conv_float_default_f32_input_dims,
        &transpose_conv_float_default_f32_filter_dims,
        &transpose_conv_float_default_f32_output_dims
    );
    int32_t reverse_required_buffer_size = arm_transpose_conv_f32_get_reverse_conv_buffer_size(
        &transpose_conv_float_default_f32_transpose_conv_params,
        &transpose_conv_float_default_f32_input_dims,
        &transpose_conv_float_default_f32_filter_dims
    );

    if (required_buffer_size > TRANSPOSE_CONV_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX) {
        printf("Buffer size error: required=%d > max=%d\r\n", required_buffer_size, TRANSPOSE_CONV_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX);
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (reverse_required_buffer_size > TRANSPOSE_CONV_FLOAT_DEFAULT_F32_REVERSE_CONV_CTX_SIZE) {
        printf("Reverse buffer size error: required=%d > max=%d\r\n",
               reverse_required_buffer_size,
               TRANSPOSE_CONV_FLOAT_DEFAULT_F32_REVERSE_CONV_CTX_SIZE);
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Initialize context buffer
    transpose_conv_float_default_f32_ctx.buf = transpose_conv_float_default_f32_buffer;
    transpose_conv_float_default_f32_ctx.size = required_buffer_size;
    HELIA_GUARD_ARM(transpose_conv_float_default_f32_buffer, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_STAMP_SLACK(transpose_conv_float_default_f32_buffer, transpose_conv_float_default_f32_ctx.buf == transpose_conv_float_default_f32_buffer ? (size_t)transpose_conv_float_default_f32_ctx.size : 0u);

    // Initialize reverse convolution context buffer (output_ctx parameter)
    transpose_conv_float_default_f32_reverse_conv_ctx.buf = transpose_conv_float_default_f32_reverse_conv_ctx_buffer;
    transpose_conv_float_default_f32_reverse_conv_ctx.size = TRANSPOSE_CONV_FLOAT_DEFAULT_F32_REVERSE_CONV_CTX_SIZE;
    HELIA_GUARD_ARM(transpose_conv_float_default_f32_reverse_conv_ctx_buffer, true /* pure scratch: poison to catch read-before-write */);


    // Call transpose convolution kernel
    arm_cmsis_nn_status kernel_status = arm_transpose_conv_f32(
        &transpose_conv_float_default_f32_ctx,
        &transpose_conv_float_default_f32_reverse_conv_ctx,
        &transpose_conv_float_default_f32_transpose_conv_params,
        &transpose_conv_float_default_f32_input_dims,
        input,
        &transpose_conv_float_default_f32_filter_dims,
        transpose_conv_float_default_f32_weights,
        &transpose_conv_float_default_f32_bias_dims,
        transpose_conv_float_default_f32_biases,
        &transpose_conv_float_default_f32_output_dims,
        output,
        ARM_NN_LAYOUT_NHWC
    );
    
    return kernel_status;
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