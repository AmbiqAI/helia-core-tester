#include "split_float_channels_pairs_f16_split.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"


#define SPLIT_FLOAT_CHANNELS_PAIRS_F16_OUT_0_OUTPUT_SIZE (8)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float16_t body[SPLIT_FLOAT_CHANNELS_PAIRS_F16_OUT_0_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} split_float_channels_pairs_f16_out_0_output_guard;
#define split_float_channels_pairs_f16_out_0_output (split_float_channels_pairs_f16_out_0_output_guard.body)
#define SPLIT_FLOAT_CHANNELS_PAIRS_F16_OUT_1_OUTPUT_SIZE (8)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float16_t body[SPLIT_FLOAT_CHANNELS_PAIRS_F16_OUT_1_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} split_float_channels_pairs_f16_out_1_output_guard;
#define split_float_channels_pairs_f16_out_1_output (split_float_channels_pairs_f16_out_1_output_guard.body)

static float16_t* split_float_channels_pairs_f16_output_ptrs[] = {
    split_float_channels_pairs_f16_out_0_output,
    split_float_channels_pairs_f16_out_1_output,
};

int32_t split_float_channels_pairs_f16_run(
    const float16_t* __restrict input
) {
    HELIA_GUARD_ARM(split_float_channels_pairs_f16_out_0_output, false /* real output, not scratch: don't poison */);
    HELIA_GUARD_ARM(split_float_channels_pairs_f16_out_1_output, false /* real output, not scratch: don't poison */);
    // Call Split kernel
    // Note: Split has multiple outputs, but we only handle the first one for now
    arm_cmsis_nn_status kernel_status = arm_split_f16(
        input,                              // input_data
        4,             // input_dims
        split_float_channels_pairs_f16_input_shape,             // input_shape
        3,                         // axis
        2,                   // num_splits
        split_float_channels_pairs_f16_split_dims,              // split_dims
        split_float_channels_pairs_f16_output_ptrs              // output_data (array of pointers)
    );
    
    return kernel_status;
}

int32_t split_float_channels_pairs_f16_test_case_run(void)
{
    int32_t status = split_float_channels_pairs_f16_run(split_float_channels_pairs_f16_input);
    int failures = 0;
    HELIA_GUARD_CHECK(split_float_channels_pairs_f16_out_0_output, "Split split_float_channels_pairs_f16_out_0 output", failures);
    HELIA_GUARD_CHECK(split_float_channels_pairs_f16_out_1_output, "Split split_float_channels_pairs_f16_out_1 output", failures);
    HELIA_VALIDATE_STATUS("Split", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        split_float_channels_pairs_f16_out_0_output,
        split_float_channels_pairs_f16_out_0_expected_output,
        SPLIT_FLOAT_CHANNELS_PAIRS_F16_OUT_0_OUTPUT_SIZE,
        1,
        0.001f,
        0.001f,
        20,
        failures
    );
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        split_float_channels_pairs_f16_out_1_output,
        split_float_channels_pairs_f16_out_1_expected_output,
        SPLIT_FLOAT_CHANNELS_PAIRS_F16_OUT_1_OUTPUT_SIZE,
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
    int32_t failures = split_float_channels_pairs_f16_test_case_run();
    helia_test_finish(failures);
}