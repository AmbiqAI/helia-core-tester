#include "split_float_channels_pairs_f16_split.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"



// The kernel this harness links must have the prototype the ns-cmsis-nn export records.
_Static_assert(__builtin_types_compatible_p(__typeof__(arm_split_f16), arm_cmsis_nn_status (const float16_t *, const int32_t, const int32_t *, const int32_t, const int32_t, const int32_t *, float16_t * const *)),
               "arm_split_f16: prototype differs from the kernel contract export; rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate");



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
        // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).

    // The sizer's answer is checked before it becomes a context size (#133): a negative
    // answer is the documented out-of-range sentinel, and one above this case's static bound
    // means the generation-time bound and the shipped kernel disagree.


    return arm_split_f16(
        input, /* input_data */
        4, /* input_dims */
        split_float_channels_pairs_f16_input_shape, /* input_shape */
        3, /* axis */
        2, /* num_splits */
        split_float_channels_pairs_f16_split_dims, /* split_dims */
        split_float_channels_pairs_f16_output_ptrs /* output_data */
    );
}


int32_t split_float_channels_pairs_f16_test_case_run(void)
{
    HELIA_GUARD_ARM(split_float_channels_pairs_f16_out_0_output, false /* real output, not scratch: don't poison */);
    HELIA_GUARD_ARM(split_float_channels_pairs_f16_out_1_output, false /* real output, not scratch: don't poison */);
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