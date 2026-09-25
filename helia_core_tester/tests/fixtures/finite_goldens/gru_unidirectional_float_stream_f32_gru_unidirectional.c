#include "gru_unidirectional_float_stream_f32_gru_unidirectional.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"


static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float32_t body[16];
    uint8_t tail[HELIA_GUARD_BYTES];
} gru_unidirectional_float_stream_f32_output_guard;
#define gru_unidirectional_float_stream_f32_output (gru_unidirectional_float_stream_f32_output_guard.body)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float32_t body[4];
    uint8_t tail[HELIA_GUARD_BYTES];
} gru_unidirectional_float_stream_f32_temp1_guard;
#define gru_unidirectional_float_stream_f32_temp1 (gru_unidirectional_float_stream_f32_temp1_guard.body)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float32_t body[4];
    uint8_t tail[HELIA_GUARD_BYTES];
} gru_unidirectional_float_stream_f32_hidden_state_guard;
#define gru_unidirectional_float_stream_f32_hidden_state (gru_unidirectional_float_stream_f32_hidden_state_guard.body)

/*
 * Streaming/stateful test case: replays the full sequence across
 * 2 chunked calls, carrying hidden_state
 * between calls (batch_size == 1 only), and validates the concatenated
 * chunked output against the single-shot golden reference.
 */
static int32_t run_gru_chunk(int32_t chunk_time_steps, int32_t input_offset, int32_t output_offset)
{
    HELIA_GUARD_ARM(gru_unidirectional_float_stream_f32_temp1, true /* pure scratch: poison to catch read-before-write */);

    const cmsis_nn_gru_params_f32 params = {
        .time_major = 0,
        .batch_size = 1,
        .time_steps = chunk_time_steps,
        .input_size = 3,
        .hidden_size = 4,
        .reset_after = 1,
        .update_gate = {
            .input_weights = gru_unidirectional_float_stream_f32_update_gate_input_weights,
            .hidden_weights = gru_unidirectional_float_stream_f32_update_gate_hidden_weights,
            .input_bias = gru_unidirectional_float_stream_f32_update_gate_input_bias,
            .hidden_bias = gru_unidirectional_float_stream_f32_update_gate_hidden_bias,
        },
        .reset_gate = {
            .input_weights = gru_unidirectional_float_stream_f32_reset_gate_input_weights,
            .hidden_weights = gru_unidirectional_float_stream_f32_reset_gate_hidden_weights,
            .input_bias = gru_unidirectional_float_stream_f32_reset_gate_input_bias,
            .hidden_bias = gru_unidirectional_float_stream_f32_reset_gate_hidden_bias,
        },
        .candidate_gate = {
            .input_weights = gru_unidirectional_float_stream_f32_candidate_gate_input_weights,
            .hidden_weights = gru_unidirectional_float_stream_f32_candidate_gate_hidden_weights,
            .input_bias = gru_unidirectional_float_stream_f32_candidate_gate_input_bias,
            .hidden_bias = gru_unidirectional_float_stream_f32_candidate_gate_hidden_bias,
        },
    };
    cmsis_nn_gru_context_f32 buffers = {
        .temp1 = gru_unidirectional_float_stream_f32_temp1,
        .hidden_state = gru_unidirectional_float_stream_f32_hidden_state,
    };

    return arm_gru_unidirectional_f32(
        gru_unidirectional_float_stream_f32_input_tensor + input_offset,
        gru_unidirectional_float_stream_f32_output + output_offset,
        &params,
        &buffers);
}

int32_t gru_unidirectional_float_stream_f32_test_case_run(void)
{
    /* Fresh streaming state for this sequence. */
    for (int32_t h = 0; h < 4; h++)
    {
        gru_unidirectional_float_stream_f32_hidden_state[h] = (float32_t)0;
    }

    int32_t status;
    int failures = 0;
    /* Canaries only, no poison: output and hidden_state carry real data
     * across chunks. Re-stamped per chunk so a breach is attributed to the
     * chunk that caused it and reported once. */
    HELIA_GUARD_ARM(gru_unidirectional_float_stream_f32_output, false);
    HELIA_GUARD_ARM(gru_unidirectional_float_stream_f32_hidden_state, false);
    status = run_gru_chunk(2, 0, 0);
    // Checked before the status validator can return, and before the next
    // chunk's run_gru_chunk() re-arms (re-poisons) temp1 and erases a breach.
    HELIA_GUARD_CHECK(gru_unidirectional_float_stream_f32_temp1, "Gruunidirectional temp1 (chunk 0)", failures);
    HELIA_GUARD_CHECK(gru_unidirectional_float_stream_f32_output, "Gruunidirectional output (chunk 0)", failures);
    HELIA_GUARD_CHECK(gru_unidirectional_float_stream_f32_hidden_state, "Gruunidirectional hidden_state (chunk 0)", failures);
    HELIA_VALIDATE_STATUS("Gruunidirectional (chunk 0)", status);
    /* Canaries only, no poison: output and hidden_state carry real data
     * across chunks. Re-stamped per chunk so a breach is attributed to the
     * chunk that caused it and reported once. */
    HELIA_GUARD_ARM(gru_unidirectional_float_stream_f32_output, false);
    HELIA_GUARD_ARM(gru_unidirectional_float_stream_f32_hidden_state, false);
    status = run_gru_chunk(2, 6, 8);
    // Checked before the status validator can return, and before the next
    // chunk's run_gru_chunk() re-arms (re-poisons) temp1 and erases a breach.
    HELIA_GUARD_CHECK(gru_unidirectional_float_stream_f32_temp1, "Gruunidirectional temp1 (chunk 1)", failures);
    HELIA_GUARD_CHECK(gru_unidirectional_float_stream_f32_output, "Gruunidirectional output (chunk 1)", failures);
    HELIA_GUARD_CHECK(gru_unidirectional_float_stream_f32_hidden_state, "Gruunidirectional hidden_state (chunk 1)", failures);
    HELIA_VALIDATE_STATUS("Gruunidirectional (chunk 1)", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        gru_unidirectional_float_stream_f32_output,
        gru_unidirectional_float_stream_f32_expected_output,
        16,
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
    int32_t failures = gru_unidirectional_float_stream_f32_test_case_run();
    helia_test_finish(failures);
}