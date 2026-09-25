#include "lstm_unidirectional_float_stream_f32_lstm_unidirectional.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"


static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[16];
    uint8_t tail[HELIA_GUARD_BYTES];
} lstm_unidirectional_float_stream_f32_output_guard;
#define lstm_unidirectional_float_stream_f32_output (lstm_unidirectional_float_stream_f32_output_guard.body)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[4];
    uint8_t tail[HELIA_GUARD_BYTES];
} lstm_unidirectional_float_stream_f32_temp1_guard;
#define lstm_unidirectional_float_stream_f32_temp1 (lstm_unidirectional_float_stream_f32_temp1_guard.body)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[4];
    uint8_t tail[HELIA_GUARD_BYTES];
} lstm_unidirectional_float_stream_f32_temp2_guard;
#define lstm_unidirectional_float_stream_f32_temp2 (lstm_unidirectional_float_stream_f32_temp2_guard.body)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[4];
    uint8_t tail[HELIA_GUARD_BYTES];
} lstm_unidirectional_float_stream_f32_cell_state_guard;
#define lstm_unidirectional_float_stream_f32_cell_state (lstm_unidirectional_float_stream_f32_cell_state_guard.body)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float body[4];
    uint8_t tail[HELIA_GUARD_BYTES];
} lstm_unidirectional_float_stream_f32_hidden_state_guard;
#define lstm_unidirectional_float_stream_f32_hidden_state (lstm_unidirectional_float_stream_f32_hidden_state_guard.body)

/*
 * Streaming/stateful test case: replays the full sequence across
 * 2 chunked calls, carrying hidden_state
 * between calls (batch_size == 1 only), and validates the concatenated
 * chunked output against the single-shot golden reference.
 *
 * Unlike GRU, LSTM's cell_state is also caller-owned state, not just
 * hidden_state: per arm_lstm_unidirectional_f32.c, the kernel only zeroes
 * cell_state itself when hidden_state == NULL (single-shot mode). In
 * streaming mode (hidden_state != NULL throughout, as here) cell_state is
 * treated as in/out and must be seeded to zero once by the caller and then
 * left alone across chunk boundaries -- it is NOT re-zeroed per chunk.
 */
static int32_t run_lstm_chunk(int32_t chunk_time_steps, int32_t input_offset, int32_t output_offset)
{
    HELIA_GUARD_ARM(lstm_unidirectional_float_stream_f32_temp1, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_ARM(lstm_unidirectional_float_stream_f32_temp2, true /* pure scratch: poison to catch read-before-write */);

    const cmsis_nn_lstm_params_f32 params = {
        .time_major = 0,
        .batch_size = 1,
        .time_steps = chunk_time_steps,
        .input_size = 3,
        .hidden_size = 4,
        .cell_clip = 0.0f,
        .forget_gate = {
            .input_weights = lstm_unidirectional_float_stream_f32_forget_gate_input_weights,
            .hidden_weights = lstm_unidirectional_float_stream_f32_forget_gate_hidden_weights,
            .bias = lstm_unidirectional_float_stream_f32_forget_gate_bias,
            .activation_type = ARM_NN_FLT_ACT_SIGMOID,
        },
        .input_gate = {
            .input_weights = lstm_unidirectional_float_stream_f32_input_gate_input_weights,
            .hidden_weights = lstm_unidirectional_float_stream_f32_input_gate_hidden_weights,
            .bias = lstm_unidirectional_float_stream_f32_input_gate_bias,
            .activation_type = ARM_NN_FLT_ACT_SIGMOID,
        },
        .cell_gate = {
            .input_weights = lstm_unidirectional_float_stream_f32_cell_gate_input_weights,
            .hidden_weights = lstm_unidirectional_float_stream_f32_cell_gate_hidden_weights,
            .bias = lstm_unidirectional_float_stream_f32_cell_gate_bias,
            .activation_type = ARM_NN_FLT_ACT_TANH,
        },
        .output_gate = {
            .input_weights = lstm_unidirectional_float_stream_f32_output_gate_input_weights,
            .hidden_weights = lstm_unidirectional_float_stream_f32_output_gate_hidden_weights,
            .bias = lstm_unidirectional_float_stream_f32_output_gate_bias,
            .activation_type = ARM_NN_FLT_ACT_SIGMOID,
        },
    };
    cmsis_nn_lstm_context_f32 buffers = {
        .temp1 = lstm_unidirectional_float_stream_f32_temp1,
        .temp2 = lstm_unidirectional_float_stream_f32_temp2,
        .cell_state = lstm_unidirectional_float_stream_f32_cell_state,
        .hidden_state = lstm_unidirectional_float_stream_f32_hidden_state,
    };

    return arm_lstm_unidirectional_f32(
        lstm_unidirectional_float_stream_f32_input_tensor + input_offset,
        lstm_unidirectional_float_stream_f32_output + output_offset,
        &params,
        &buffers);
}

int32_t lstm_unidirectional_float_stream_f32_test_case_run(void)
{
    /* Fresh streaming state for this sequence: both hidden_state and
     * cell_state start at zero, matching the kernel's own single-shot
     * zero-init (see comment above) -- but seeded once here since the
     * kernel will not zero cell_state itself once hidden_state is non-NULL. */
    for (int32_t h = 0; h < 4; h++)
    {
        lstm_unidirectional_float_stream_f32_hidden_state[h] = (float)0;
        lstm_unidirectional_float_stream_f32_cell_state[h] = (float)0;
    }

    int32_t status;
    int failures = 0;
    /* Canaries only, no poison: output, cell_state and hidden_state carry
     * real data across chunks. Re-stamped per chunk so a breach is attributed
     * to the chunk that caused it and reported once. */
    HELIA_GUARD_ARM(lstm_unidirectional_float_stream_f32_output, false);
    HELIA_GUARD_ARM(lstm_unidirectional_float_stream_f32_cell_state, false);
    HELIA_GUARD_ARM(lstm_unidirectional_float_stream_f32_hidden_state, false);
    status = run_lstm_chunk(2, 0, 0);
    // Checked before the status validator can return, and before the next
    // chunk's run_lstm_chunk() re-arms (re-poisons) temp1/temp2 and erases a
    // breach.
    HELIA_GUARD_CHECK(lstm_unidirectional_float_stream_f32_temp1, "Lstmunidirectional temp1 (chunk 0)", failures);
    HELIA_GUARD_CHECK(lstm_unidirectional_float_stream_f32_temp2, "Lstmunidirectional temp2 (chunk 0)", failures);
    HELIA_GUARD_CHECK(lstm_unidirectional_float_stream_f32_output, "Lstmunidirectional output (chunk 0)", failures);
    HELIA_GUARD_CHECK(lstm_unidirectional_float_stream_f32_cell_state, "Lstmunidirectional cell_state (chunk 0)", failures);
    HELIA_GUARD_CHECK(lstm_unidirectional_float_stream_f32_hidden_state, "Lstmunidirectional hidden_state (chunk 0)", failures);
    HELIA_VALIDATE_STATUS("Lstmunidirectional (chunk 0)", status);
    /* Canaries only, no poison: output, cell_state and hidden_state carry
     * real data across chunks. Re-stamped per chunk so a breach is attributed
     * to the chunk that caused it and reported once. */
    HELIA_GUARD_ARM(lstm_unidirectional_float_stream_f32_output, false);
    HELIA_GUARD_ARM(lstm_unidirectional_float_stream_f32_cell_state, false);
    HELIA_GUARD_ARM(lstm_unidirectional_float_stream_f32_hidden_state, false);
    status = run_lstm_chunk(2, 6, 8);
    // Checked before the status validator can return, and before the next
    // chunk's run_lstm_chunk() re-arms (re-poisons) temp1/temp2 and erases a
    // breach.
    HELIA_GUARD_CHECK(lstm_unidirectional_float_stream_f32_temp1, "Lstmunidirectional temp1 (chunk 1)", failures);
    HELIA_GUARD_CHECK(lstm_unidirectional_float_stream_f32_temp2, "Lstmunidirectional temp2 (chunk 1)", failures);
    HELIA_GUARD_CHECK(lstm_unidirectional_float_stream_f32_output, "Lstmunidirectional output (chunk 1)", failures);
    HELIA_GUARD_CHECK(lstm_unidirectional_float_stream_f32_cell_state, "Lstmunidirectional cell_state (chunk 1)", failures);
    HELIA_GUARD_CHECK(lstm_unidirectional_float_stream_f32_hidden_state, "Lstmunidirectional hidden_state (chunk 1)", failures);
    HELIA_VALIDATE_STATUS("Lstmunidirectional (chunk 1)", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        lstm_unidirectional_float_stream_f32_output,
        lstm_unidirectional_float_stream_f32_expected_output,
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
    int32_t failures = lstm_unidirectional_float_stream_f32_test_case_run();
    helia_test_finish(failures);
}