/* SVDF and LSTM adapter size checks. */
#include <stdio.h>

#include "arm_nnfunctions.h"

/* Stub kernels count calls. */
static int kernel_calls;

arm_cmsis_nn_status arm_svdf_s8(const cmsis_nn_context *ctx, const cmsis_nn_context *input_ctx,
                                const cmsis_nn_context *output_ctx, const cmsis_nn_svdf_params *params,
                                const cmsis_nn_per_tensor_quant_params *input_quant,
                                const cmsis_nn_per_tensor_quant_params *output_quant,
                                const cmsis_nn_dims *input_dims, const int8_t *input,
                                const cmsis_nn_dims *state_dims, int8_t *state,
                                const cmsis_nn_dims *feature_dims, const int8_t *feature,
                                const cmsis_nn_dims *time_dims, const int8_t *time,
                                const cmsis_nn_dims *bias_dims, const int32_t *bias,
                                const cmsis_nn_dims *output_dims, int8_t *output)
{
    (void)ctx; (void)input_ctx; (void)output_ctx; (void)params; (void)input_quant; (void)output_quant;
    (void)input_dims; (void)input; (void)state_dims; (void)state; (void)feature_dims; (void)feature;
    (void)time_dims; (void)time; (void)bias_dims; (void)bias; (void)output_dims; (void)output;
    ++kernel_calls;
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_svdf_state_s16_s8(const cmsis_nn_context *input_ctx, const cmsis_nn_context *output_ctx,
                                          const cmsis_nn_svdf_params *params,
                                          const cmsis_nn_per_tensor_quant_params *input_quant,
                                          const cmsis_nn_per_tensor_quant_params *output_quant,
                                          const cmsis_nn_dims *input_dims, const int8_t *input,
                                          const cmsis_nn_dims *state_dims, int16_t *state,
                                          const cmsis_nn_dims *feature_dims, const int8_t *feature,
                                          const cmsis_nn_dims *time_dims, const int16_t *time,
                                          const cmsis_nn_dims *bias_dims, const int32_t *bias,
                                          const cmsis_nn_dims *output_dims, int8_t *output)
{
    (void)input_ctx; (void)output_ctx; (void)params; (void)input_quant; (void)output_quant;
    (void)input_dims; (void)input; (void)state_dims; (void)state; (void)feature_dims; (void)feature;
    (void)time_dims; (void)time; (void)bias_dims; (void)bias; (void)output_dims; (void)output;
    ++kernel_calls;
    return ARM_CMSIS_NN_SUCCESS;
}

int32_t arm_svdf_s8_get_buffer_size(const cmsis_nn_dims *dims) { (void)dims; return 16; }
int32_t arm_svdf_s8_input_ctx_get_buffer_size(const cmsis_nn_dims *a, const cmsis_nn_dims *b) { (void)a; (void)b; return 16; }
int32_t arm_svdf_s8_output_ctx_get_buffer_size(const cmsis_nn_svdf_params *p, const cmsis_nn_dims *a, const cmsis_nn_dims *b)
{
    (void)p; (void)a; (void)b;
    return 16;
}
int32_t arm_svdf_state_s16_s8_input_ctx_get_buffer_size(const cmsis_nn_dims *a, const cmsis_nn_dims *b) { (void)a; (void)b; return 16; }
int32_t arm_svdf_state_s16_s8_output_ctx_get_buffer_size(const cmsis_nn_svdf_params *p, const cmsis_nn_dims *a, const cmsis_nn_dims *b)
{
    (void)p; (void)a; (void)b;
    return 16;
}

arm_cmsis_nn_status arm_vector_sum_s8(int32_t *sum, const int32_t cols, const int32_t rows, const int8_t *data,
                                      const int32_t lhs_offset, const int32_t rhs_offset, const int32_t *bias)
{
    (void)sum; (void)cols; (void)rows; (void)data; (void)lhs_offset; (void)rhs_offset; (void)bias;
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_lstm_unidirectional_s8(const int8_t *input, int8_t *output, const cmsis_nn_lstm_params *params,
                                               cmsis_nn_lstm_context *buffers)
{
    (void)input; (void)output; (void)params; (void)buffers;
    ++kernel_calls;
    return ARM_CMSIS_NN_SUCCESS;
}

int32_t arm_lstm_unidirectional_s8_temp1_get_buffer_size(const cmsis_nn_lstm_params *params)
{
    return params->batch_size * params->hidden_size * 2;
}

int32_t arm_lstm_unidirectional_s8_temp2_get_buffer_size(const cmsis_nn_lstm_params *params)
{
    return params->batch_size * params->hidden_size * 2;
}

#include "benchmark_server_adapters.gen.c"

hct_window_t hct_window;

static _Alignas(16) uint8_t workspace[4096];
static int8_t operand[1024];
static int32_t meta[31];

static hct_server_blob_t *add_blob(hct_server_session_t *session, uint8_t role, uint8_t dtype,
                                   uint32_t n, uint32_t h, uint32_t w, uint32_t bytes, const void *data)
{
    hct_server_blob_t *blob = &session->blobs[session->blob_count++];
    blob->role = role;
    blob->dtype = dtype;
    blob->rank = 4u;
    blob->dimensions[0] = n;
    blob->dimensions[1] = h;
    blob->dimensions[2] = w;
    blob->dimensions[3] = 1u;
    blob->byte_length = bytes;
    blob->placed = (const uint8_t *)data;
    return blob;
}

static void reset(hct_server_session_t *session, uint32_t kernel_id)
{
    memset(session, 0, sizeof(*session));
    session->workspace = workspace;
    session->workspace_bytes = sizeof(workspace);
    session->scratch_offset = 0u;
    session->scratch_bytes = 1024u;
    session->output_workspace_offset = 2048u;
    session->output_capacity_bytes = 1024u;
    session->expected_kernel_id = kernel_id;
    kernel_calls = 0;
}

/* 2 batches, 4 inputs, 4 filters, rank 2. */
static arm_cmsis_nn_status run_svdf(uint32_t time_n, uint32_t time_h, uint32_t time_bytes)
{
    hct_server_session_t session;
    reset(&session, HCT_KERNEL_ID_SVDF_S8);
    memset(meta, 0, sizeof(meta));
    meta[0] = 2;
    add_blob(&session, HCT_BLOB_ROLE_INPUT_0, HCT_DTYPE_S8, 2u, 4u, 1u, 8u, operand);
    add_blob(&session, HCT_BLOB_ROLE_INPUT_1, HCT_DTYPE_S8, 2u, 4u, time_h, 2u * 4u * time_h, operand);
    add_blob(&session, HCT_BLOB_ROLE_INPUT_2, HCT_DTYPE_S8, time_n, time_h, 1u, time_bytes, operand);
    add_blob(&session, HCT_BLOB_ROLE_WEIGHTS, HCT_DTYPE_S8, 4u, 4u, 1u, 16u, operand);
    add_blob(&session, HCT_BLOB_ROLE_META_0, HCT_DTYPE_S32, 7u, 1u, 1u, 7u * 4u, meta);
    return run_svdf_once(&session);
}

/* Input blob holds four bytes. */
static arm_cmsis_nn_status run_lstm(int32_t time_steps, uint32_t input_bytes)
{
    hct_server_session_t session;
    reset(&session, HCT_KERNEL_ID_LSTM_UNIDIRECTIONAL_S8);
    memset(meta, 0, sizeof(meta));
    meta[1] = 1;
    meta[2] = time_steps;
    meta[3] = 4;
    meta[4] = 4;
    add_blob(&session, HCT_BLOB_ROLE_INPUT_0, HCT_DTYPE_S8, input_bytes, 1u, 1u, input_bytes, operand);
    add_blob(&session, HCT_BLOB_ROLE_WEIGHTS, HCT_DTYPE_S8, 128u, 1u, 1u, 128u, operand);
    add_blob(&session, HCT_BLOB_ROLE_BIAS, HCT_DTYPE_S32, 16u, 1u, 1u, 64u, operand);
    add_blob(&session, HCT_BLOB_ROLE_META_0, HCT_DTYPE_S32, 31u, 1u, 1u, 31u * 4u, meta);
    return run_lstm_once(&session);
}

#define EXPECT(code, cond)                                     \
    do                                                         \
    {                                                          \
        if (!(cond))                                           \
        {                                                      \
            printf("check %d failed\n", code);                 \
            return code;                                       \
        }                                                      \
    } while (0)

int main(void)
{
    /* Valid shapes reach the kernel. */
    EXPECT(1, run_svdf(4u, 3u, 12u) == ARM_CMSIS_NN_SUCCESS && kernel_calls == 1);
    EXPECT(2, run_lstm(2, 8u) == ARM_CMSIS_NN_SUCCESS && kernel_calls == 1);
    /* Kernel reads 4 x 32 time weights. */
    EXPECT(3, run_svdf(1u, 32u, 32u) == ARM_CMSIS_NN_ARG_ERROR && kernel_calls == 0);
    /* Wrapped 1 x 1073741825 x 4 input. */
    EXPECT(4, run_lstm(1073741825, 4u) == ARM_CMSIS_NN_ARG_ERROR && kernel_calls == 0);
    printf("recurrent adapters ok\n");
    return 0;
}
