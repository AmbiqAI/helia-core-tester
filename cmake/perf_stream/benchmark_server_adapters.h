#ifndef HCT_BENCHMARK_SERVER_ADAPTERS_H
#define HCT_BENCHMARK_SERVER_ADAPTERS_H

/* Shared by the hand-written session state machine (benchmark_server_session.c) and
 * the generated per-kernel adapters (benchmark_server_adapters.gen.c, rendered from
 * helia_core_tester/perf_stream/adapter_specs.py): the wire enumerations for blob
 * roles, dtypes and comparison modes, the kernel-id table, and the small session
 * accessors every adapter body uses. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "benchmark_server_session.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HCT_BLOB_ROLE_UNKNOWN 0u
#define HCT_BLOB_ROLE_INPUT_0 1u
#define HCT_BLOB_ROLE_WEIGHTS 2u
#define HCT_BLOB_ROLE_BIAS 3u
#define HCT_BLOB_ROLE_MULTIPLIER 4u
#define HCT_BLOB_ROLE_SHIFT 5u
#define HCT_BLOB_ROLE_INPUT_1 6u
#define HCT_BLOB_ROLE_INPUT_2 7u
#define HCT_BLOB_ROLE_META_0 8u

#define HCT_DTYPE_UNKNOWN 0u
#define HCT_DTYPE_S8 1u
#define HCT_DTYPE_S32 2u
#define HCT_DTYPE_S16 3u
#define HCT_DTYPE_S64 4u
#define HCT_DTYPE_BOOL 5u
#define HCT_DTYPE_F32 6u
#define HCT_DTYPE_F16 7u
/* S4 weight blobs are packed 2 nibbles/byte on the wire (same convention the host
 * uses -- see generated_test_bridge.py's expected_weight_bytes) and are never used
 * as an activation dtype, so they never flow through hct_dtype_size_bytes()/
 * hct_blob_element_count() -- weights blob sizing always comes from the wire's own
 * byte_length/alignment fields (see allocate_blob()), not a per-dtype element size. */
#define HCT_DTYPE_S4 8u

#define HCT_PADDING_VALID 0
#define HCT_PADDING_SAME 1

#define HCT_COMPARISON_MODE_EXACT_INT 1u
#define HCT_COMPARISON_MODE_TOLERANT_INT 2u
#define HCT_COMPARISON_MODE_FLOAT 3u
#define HCT_COMPARISON_MODE_BOOL 4u
#define HCT_COMPARISON_MODE_EXACT_STATUS 5u

#define HCT_NULL_ARG_INPUT0_BIT (1 << 0)
#define HCT_NULL_ARG_INPUT1_BIT (1 << 1)
#define HCT_NULL_ARG_INPUT2_BIT (1 << 2)
#define HCT_NULL_ARG_PARAMS_BIT (1 << 3)
#define HCT_NULL_ARG_OUTPUT_BIT (1 << 4)

/* Kernel IDs sent by the host in CASE_META -- must match assets/kernel_registry.yaml
 * (the Python-side single source of truth) and helia_core_tester/perf_stream/kernel_registry.py. */
#define HCT_KERNEL_ID_ABS_S8 1u
#define HCT_KERNEL_ID_CONVOLVE_S8 2u
#define HCT_KERNEL_ID_ADD_S8 3u
#define HCT_KERNEL_ID_SUB_S8 4u
#define HCT_KERNEL_ID_MUL_S8 5u
#define HCT_KERNEL_ID_MAXIMUM_S8 6u
#define HCT_KERNEL_ID_MINIMUM_S8 7u
#define HCT_KERNEL_ID_DEPTHWISE_CONV_S8 8u
#define HCT_KERNEL_ID_ADD_S16 9u
#define HCT_KERNEL_ID_SUB_S16 10u
#define HCT_KERNEL_ID_MUL_S16 11u
#define HCT_KERNEL_ID_MAXIMUM_S16 12u
#define HCT_KERNEL_ID_MINIMUM_S16 13u
#define HCT_KERNEL_ID_CONVOLVE_S16 14u
#define HCT_KERNEL_ID_DEPTHWISE_CONV_S16 15u
#define HCT_KERNEL_ID_AVGPOOL_S8 16u
#define HCT_KERNEL_ID_MAXPOOL_S8 17u
#define HCT_KERNEL_ID_AVGPOOL_S16 18u
#define HCT_KERNEL_ID_MAXPOOL_S16 19u
#define HCT_KERNEL_ID_RELU_S8 20u
#define HCT_KERNEL_ID_RELU_S16 21u
#define HCT_KERNEL_ID_RELU6_S8 22u
#define HCT_KERNEL_ID_RELU6_S16 23u
#define HCT_KERNEL_ID_CLAMP_S8 24u
#define HCT_KERNEL_ID_CLAMP_S16 25u
#define HCT_KERNEL_ID_LEAKY_RELU_S8 26u
#define HCT_KERNEL_ID_LEAKY_RELU_S16 27u
#define HCT_KERNEL_ID_LOGISTIC_S16 28u
#define HCT_KERNEL_ID_TANH_S16 29u
#define HCT_KERNEL_ID_HARD_SWISH_COMPAT_S8 30u
#define HCT_KERNEL_ID_HARD_SWISH_PRECISE_S8 31u
#define HCT_KERNEL_ID_HARD_SWISH_PRECISE_S16 32u
#define HCT_KERNEL_ID_PRELU_S8 33u
#define HCT_KERNEL_ID_PRELU_S16 34u
#define HCT_KERNEL_ID_PRELU_SCALAR_S8 35u
#define HCT_KERNEL_ID_PRELU_SCALAR_S16 36u
#define HCT_KERNEL_ID_QUANTIZE_S8 37u
#define HCT_KERNEL_ID_QUANTIZE_S16 38u
#define HCT_KERNEL_ID_DEQUANTIZE_S8 39u
#define HCT_KERNEL_ID_DEQUANTIZE_S16 40u
#define HCT_KERNEL_ID_SOFTMAX_S8 41u
#define HCT_KERNEL_ID_SOFTMAX_S16 42u
#define HCT_KERNEL_ID_SOFTMAX_S8_S16 43u
#define HCT_KERNEL_ID_FULLY_CONNECTED_S8 44u
#define HCT_KERNEL_ID_FULLY_CONNECTED_S16 45u
#define HCT_KERNEL_ID_BATCH_MATMUL_S8 46u
#define HCT_KERNEL_ID_BATCH_MATMUL_S16 47u
#define HCT_KERNEL_ID_ABS_S16 48u
#define HCT_KERNEL_ID_ARGMAX_S8 49u
#define HCT_KERNEL_ID_ARGMAX_S16 50u
#define HCT_KERNEL_ID_ARGMIN_S8 51u
#define HCT_KERNEL_ID_ARGMIN_S16 52u
#define HCT_KERNEL_ID_MEAN_S8 53u
#define HCT_KERNEL_ID_MEAN_S16 54u
#define HCT_KERNEL_ID_REDUCE_MAX_S8 55u
#define HCT_KERNEL_ID_REDUCE_MAX_S16 56u
#define HCT_KERNEL_ID_REDUCE_MIN_S8 57u
#define HCT_KERNEL_ID_REDUCE_MIN_S16 58u
#define HCT_KERNEL_ID_RSQRT_S16_PER_OP 59u
#define HCT_KERNEL_ID_RSQRT_S16_UNIVERSAL 60u
#define HCT_KERNEL_ID_SQRT_S8 61u
#define HCT_KERNEL_ID_SQRT_S16 62u
#define HCT_KERNEL_ID_SQUARED_DIFFERENCE_S8 63u
#define HCT_KERNEL_ID_SQUARED_DIFFERENCE_S16 64u
#define HCT_KERNEL_ID_REQUANTIZE_S8 65u
#define HCT_KERNEL_ID_REQUANTIZE_S16 66u
#define HCT_KERNEL_ID_EQUAL_S8 67u
#define HCT_KERNEL_ID_EQUAL_S16 68u
#define HCT_KERNEL_ID_NOT_EQUAL_S8 69u
#define HCT_KERNEL_ID_NOT_EQUAL_S16 70u
#define HCT_KERNEL_ID_GREATER_S8 71u
#define HCT_KERNEL_ID_GREATER_S16 72u
#define HCT_KERNEL_ID_GREATER_EQUAL_S8 73u
#define HCT_KERNEL_ID_GREATER_EQUAL_S16 74u
#define HCT_KERNEL_ID_LESS_S8 75u
#define HCT_KERNEL_ID_LESS_S16 76u
#define HCT_KERNEL_ID_LESS_EQUAL_S8 77u
#define HCT_KERNEL_ID_LESS_EQUAL_S16 78u
#define HCT_KERNEL_ID_TRANSPOSE_CONV_S8 79u
#define HCT_KERNEL_ID_RESHAPE_S8 80u
#define HCT_KERNEL_ID_SQUEEZE_S8 81u
#define HCT_KERNEL_ID_TRANSPOSE_S8 82u
#define HCT_KERNEL_ID_TRANSPOSE_S16 83u
#define HCT_KERNEL_ID_PAD_S8 84u
#define HCT_KERNEL_ID_PAD_S16 85u
#define HCT_KERNEL_ID_MIRROR_PAD_S8 86u
#define HCT_KERNEL_ID_MIRROR_PAD_S16 87u
#define HCT_KERNEL_ID_CONCATENATION_S8 88u
#define HCT_KERNEL_ID_CONCATENATION_S16 89u
#define HCT_KERNEL_ID_CONCATENATION_S32 90u
#define HCT_KERNEL_ID_SPLIT_S8 91u
#define HCT_KERNEL_ID_SPLIT_S16 92u
#define HCT_KERNEL_ID_BATCH_TO_SPACE_ND_S8 93u
#define HCT_KERNEL_ID_BATCH_TO_SPACE_ND_S16 94u
#define HCT_KERNEL_ID_SPACE_TO_BATCH_ND_S8 95u
#define HCT_KERNEL_ID_SPACE_TO_BATCH_ND_S16 96u
#define HCT_KERNEL_ID_SPACE_TO_DEPTH_S8 97u
#define HCT_KERNEL_ID_SPACE_TO_DEPTH_S16 98u
#define HCT_KERNEL_ID_DEPTH_TO_SPACE_S8 99u
#define HCT_KERNEL_ID_DEPTH_TO_SPACE_S16 100u
#define HCT_KERNEL_ID_RESIZE_NEAREST_NEIGHBOR_S8 101u
#define HCT_KERNEL_ID_RESIZE_NEAREST_NEIGHBOR_S16 102u
#define HCT_KERNEL_ID_TILE_S8 103u
#define HCT_KERNEL_ID_TILE_S16 104u
#define HCT_KERNEL_ID_GATHER_S8 105u
#define HCT_KERNEL_ID_GATHER_S16 106u
#define HCT_KERNEL_ID_GATHER_ND_S8 107u
#define HCT_KERNEL_ID_GATHER_ND_S16 108u
#define HCT_KERNEL_ID_WHERE_S8 109u
#define HCT_KERNEL_ID_WHERE_S16 110u
#define HCT_KERNEL_ID_SELECT_V2_S8 111u
#define HCT_KERNEL_ID_SELECT_V2_S16 112u
#define HCT_KERNEL_ID_REVERSE_SEQUENCE_S8 113u
#define HCT_KERNEL_ID_REVERSE_SEQUENCE_S16 114u
#define HCT_KERNEL_ID_SCATTER_ND_S8 115u
#define HCT_KERNEL_ID_SCATTER_ND_S16 116u
#define HCT_KERNEL_ID_BROADCAST_TO_S8 117u
#define HCT_KERNEL_ID_BROADCAST_TO_S16 118u
#define HCT_KERNEL_ID_DYNAMIC_UPDATE_SLICE_S8 119u
#define HCT_KERNEL_ID_DYNAMIC_UPDATE_SLICE_S16 120u
#define HCT_KERNEL_ID_STRIDED_SLICE_S8 121u
#define HCT_KERNEL_ID_STRIDED_SLICE_S16 122u
#define HCT_KERNEL_ID_STRIDED_SLICE_S32 123u
#define HCT_KERNEL_ID_CONVOLVE_S4 124u
#define HCT_KERNEL_ID_FULLY_CONNECTED_S4 125u
#define HCT_KERNEL_ID_DEPTHWISE_CONV_S4 126u
#define HCT_KERNEL_ID_AVGPOOL_F32 127u
#define HCT_KERNEL_ID_MAXPOOL_F32 128u
#define HCT_KERNEL_ID_RESHAPE_F32 129u
#define HCT_KERNEL_ID_RESHAPE_F16 130u
#define HCT_KERNEL_ID_TRANSPOSE_F32 131u
#define HCT_KERNEL_ID_TRANSPOSE_F16 132u
#define HCT_KERNEL_ID_PAD_F32 133u
#define HCT_KERNEL_ID_PAD_F16 134u
#define HCT_KERNEL_ID_STRIDED_SLICE_F32 135u
#define HCT_KERNEL_ID_STRIDED_SLICE_F16 136u
#define HCT_KERNEL_ID_CONCATENATION_F32 137u
#define HCT_KERNEL_ID_CONCATENATION_F16 138u
#define HCT_KERNEL_ID_SPLIT_F16 139u
#define HCT_KERNEL_ID_ABS_F32 140u
#define HCT_KERNEL_ID_ABS_F16 141u
#define HCT_KERNEL_ID_ADD_F32 142u
#define HCT_KERNEL_ID_ADD_F16 143u
#define HCT_KERNEL_ID_SUB_F32 144u
#define HCT_KERNEL_ID_SUB_F16 145u
#define HCT_KERNEL_ID_MUL_F32 146u
#define HCT_KERNEL_ID_MUL_F16 147u
#define HCT_KERNEL_ID_MAXIMUM_F32 148u
#define HCT_KERNEL_ID_MAXIMUM_F16 149u
#define HCT_KERNEL_ID_MINIMUM_F32 150u
#define HCT_KERNEL_ID_MINIMUM_F16 151u
#define HCT_KERNEL_ID_PRELU_F32 152u
#define HCT_KERNEL_ID_PRELU_F16 153u
#define HCT_KERNEL_ID_SOFTMAX_F32 154u
#define HCT_KERNEL_ID_SOFTMAX_F16 155u
#define HCT_KERNEL_ID_AVGPOOL_F16 156u
#define HCT_KERNEL_ID_MAXPOOL_F16 157u
#define HCT_KERNEL_ID_NN_ACTIVATION_FLOAT_F32 158u
#define HCT_KERNEL_ID_NN_ACTIVATION_FLOAT_F16 159u
#define HCT_KERNEL_ID_REDUCE_SUM_F32 160u
#define HCT_KERNEL_ID_REDUCE_SUM_F16 161u
#define HCT_KERNEL_ID_BATCH_NORM_F32 162u
#define HCT_KERNEL_ID_BATCH_NORM_F16 163u
#define HCT_KERNEL_ID_FULLY_CONNECTED_F32 164u
#define HCT_KERNEL_ID_FULLY_CONNECTED_F16 165u
#define HCT_KERNEL_ID_TRANSPOSE_CONV_F32 166u
#define HCT_KERNEL_ID_TRANSPOSE_CONV_F16 167u
#define HCT_KERNEL_ID_DEPTHWISE_CONV_F32 168u
#define HCT_KERNEL_ID_DEPTHWISE_CONV_F16 169u
#define HCT_KERNEL_ID_BATCH_MATMUL_F32 170u
#define HCT_KERNEL_ID_BATCH_MATMUL_F16 171u
#define HCT_KERNEL_ID_CONVOLVE_F32 172u
#define HCT_KERNEL_ID_CONVOLVE_F16 173u

static inline hct_server_blob_t *find_blob_by_role(hct_server_session_t *session, uint8_t role)
{
    uint16_t index;
    for (index = 0u; index < session->blob_count; ++index)
    {
        if (session->blobs[index].role == role)
        {
            return &session->blobs[index];
        }
    }
    return NULL;
}

static inline uint8_t *blob_ptr(hct_server_session_t *session, const hct_server_blob_t *blob)
{
    return &session->workspace[blob->arena_offset];
}

static inline uint8_t *hct_output_ptr(hct_server_session_t *session)
{
    return &session->workspace[session->output_workspace_offset];
}

static inline bool expects_exact_status(const hct_server_session_t *session)
{
    return session->comparison_mode == HCT_COMPARISON_MODE_EXACT_STATUS;
}

static inline bool null_arg_requested(const hct_server_session_t *session, int32_t bit)
{
    return (session->null_arg_mask & bit) != 0;
}


/* Generated dispatch (benchmark_server_adapters.gen.c): runs the adapter for every
 * kernel id except the hand-written abs adapters; ARM_CMSIS_NN_ARG_ERROR for an id it
 * does not know. Not compiled into the HCT_HOST_ABS_ONLY host harness. */
arm_cmsis_nn_status hct_run_adapter_once(hct_server_session_t *session);

#ifdef __cplusplus
}
#endif

#endif
