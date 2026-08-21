#include "benchmark_server_catalog.h"


#ifndef HCT_BENCHMARK_SERVER_BOARD_ID
#define HCT_BENCHMARK_SERVER_BOARD_ID "apollo510_evb"
#endif

#ifndef HCT_BENCHMARK_SERVER_TARGET_CPU
#define HCT_BENCHMARK_SERVER_TARGET_CPU "cortex-m55"
#endif

/* GENERATED FILE -- do not edit by hand.
 * Regenerate with: python3 scripts/generate_kernel_catalog.py
 * Source of truth: assets/kernel_registry.yaml
 */
static const hct_kernel_catalog_entry_t g_hct_kernel_catalog[] = {
    {1u, "arm_abs_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {2u, "arm_convolve_s8", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {3u, "arm_add_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {4u, "arm_sub_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {5u, "arm_mul_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {6u, "arm_maximum_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {7u, "arm_minimum_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {8u, "arm_depthwise_conv_s8", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {9u, "arm_add_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {10u, "arm_sub_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {11u, "arm_mul_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {12u, "arm_maximum_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {13u, "arm_minimum_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {14u, "arm_convolve_wrapper_s16", "ConvolutionFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {15u, "arm_depthwise_conv_wrapper_s16", "ConvolutionFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {16u, "arm_avgpool_s8", "PoolingFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {17u, "arm_max_pool_s8", "PoolingFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {18u, "arm_avgpool_s16", "PoolingFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {19u, "arm_max_pool_s16", "PoolingFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {20u, "arm_relu_s8", "ActivationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {21u, "arm_relu_s16", "ActivationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {22u, "arm_relu_generic_s8", "ActivationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {23u, "arm_relu_generic_s16", "ActivationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {24u, "arm_clamp_s8", "ActivationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {25u, "arm_clamp_s16", "ActivationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {26u, "arm_leaky_relu_s8", "ActivationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {27u, "arm_leaky_relu_s16", "ActivationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {28u, "arm_logistic_s16", "ActivationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {29u, "arm_tanh_s16", "ActivationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {30u, "arm_hard_swish_compat_s8", "ActivationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {31u, "arm_hard_swish_precise_s8", "ActivationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {32u, "arm_hard_swish_precise_s16", "ActivationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {33u, "arm_prelu_s8", "ActivationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {34u, "arm_prelu_s16", "ActivationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {35u, "arm_prelu_scalar_s8", "ActivationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {36u, "arm_prelu_scalar_s16", "ActivationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {37u, "arm_quantize_f32_s8", "QuantizationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {38u, "arm_quantize_f32_s16", "QuantizationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {39u, "arm_dequantize_s8_f32", "QuantizationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {40u, "arm_dequantize_s16_f32", "QuantizationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {41u, "arm_softmax_s8", "SoftmaxFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {42u, "arm_softmax_s16", "SoftmaxFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {43u, "arm_softmax_s8_s16", "SoftmaxFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {44u, "arm_fully_connected_wrapper_s8", "FullyConnectedFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {45u, "arm_fully_connected_wrapper_s16", "FullyConnectedFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {46u, "arm_batch_matmul_s8", "FullyConnectedFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {47u, "arm_batch_matmul_s16", "FullyConnectedFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {48u, "arm_abs_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {49u, "arm_argmax_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {50u, "arm_argmax_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {51u, "arm_argmin_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {52u, "arm_argmin_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {53u, "arm_mean_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {54u, "arm_mean_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {55u, "arm_reduce_max_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {56u, "arm_reduce_max_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {57u, "arm_reduce_min_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {58u, "arm_reduce_min_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {59u, "arm_rsqrt_s16_per_op", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {60u, "arm_rsqrt_s16_universal", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {61u, "arm_sqrt_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {62u, "arm_sqrt_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {63u, "arm_squared_difference_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {64u, "arm_squared_difference_s16", "BasicMathFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {65u, "arm_requantize_s8_s8", "NNSupportFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {66u, "arm_requantize_s16_s16", "NNSupportFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {67u, "arm_equal_s8", "ComparisonFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {68u, "arm_equal_s16", "ComparisonFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {69u, "arm_not_equal_s8", "ComparisonFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {70u, "arm_not_equal_s16", "ComparisonFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {71u, "arm_greater_s8", "ComparisonFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {72u, "arm_greater_s16", "ComparisonFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {73u, "arm_greater_equal_s8", "ComparisonFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {74u, "arm_greater_equal_s16", "ComparisonFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {75u, "arm_less_s8", "ComparisonFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {76u, "arm_less_s16", "ComparisonFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {77u, "arm_less_equal_s8", "ComparisonFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {78u, "arm_less_equal_s16", "ComparisonFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {79u, "arm_transpose_conv_wrapper_s8", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {80u, "arm_reshape_s8", "ReshapeFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {81u, "arm_reshape_s8", "TesterExtensions", 1u, "S8", 1u, true, true, false, 0u},
    {82u, "arm_transpose_s8", "TransposeFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {83u, "arm_transpose_s16", "TransposeFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {84u, "arm_pad_s8", "PadFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {85u, "arm_pad_s16", "PadFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {86u, "arm_mirror_pad_s8", "PadFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {87u, "arm_mirror_pad_s16", "PadFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {88u, "arm_concatenation_s8", "ConcatenationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {89u, "arm_concatenation_s16", "ConcatenationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {90u, "arm_concatenation_s32", "ConcatenationFunctions", 1u, "S32", 1u, true, true, false, 0u},
    {91u, "arm_split_s8", "ConcatenationFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {92u, "arm_split_s16", "ConcatenationFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {93u, "arm_batch_to_space_nd_s8", "ReshapeFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {94u, "arm_batch_to_space_nd_s16", "ReshapeFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {95u, "arm_space_to_batch_nd_s8", "ReshapeFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {96u, "arm_space_to_batch_nd_s16", "ReshapeFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {97u, "arm_space_to_depth_s8", "ReshapeFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {98u, "arm_space_to_depth_s16", "ReshapeFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {99u, "arm_depth_to_space_s8", "ReshapeFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {100u, "arm_depth_to_space_s16", "ReshapeFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {101u, "arm_resize_nearest_neighbor_s8", "ReshapeFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {102u, "arm_resize_nearest_neighbor_s16", "ReshapeFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {103u, "arm_tile_s8", "TileFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {104u, "arm_tile_s16", "TileFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {105u, "arm_gather_s8", "GatherFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {106u, "arm_gather_s16", "GatherFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {107u, "arm_gather_nd_s8", "GatherFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {108u, "arm_gather_nd_s16", "GatherFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {109u, "arm_where_s8", "SelectFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {110u, "arm_where_s16", "SelectFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {111u, "arm_select_v2_s8", "SelectFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {112u, "arm_select_v2_s16", "SelectFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {113u, "arm_reverse_sequence_s8", "ReverseSequenceFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {114u, "arm_reverse_sequence_s16", "ReverseSequenceFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {115u, "arm_scatter_nd_s8", "ScatterFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {116u, "arm_scatter_nd_s16", "ScatterFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {117u, "arm_broadcast_to_s8", "BroadcastFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {118u, "arm_broadcast_to_s16", "BroadcastFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {119u, "arm_dynamic_update_slice_s8", "DynamicUpdateSliceFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {120u, "arm_dynamic_update_slice_s16", "DynamicUpdateSliceFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {121u, "arm_strided_slice_s8", "StridedSliceFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {122u, "arm_strided_slice_s16", "StridedSliceFunctions", 1u, "S16", 1u, true, true, false, 0u},
    {123u, "arm_strided_slice_s32", "StridedSliceFunctions", 1u, "S32", 1u, true, true, false, 0u},
    {124u, "arm_convolve_wrapper_s4", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {125u, "arm_fully_connected_s4", "FullyConnectedFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {126u, "arm_depthwise_conv_wrapper_s4", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {127u, "arm_avg_pool_f32", "PoolingFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {128u, "arm_max_pool_f32", "PoolingFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {129u, "arm_reshape_f32", "ReshapeFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {130u, "arm_reshape_f16", "ReshapeFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {131u, "arm_transpose_f32", "TransposeFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {132u, "arm_transpose_f16", "TransposeFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {133u, "arm_pad_f32", "PadFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {134u, "arm_pad_f16", "PadFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {135u, "arm_strided_slice_f32", "StridedSliceFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {136u, "arm_strided_slice_f16", "StridedSliceFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {137u, "arm_concatenation_f32_w", "ConcatenationFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {138u, "arm_concatenation_f16_w", "ConcatenationFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {139u, "arm_split_f16", "ConcatenationFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {140u, "arm_abs_f32", "BasicMathFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {141u, "arm_abs_f16", "BasicMathFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {142u, "arm_elementwise_add_f32", "BasicMathFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {143u, "arm_elementwise_add_f16", "BasicMathFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {144u, "arm_elementwise_sub_f32", "BasicMathFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {145u, "arm_elementwise_sub_f16", "BasicMathFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {146u, "arm_elementwise_mul_f32", "BasicMathFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {147u, "arm_elementwise_mul_f16", "BasicMathFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {148u, "arm_maximum_f32", "BasicMathFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {149u, "arm_maximum_f16", "BasicMathFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {150u, "arm_minimum_f32", "BasicMathFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {151u, "arm_minimum_f16", "BasicMathFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {152u, "arm_prelu_f32", "ActivationFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {153u, "arm_prelu_f16", "ActivationFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {154u, "arm_softmax_f32", "SoftmaxFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {155u, "arm_softmax_f16", "SoftmaxFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {156u, "arm_avg_pool_f16", "PoolingFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {157u, "arm_max_pool_f16", "PoolingFunctions", 1u, "FP16", 1u, true, true, false, 0u},
};

static const uint8_t g_hct_kernel_catalog_hash[32] = {
    0xdeu, 0x0eu, 0x02u, 0x49u, 0x88u, 0x39u, 0xceu, 0x3fu,
    0x47u, 0xa3u, 0xc6u, 0xcfu, 0x82u, 0xd4u, 0xb7u, 0xadu,
    0x6bu, 0xfau, 0xddu, 0x28u, 0x78u, 0x8cu, 0xfbu, 0x5cu,
    0x81u, 0xf1u, 0xb4u, 0x95u, 0x68u, 0xdcu, 0xc5u, 0xc7u,
};

const hct_kernel_catalog_entry_t *hct_benchmark_server_catalog(size_t *count)
{
    if (count != NULL)
    {
        *count = sizeof(g_hct_kernel_catalog) / sizeof(g_hct_kernel_catalog[0]);
    }
    return g_hct_kernel_catalog;
}

const uint8_t *hct_benchmark_server_catalog_hash(void)
{
    return g_hct_kernel_catalog_hash;
}

const char *hct_benchmark_server_board_id(void)
{
    return HCT_BENCHMARK_SERVER_BOARD_ID;
}

const char *hct_benchmark_server_target_cpu(void)
{
    return HCT_BENCHMARK_SERVER_TARGET_CPU;
}

const char *hct_benchmark_server_build_id(void)
{
    return "hct-benchmark-server-v0";
}

uint32_t hct_benchmark_server_capability_flags(void)
{
    return HCT_CAP_CASE_STREAMING
         | HCT_CAP_CORRECTNESS
         | HCT_CAP_PERFORMANCE
         | HCT_CAP_RTT_TRANSPORT
         | HCT_CAP_KERNEL_CATALOG
         | HCT_CAP_ABS_S8;
}
