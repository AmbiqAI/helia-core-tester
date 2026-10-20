#include "benchmark_server_catalog.h"
#include "benchmark_server_blob_types.h"
#include "benchmark_server_mram.h"


/* The PMU capability is decided by the device header of the hardware build
 * (__PMU_PRESENT / __PMU_NUM_EVENTCNT from apollo*.h via am_mcu_apollo.h); a host
 * compile or a DWT-only core advertises no PMU and zero slots. */
#ifdef HELIA_HARDWARE_BUILD
#include "am_mcu_apollo.h"
#endif
#if defined(__PMU_PRESENT) && (__PMU_PRESENT == 1)
#define HCT_PMU_CAPABILITY_FLAGS HCT_CAP_PMU_ARMV8M
#define HCT_PMU_COUNTER_SLOTS ((uint8_t)__PMU_NUM_EVENTCNT)
#else
#define HCT_PMU_CAPABILITY_FLAGS 0u
#define HCT_PMU_COUNTER_SLOTS 0u
#endif
#if defined(HCT_PLACEMENT_MRAM)
#define HCT_PLACEMENT_CAPABILITY_FLAGS HCT_CAP_WEIGHTS_MRAM
#else
#define HCT_PLACEMENT_CAPABILITY_FLAGS 0u
#endif
#if HCT_BLOB_STORE_BYTES > 0
#define HCT_STORE_CAPABILITY_FLAGS HCT_CAP_BLOB_STORE
#else
#define HCT_STORE_CAPABILITY_FLAGS 0u
#endif

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
    {2u, "arm_convolve_wrapper_s8", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {3u, "arm_add_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {4u, "arm_sub_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {5u, "arm_mul_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {6u, "arm_maximum_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {7u, "arm_minimum_s8", "BasicMathFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {8u, "arm_depthwise_conv_wrapper_s8", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
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
    {140u, "arm_nn_abs_f32", "BasicMathFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {141u, "arm_nn_abs_f16", "BasicMathFunctions", 1u, "FP16", 1u, true, true, false, 0u},
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
    {158u, "arm_nn_activation_f32", "ActivationFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {159u, "arm_nn_activation_f16", "ActivationFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {160u, "arm_reduce_sum_f32", "BasicMathFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {161u, "arm_reduce_sum_f16", "BasicMathFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {162u, "arm_batch_norm_f32", "NNSupportFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {163u, "arm_batch_norm_f16", "NNSupportFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {164u, "arm_fully_connected_f32", "FullyConnectedFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {165u, "arm_fully_connected_f16", "FullyConnectedFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {166u, "arm_transpose_conv_f32", "ConvolutionFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {167u, "arm_transpose_conv_f16", "ConvolutionFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {168u, "arm_depthwise_conv_f32", "ConvolutionFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {169u, "arm_depthwise_conv_f16", "ConvolutionFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {170u, "arm_batch_matmul_f32", "FullyConnectedFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {171u, "arm_batch_matmul_f16", "FullyConnectedFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {172u, "arm_convolve_f32", "ConvolutionFunctions", 1u, "FP32", 1u, true, true, false, 0u},
    {173u, "arm_convolve_f16", "ConvolutionFunctions", 1u, "FP16", 1u, true, true, false, 0u},
    {174u, "hct_empty_call", "Timing", 1u, "S8", 1u, true, true, false, 0u},
    {175u, "arm_convolve_s8_small_cin", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {176u, "arm_convolve_s8_3x3_c16_s1", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {177u, "arm_convolve_1x1_s8_short_k", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {178u, "arm_depthwise_conv_s8_opt_3x3", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {179u, "arm_depthwise_conv_s8_opt_3x3_c64_s1", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {180u, "arm_depthwise_conv_s8_opt_planar", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {181u, "arm_depthwise_conv_s8_opt_channelwise", "ConvolutionFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {182u, "arm_fully_connected_per_channel_packed_s8", "FullyConnectedFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {183u, "arm_svdf_s8", "SVDFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {184u, "arm_svdf_state_s16_s8", "SVDFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {185u, "arm_lstm_unidirectional_s8", "LSTMFunctions", 1u, "S8", 1u, true, true, false, 0u},
    {186u, "arm_transpose_conv_s16", "ConvolutionFunctions", 1u, "S16", 1u, true, true, false, 0u},
};

static const uint8_t g_hct_kernel_catalog_hash[32] = {
    0xc5u, 0x4fu, 0xdfu, 0x00u, 0xbdu, 0x43u, 0x00u, 0x7au,
    0xd2u, 0xd5u, 0x6du, 0x02u, 0x5eu, 0x56u, 0xe3u, 0xeau,
    0x95u, 0x23u, 0x8eu, 0x5fu, 0xd1u, 0x9bu, 0xa6u, 0x56u,
    0xe9u, 0x30u, 0x42u, 0x82u, 0x53u, 0x05u, 0xe4u, 0x69u,
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

/* Per-kernel CASE_META blob contracts: a bit per HCT_DTYPE_* each accepted role may
 * carry, 0 for a role the kernel does not take. */
typedef struct
{
    uint32_t kernel_id;
    uint16_t accepted[HCT_BLOB_ROLE_COUNT];
} hct_kernel_blob_contract_t;

#define HCT_DTYPE_BIT(dtype) ((uint16_t)(1u << (dtype)))

static const hct_kernel_blob_contract_t g_hct_kernel_blob_contracts[] = {
    {1u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {2u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {3u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {4u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {5u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {6u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {7u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {8u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {9u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {10u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {11u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {12u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {13u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {14u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32) | HCT_DTYPE_BIT(HCT_DTYPE_S64), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {15u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S64), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {16u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {17u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {18u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {19u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {20u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {21u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {22u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {23u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {24u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {25u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {26u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {27u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {28u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {29u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {30u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {31u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {32u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {33u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {34u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {35u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {36u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {37u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {38u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {39u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {40u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {41u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {42u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {43u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {44u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {45u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S64), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {46u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {47u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {48u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {49u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {50u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {51u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {52u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {53u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {54u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {55u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {56u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {57u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {58u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {59u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {60u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {61u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {62u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {63u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {64u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {65u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {66u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {67u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {68u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {69u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {70u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {71u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {72u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {73u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {74u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {75u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {76u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {77u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {78u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16)}},
    {79u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {80u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {81u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {82u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {83u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {84u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {85u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {86u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {87u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {88u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_2] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {89u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_2] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {90u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_INPUT_2] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {91u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {92u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {93u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {94u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {95u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {96u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {97u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {98u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {99u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {100u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {101u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {102u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {103u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {104u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {105u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {106u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {107u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {108u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {109u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {110u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {111u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_BOOL), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_2] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {112u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_BOOL), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_2] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {113u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {114u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {115u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {116u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {117u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {118u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {119u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_2] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {120u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_2] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {121u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {122u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {123u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {124u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S4), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {125u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S4), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {126u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S4), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {127u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {128u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {129u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {130u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {131u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {132u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {133u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {134u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {135u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {136u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {137u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_INPUT_2] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {138u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_INPUT_2] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {139u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {140u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {141u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {142u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {143u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {144u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {145u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {146u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {147u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {148u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {149u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {150u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {151u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {152u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {153u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {154u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {155u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {156u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {157u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {158u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {159u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {160u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {161u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {162u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {163u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {164u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {165u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {166u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {167u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {168u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {169u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {170u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {171u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {172u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_F32), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_F32)}},
    {173u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_F16), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_F16)}},
    {174u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8)}},
    {175u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {176u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {177u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {178u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {179u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {180u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {181u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {182u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {183u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_INPUT_2] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {184u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_INPUT_1] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_INPUT_2] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {185u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_META_0] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
    {186u, {[HCT_BLOB_ROLE_INPUT_0] = HCT_DTYPE_BIT(HCT_DTYPE_S16), [HCT_BLOB_ROLE_WEIGHTS] = HCT_DTYPE_BIT(HCT_DTYPE_S8), [HCT_BLOB_ROLE_BIAS] = HCT_DTYPE_BIT(HCT_DTYPE_S64), [HCT_BLOB_ROLE_MULTIPLIER] = HCT_DTYPE_BIT(HCT_DTYPE_S32), [HCT_BLOB_ROLE_SHIFT] = HCT_DTYPE_BIT(HCT_DTYPE_S32)}},
};

bool hct_kernel_blob_accepts(uint32_t kernel_id, uint8_t role, uint8_t dtype)
{
    size_t index;
    if (role == HCT_BLOB_ROLE_UNKNOWN || role >= HCT_BLOB_ROLE_COUNT || dtype == HCT_DTYPE_UNKNOWN ||
        dtype >= 8u * sizeof(uint16_t))
    {
        return false;
    }
    for (index = 0u; index < sizeof(g_hct_kernel_blob_contracts) / sizeof(g_hct_kernel_blob_contracts[0]); ++index)
    {
        if (g_hct_kernel_blob_contracts[index].kernel_id == kernel_id)
        {
            return (g_hct_kernel_blob_contracts[index].accepted[role] & HCT_DTYPE_BIT(dtype)) != 0u;
        }
    }
    return false;
}

const char *hct_benchmark_server_board_id(void)
{
    return HCT_BENCHMARK_SERVER_BOARD_ID;
}

const char *hct_benchmark_server_target_cpu(void)
{
    return HCT_BENCHMARK_SERVER_TARGET_CPU;
}

/* The firmware build defines HCT_BENCHMARK_SERVER_BUILD_ID_PATCHED and links
 * hct_build_id.c, whose slot scripts/patch_build_id.py fills in after the link;
 * this constant only serves host-side unit builds of the protocol code. */
#ifndef HCT_BENCHMARK_SERVER_BUILD_ID_PATCHED
const char *hct_benchmark_server_build_id(void)
{
    return "hct-benchmark-server-v0";
}
#endif

uint32_t hct_benchmark_server_capability_flags(void)
{
    return HCT_CAP_CASE_STREAMING
         | HCT_CAP_CORRECTNESS
         | HCT_CAP_PERFORMANCE
         | HCT_CAP_RTT_TRANSPORT
         | HCT_CAP_KERNEL_CATALOG
         | HCT_CAP_ABS_S8
         | HCT_PMU_CAPABILITY_FLAGS
         | HCT_PLACEMENT_CAPABILITY_FLAGS
         | HCT_STORE_CAPABILITY_FLAGS;
}

uint8_t hct_benchmark_server_pmu_counter_slots(void)
{
    return HCT_PMU_COUNTER_SLOTS;
}
