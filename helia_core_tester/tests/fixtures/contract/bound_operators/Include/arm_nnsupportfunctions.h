/* Verbatim declarations of every function of the contract-bound operators (arm_convolve_*, arm_depthwise_*, arm_fully_connected_*, arm_batch_matmul_*, arm_avgpool_*, arm_avg_pool_*, arm_max_pool_*, arm_relu*, arm_clamp_*, arm_hard_swish_*, arm_leaky_relu_*, arm_logistic_*, arm_tanh_*, arm_nn_activation_*, arm_prelu_*, arm_abs_*, arm_nn_abs_*, arm_mean_*, arm_nn_mean_*, arm_reduce_*, arm_add_*, arm_sub_*, arm_mul_*, arm_elementwise_*, arm_squared_difference_*, arm_maximum_*, arm_minimum_*, arm_argmax_*, arm_argmin_*, arm_nn_fill_*, arm_sqrt_*, arm_equal_*, arm_not_equal_*, arm_greater_*, arm_less_*, arm_comparison_*, arm_broadcast_to_*, arm_batch_to_space_*, arm_space_to_batch_*, arm_depth_to_space_*, arm_space_to_depth_*, arm_strided_slice_*, arm_pad_*, arm_transpose_*, arm_gather_*, arm_resize_nearest_neighbor_*, arm_pack_*, arm_mirror_pad_*, arm_tile_*, arm_reverse_sequence_*, arm_select_v2_*, arm_scatter_nd_*, arm_dynamic_update_slice_*, arm_where_*)
 * from ns-cmsis-nn arm_nnsupportfunctions.h; the fallback contract for unit tests without a checkout.
 * test_bound_operator_fixture_matches_the_real_tree keeps it equal to the export. */

int32_t arm_depthwise_conv_s8_opt_get_buffer_size_dsp(const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims);

int32_t arm_depthwise_conv_s8_opt_get_buffer_size_mve(const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_elementwise_mul_acc_s16(const int16_t *input_1_vect,
                                                const int16_t *input_2_vect,
                                                const int32_t input_1_offset,
                                                const int32_t input_2_offset,
                                                int16_t *output,
                                                const int32_t out_offset,
                                                const int32_t out_mult,
                                                const int32_t out_shift,
                                                const int32_t out_activation_min,
                                                const int32_t out_activation_max,
                                                const int32_t block_size);

arm_cmsis_nn_status arm_elementwise_mul_s16_batch_offset(const int16_t *input_1_vect,
                                                         const int16_t *input_2_vect,
                                                         int16_t *output,
                                                         const int32_t out_offset,
                                                         const int32_t out_mult,
                                                         const int32_t out_shift,
                                                         const int32_t block_size,
                                                         const int32_t batch_size,
                                                         const int32_t batch_offset);

arm_cmsis_nn_status arm_elementwise_mul_s16_s8(const int16_t *input_1_vect,
                                               const int16_t *input_2_vect,
                                               int8_t *output,
                                               const int32_t out_offset,
                                               const int32_t out_mult,
                                               const int32_t out_shift,
                                               const int32_t block_size,
                                               const int32_t batch_size,
                                               const int32_t batch_offset);

