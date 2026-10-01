/* Verbatim declarations of every function of the contract-bound operators (arm_convolve_*, arm_depthwise_*, arm_fully_connected_*, arm_batch_matmul_*, arm_avgpool_*, arm_avg_pool_*, arm_max_pool_*, arm_relu*, arm_clamp_*, arm_hard_swish_*, arm_leaky_relu_*, arm_logistic_*, arm_tanh_*, arm_nn_activation_*, arm_prelu_*, arm_abs_*, arm_nn_abs_*, arm_mean_*, arm_nn_mean_*, arm_reduce_*, arm_add_*, arm_sub_*, arm_mul_*, arm_elementwise_*, arm_squared_difference_*, arm_maximum_*, arm_minimum_*, arm_argmax_*, arm_argmin_*, arm_nn_fill_*, arm_sqrt_*, arm_equal_*, arm_not_equal_*, arm_greater_*, arm_less_*, arm_comparison_*, arm_broadcast_to_*, arm_batch_to_space_*, arm_space_to_batch_*, arm_depth_to_space_*, arm_space_to_depth_*, arm_strided_slice_*, arm_pad_*, arm_transpose_*, arm_gather_*, arm_resize_nearest_neighbor_*, arm_pack_*, arm_mirror_pad_*, arm_tile_*, arm_reverse_sequence_*, arm_select_v2_*, arm_scatter_nd_*, arm_dynamic_update_slice_*, arm_where_*, arm_requantize_*, arm_batch_norm_*, arm_softmax_*, arm_split_*, arm_unpack_*, arm_quantize_*, arm_dequantize_*, arm_concatenation_*, arm_rsqrt_*, arm_reshape_*, arm_nn_sqrt_*, arm_svdf_*, arm_lstm_unidirectional_*, arm_gru_unidirectional_*)
 * from ns-cmsis-nn arm_nnfunctions.h; the fallback contract for unit tests without a checkout.
 * test_bound_operator_fixture_matches_the_real_tree keeps it equal to the export. */

arm_cmsis_nn_status arm_abs_s16(const int16_t *input,
                                const int32_t input_offset,
                                int16_t *output,
                                const int32_t out_offset,
                                const int32_t out_mult,
                                const int32_t out_shift,
                                const bool needs_rescale,
                                const int32_t out_activation_min,
                                const int32_t out_activation_max,
                                const int32_t block_size);

arm_cmsis_nn_status arm_abs_s8(const int8_t *input,
                               const int32_t input_offset,
                               int8_t *output,
                               const int32_t out_offset,
                               const int32_t out_mult,
                               const int32_t out_shift,
                               const bool needs_rescale,
                               const int32_t out_activation_min,
                               const int32_t out_activation_max,
                               const int32_t block_size);

arm_cmsis_nn_status arm_add_s16(const int16_t *input1_data,
                                const cmsis_nn_dims *input1_dims,
                                const int16_t *input2_data,
                                const cmsis_nn_dims *input2_dims,
                                const int32_t input1_offset,
                                const int32_t input1_mult,
                                const int32_t input1_shift,
                                const int32_t input2_offset,
                                const int32_t input2_mult,
                                const int32_t input2_shift,
                                const int32_t left_shift,
                                int16_t *output_data,
                                const cmsis_nn_dims *output_dims,
                                const int32_t out_offset,
                                const int32_t out_mult,
                                const int32_t out_shift,
                                const int32_t out_activation_min,
                                const int32_t out_activation_max);

arm_cmsis_nn_status arm_add_s8(const int8_t *input1_data,
                               const cmsis_nn_dims *input1_dims,
                               const int8_t *input2_data,
                               const cmsis_nn_dims *input2_dims,
                               const int32_t input1_offset,
                               const int32_t input1_mult,
                               const int32_t input1_shift,
                               const int32_t input2_offset,
                               const int32_t input2_mult,
                               const int32_t input2_shift,
                               const int32_t left_shift,
                               int8_t *output_data,
                               const cmsis_nn_dims *output_dims,
                               const int32_t out_offset,
                               const int32_t out_mult,
                               const int32_t out_shift,
                               const int32_t out_activation_min,
                               const int32_t out_activation_max);

arm_cmsis_nn_status arm_add_scalar_s16(const int16_t *input_1_vect,
                                       const int16_t *input_2_vect,
                                       const int32_t input_1_offset,
                                       const int32_t input_1_mult,
                                       const int32_t input_1_shift,
                                       const int32_t input_2_offset,
                                       const int32_t input_2_mult,
                                       const int32_t input_2_shift,
                                       const int32_t left_shift,
                                       int16_t *output,
                                       const int32_t out_offset,
                                       const int32_t out_mult,
                                       const int32_t out_shift,
                                       const int32_t out_activation_min,
                                       const int32_t out_activation_max,
                                       const int32_t block_size);

arm_cmsis_nn_status arm_add_scalar_s8(const int8_t *input_1_vect,
                                      const int8_t *input_2_vect,
                                      const int32_t input_1_offset,
                                      const int32_t input_1_mult,
                                      const int32_t input_1_shift,
                                      const int32_t input_2_offset,
                                      const int32_t input_2_mult,
                                      const int32_t input_2_shift,
                                      const int32_t left_shift,
                                      int8_t *output,
                                      const int32_t out_offset,
                                      const int32_t out_mult,
                                      const int32_t out_shift,
                                      const int32_t out_activation_min,
                                      const int32_t out_activation_max,
                                      const int32_t block_size);

arm_cmsis_nn_status
arm_argmax_s16(const int16_t *input_data, const cmsis_nn_dims *input_dims, const int32_t axis, int32_t *output_data);

arm_cmsis_nn_status
arm_argmax_s8(const int8_t *input_data, const cmsis_nn_dims *input_dims, const int32_t axis, int32_t *output_data);

arm_cmsis_nn_status
arm_argmin_s16(const int16_t *input_data, const cmsis_nn_dims *input_dims, const int32_t axis, int32_t *output_data);

arm_cmsis_nn_status
arm_argmin_s8(const int8_t *input_data, const cmsis_nn_dims *input_dims, const int32_t axis, int32_t *output_data);

arm_cmsis_nn_status arm_avgpool_s16(const cmsis_nn_context *ctx,
                                    const cmsis_nn_pool_params *pool_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int16_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const cmsis_nn_dims *output_dims,
                                    int16_t *output_data);

int32_t arm_avgpool_s16_get_buffer_size(const int dim_dst_width, const int ch_src);

int32_t arm_avgpool_s16_get_buffer_size_dsp(const int dim_dst_width, const int ch_src);

int32_t arm_avgpool_s16_get_buffer_size_mve(const int dim_dst_width, const int ch_src);

arm_cmsis_nn_status arm_avgpool_s8(const cmsis_nn_context *ctx,
                                   const cmsis_nn_pool_params *pool_params,
                                   const cmsis_nn_dims *input_dims,
                                   const int8_t *input_data,
                                   const cmsis_nn_dims *filter_dims,
                                   const cmsis_nn_dims *output_dims,
                                   int8_t *output_data);

int32_t arm_avgpool_s8_get_buffer_size(const int dim_dst_width, const int ch_src);

int32_t arm_avgpool_s8_get_buffer_size_dsp(const int dim_dst_width, const int ch_src);

int32_t arm_avgpool_s8_get_buffer_size_mve(const int dim_dst_width, const int ch_src);

arm_cmsis_nn_status arm_batch_matmul_s16(const cmsis_nn_context *ctx,
                                         const cmsis_nn_bmm_params *bmm_params,
                                         const cmsis_nn_per_tensor_quant_params *quant_params,
                                         const cmsis_nn_dims *input_lhs_dims,
                                         const int16_t *input_lhs,
                                         const cmsis_nn_dims *input_rhs_dims,
                                         const int16_t *input_rhs,
                                         const cmsis_nn_dims *output_dims,
                                         int16_t *output);

arm_cmsis_nn_status arm_batch_matmul_s8(const cmsis_nn_context *ctx,
                                        const cmsis_nn_bmm_params *bmm_params,
                                        const cmsis_nn_per_tensor_quant_params *quant_params,
                                        const cmsis_nn_dims *input_lhs_dims,
                                        const int8_t *input_lhs,
                                        const cmsis_nn_dims *input_rhs_dims,
                                        const int8_t *input_rhs,
                                        const cmsis_nn_dims *output_dims,
                                        int8_t *output);

int32_t arm_batch_matmul_s8_get_buffer_size(const cmsis_nn_dims *input_rhs_dims);

int32_t arm_batch_matmul_s8_get_buffer_size_dsp(const cmsis_nn_dims *input_rhs_dims);

int32_t arm_batch_matmul_s8_get_buffer_size_mve(const cmsis_nn_dims *input_rhs_dims);

arm_cmsis_nn_status arm_batch_to_space_nd_s16(const int16_t *input_data,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_tile *block_shape,
                                              const cmsis_nn_dims *crop, // n->top, h->left, w->bottom, c->right
                                              int16_t *output_data,
                                              const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_batch_to_space_nd_s8(const int8_t *input_data,
                                             const cmsis_nn_dims *input_dims,
                                             const cmsis_nn_tile *block_shape,
                                             const cmsis_nn_dims *crop, // n->top, h->left, w->bottom, c->right
                                             int8_t *output_data,
                                             const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status
arm_broadcast_to_s16(const int16_t *input, const cmsis_nn_broadcast_to_params *params, int16_t *output);

arm_cmsis_nn_status
arm_broadcast_to_s8(const int8_t *input, const cmsis_nn_broadcast_to_params *params, int8_t *output);

arm_cmsis_nn_status arm_clamp_s16(const int16_t *input,
                                  const int16_t act_min,
                                  const int16_t act_max,
                                  int16_t *output,
                                  const int32_t output_size);

arm_cmsis_nn_status arm_clamp_s8(const int8_t *input,
                                 const int8_t act_min,
                                 const int8_t act_max,
                                 int8_t *output,
                                 const int32_t output_size);

arm_cmsis_nn_status arm_comparison_s16(const cmsis_nn_context *ctx,
                                       const int16_t *input_1_data,
                                       const cmsis_nn_dims *input_1_dims,
                                       const int16_t *input_2_data,
                                       const cmsis_nn_dims *input_2_dims,
                                       bool *output_data,
                                       const cmsis_nn_dims *output_dims,
                                       const int32_t input_1_offset,
                                       const int32_t input_1_mult,
                                       const int32_t input_1_shift,
                                       const int32_t input_2_offset,
                                       const int32_t input_2_mult,
                                       const int32_t input_2_shift,
                                       const int32_t left_shift,
                                       arm_nn_compare_operation operation);

arm_cmsis_nn_status arm_comparison_s8(const cmsis_nn_context *ctx,
                                      const int8_t *input_1_data,
                                      const cmsis_nn_dims *input_1_dims,
                                      const int8_t *input_2_data,
                                      const cmsis_nn_dims *input_2_dims,
                                      bool *output_data,
                                      const cmsis_nn_dims *output_dims,
                                      const int32_t input_1_offset,
                                      const int32_t input_1_mult,
                                      const int32_t input_1_shift,
                                      const int32_t input_2_offset,
                                      const int32_t input_2_mult,
                                      const int32_t input_2_shift,
                                      const int32_t left_shift,
                                      arm_nn_compare_operation operation);

arm_cmsis_nn_status arm_concatenation_s16(const int16_t *const *input_data,
                                          const int32_t inputs_count,
                                          const int32_t *input_concat_dims,
                                          const int32_t axis,
                                          int16_t *output_data,
                                          const int32_t output_dims,
                                          const int32_t *output_shape);

arm_cmsis_nn_status arm_concatenation_s32(const int32_t *const *input_data,
                                          const int32_t inputs_count,
                                          const int32_t *input_concat_dims,
                                          const int32_t axis,
                                          int32_t *output_data,
                                          const int32_t output_dims,
                                          const int32_t *output_shape);

arm_cmsis_nn_status arm_concatenation_s8(const int8_t *const *input_data,
                                         const int32_t inputs_count,
                                         const int32_t *input_concat_dims,
                                         const int32_t axis,
                                         int8_t *output_data,
                                         const int32_t output_dims,
                                         const int32_t *output_shape);

void arm_concatenation_s8_w(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint32_t offset_w);

void arm_concatenation_s8_x(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint16_t output_x,
                            const uint32_t offset_x);

void arm_concatenation_s8_y(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint16_t output_y,
                            const uint32_t offset_y);

void arm_concatenation_s8_z(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint16_t output_z,
                            const uint32_t offset_z);

arm_cmsis_nn_status arm_convolve_1_x_n_s4(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params *conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data);

int32_t arm_convolve_1_x_n_s4_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_convolve_1_x_n_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_context *weight_sum_ctx,
                                          const cmsis_nn_conv_params *conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data);

int32_t arm_convolve_1_x_n_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_convolve_1x1_out_s8(const cmsis_nn_context *ctx,
                                            const cmsis_nn_context *weight_sum_ctx,
                                            const cmsis_nn_conv_params *conv_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int8_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            int8_t *output_data);

int32_t arm_convolve_1x1_out_s8_get_buffer_size(const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_convolve_1x1_s16_ns_np_nd(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_conv_params *conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int16_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const cmsis_nn_bias_data *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  int16_t *output_data);

arm_cmsis_nn_status arm_convolve_1x1_s4(const cmsis_nn_context *ctx,
                                        const cmsis_nn_conv_params *conv_params,
                                        const cmsis_nn_per_channel_quant_params *quant_params,
                                        const cmsis_nn_dims *input_dims,
                                        const int8_t *input_data,
                                        const cmsis_nn_dims *filter_dims,
                                        const int8_t *filter_data,
                                        const cmsis_nn_dims *bias_dims,
                                        const int32_t *bias_data,
                                        const cmsis_nn_dims *output_dims,
                                        int8_t *output_data);

arm_cmsis_nn_status arm_convolve_1x1_s4_fast(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params *conv_params,
                                             const cmsis_nn_per_channel_quant_params *quant_params,
                                             const cmsis_nn_dims *input_dims,
                                             const int8_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const int8_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const int32_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             int8_t *output_data);

int32_t arm_convolve_1x1_s4_fast_get_buffer_size(const cmsis_nn_dims *input_dims);

arm_cmsis_nn_status arm_convolve_1x1_s8(const cmsis_nn_context *ctx,
                                        const cmsis_nn_context *weight_sum_ctx,
                                        const cmsis_nn_conv_params *conv_params,
                                        const cmsis_nn_per_channel_quant_params *quant_params,
                                        const cmsis_nn_dims *input_dims,
                                        const int8_t *input_data,
                                        const cmsis_nn_dims *filter_dims,
                                        const int8_t *filter_data,
                                        const cmsis_nn_dims *bias_dims,
                                        const int32_t *bias_data,
                                        const cmsis_nn_dims *output_dims,
                                        int8_t *output_data);

arm_cmsis_nn_status arm_convolve_1x1_s8_fast(const cmsis_nn_context *ctx,
                                             const cmsis_nn_context *weight_sum_ctx,
                                             const cmsis_nn_conv_params *conv_params,
                                             const cmsis_nn_per_channel_quant_params *quant_params,
                                             const cmsis_nn_dims *input_dims,
                                             const int8_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const int8_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const int32_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             int8_t *output_data);

int32_t arm_convolve_1x1_s8_fast_get_buffer_size(const cmsis_nn_dims *input_dims);

arm_cmsis_nn_status arm_convolve_even_s4(const cmsis_nn_context *ctx,
                                         const cmsis_nn_conv_params *conv_params,
                                         const cmsis_nn_per_channel_quant_params *quant_params,
                                         const cmsis_nn_dims *input_dims,
                                         const int8_t *input_data,
                                         const cmsis_nn_dims *filter_dims,
                                         const int8_t *filter_data,
                                         const cmsis_nn_dims *bias_dims,
                                         const int32_t *bias_data,
                                         const cmsis_nn_dims *output_dims,
                                         int8_t *output_data);

int32_t arm_convolve_even_s4_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_convolve_s16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params *conv_params,
                                     const cmsis_nn_per_channel_quant_params *quant_params,
                                     const cmsis_nn_dims *input_dims,
                                     const int16_t *input_data,
                                     const cmsis_nn_dims *filter_dims,
                                     const int8_t *filter_data,
                                     const cmsis_nn_dims *bias_dims,
                                     const cmsis_nn_bias_data *bias_data,
                                     const cmsis_nn_dims *output_dims,
                                     int16_t *output_data);

arm_cmsis_nn_status arm_convolve_s16_fast_small_kernel(const cmsis_nn_context *ctx,
                                                       const cmsis_nn_conv_params *conv_params,
                                                       const cmsis_nn_per_channel_quant_params *quant_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const int16_t *input_data,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const int8_t *filter_data,
                                                       const cmsis_nn_dims *bias_dims,
                                                       const cmsis_nn_bias_data *bias_data,
                                                       const cmsis_nn_dims *output_dims,
                                                       int16_t *output_data);

int32_t arm_convolve_s16_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_convolve_s16_group_ch_mult_1(const cmsis_nn_context *ctx,
                                                     const cmsis_nn_conv_params *conv_params,
                                                     const cmsis_nn_per_channel_quant_params *quant_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const int16_t *input_data,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const int8_t *filter_data,
                                                     const cmsis_nn_dims *bias_dims,
                                                     const cmsis_nn_bias_data *bias_data,
                                                     const cmsis_nn_dims *output_dims,
                                                     int16_t *output_data);

arm_cmsis_nn_status arm_convolve_s4(const cmsis_nn_context *ctx,
                                    const cmsis_nn_conv_params *conv_params,
                                    const cmsis_nn_per_channel_quant_params *quant_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int8_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const int8_t *filter_data,
                                    const cmsis_nn_dims *bias_dims,
                                    const int32_t *bias_data,
                                    const cmsis_nn_dims *output_dims,
                                    int8_t *output_data);

int32_t arm_convolve_s4_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_convolve_s8(const cmsis_nn_context *ctx,
                                    const cmsis_nn_context *weight_sum_ctx,
                                    const cmsis_nn_conv_params *conv_params,
                                    const cmsis_nn_per_channel_quant_params *quant_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int8_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const int8_t *filter_data,
                                    const cmsis_nn_dims *bias_dims,
                                    const int32_t *bias_data,
                                    const cmsis_nn_dims *upscale_dims,
                                    const cmsis_nn_dims *output_dims,
                                    int8_t *output_data);

arm_cmsis_nn_status arm_convolve_s8_3x3_c16_s1(const cmsis_nn_context *ctx,
                                               const cmsis_nn_context *weight_sum_ctx,
                                               const cmsis_nn_conv_params *conv_params,
                                               const cmsis_nn_per_channel_quant_params *quant_params,
                                               const cmsis_nn_dims *input_dims,
                                               const int8_t *input_data,
                                               const cmsis_nn_dims *filter_dims,
                                               const int8_t *filter_data,
                                               const cmsis_nn_dims *bias_dims,
                                               const int32_t *bias_data,
                                               const cmsis_nn_dims *upscale_dims,
                                               const cmsis_nn_dims *output_dims,
                                               int8_t *output_data);

int32_t arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

int32_t arm_convolve_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

int32_t arm_convolve_s8_get_weights_sum_size(const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_convolve_s8_small_cin(const cmsis_nn_context *ctx,
                                              const cmsis_nn_context *weight_sum_ctx,
                                              const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int8_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const int8_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int32_t *bias_data,
                                              const cmsis_nn_dims *upscale_dims,
                                              const cmsis_nn_dims *output_dims,
                                              int8_t *output_data);

arm_cmsis_nn_status arm_convolve_weight_sum(int32_t *vector_sum_buf,
                                            const int8_t *rhs,
                                            const cmsis_nn_dims *input_dims,
                                            const cmsis_nn_dims *filter_dims,
                                            const cmsis_nn_dims *output_dims,
                                            const int32_t lhs_offset,
                                            const int32_t *bias_data);

arm_cmsis_nn_status arm_convolve_wrapper_s16(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params *conv_params,
                                             const cmsis_nn_per_channel_quant_params *quant_params,
                                             const cmsis_nn_dims *input_dims,
                                             const int16_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const int8_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const cmsis_nn_bias_data *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             int16_t *output_data);

int32_t arm_convolve_wrapper_s16_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const cmsis_nn_dims *output_dims);

int32_t arm_convolve_wrapper_s16_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims);

int32_t arm_convolve_wrapper_s16_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_convolve_wrapper_s4(const cmsis_nn_context *ctx,
                                            const cmsis_nn_conv_params *conv_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int8_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            int8_t *output_data);

int32_t arm_convolve_wrapper_s4_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims);

int32_t arm_convolve_wrapper_s4_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims);

int32_t arm_convolve_wrapper_s4_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_convolve_wrapper_s8(const cmsis_nn_context *ctx,
                                            const cmsis_nn_context *weight_sum_ctx,
                                            const cmsis_nn_conv_params *conv_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int8_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            int8_t *output_data);

int32_t arm_convolve_wrapper_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims);

int32_t arm_convolve_wrapper_s8_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims);

int32_t arm_convolve_wrapper_s8_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_depth_to_space_s16(const int16_t *input_data,
                                           const cmsis_nn_dims *input_dims,
                                           const int32_t block_size,
                                           int16_t *output_data,
                                           const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_depth_to_space_s8(const int8_t *input_data,
                                          const cmsis_nn_dims *input_dims,
                                          const int32_t block_size,
                                          int8_t *output_data,
                                          const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_depthwise_conv_3x3_s8(const cmsis_nn_context *ctx,
                                              const cmsis_nn_dw_conv_params *dw_conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int8_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const int8_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int32_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              int8_t *output_data);

arm_cmsis_nn_status arm_depthwise_conv_fast_s16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_dw_conv_params *dw_conv_params,
                                                const cmsis_nn_per_channel_quant_params *quant_params,
                                                const cmsis_nn_dims *input_dims,
                                                const int16_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const int8_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const int64_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                int16_t *output_data);

int32_t arm_depthwise_conv_fast_s16_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_depthwise_conv_s16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params *dw_conv_params,
                                           const cmsis_nn_per_channel_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int16_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const int64_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           int16_t *output_data);

arm_cmsis_nn_status arm_depthwise_conv_s4(const cmsis_nn_context *ctx,
                                          const cmsis_nn_dw_conv_params *dw_conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *kernel,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output);

arm_cmsis_nn_status arm_depthwise_conv_s4_opt(const cmsis_nn_context *ctx,
                                              const cmsis_nn_dw_conv_params *dw_conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int8_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const int8_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int32_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              int8_t *output_data);

int32_t arm_depthwise_conv_s4_opt_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_depthwise_conv_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_dw_conv_params *dw_conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data);

arm_cmsis_nn_status arm_depthwise_conv_s8_opt(const cmsis_nn_context *ctx,
                                              const cmsis_nn_context *weight_sum_ctx,
                                              const cmsis_nn_dw_conv_params *dw_conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int8_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const int8_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int32_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              int8_t *output_data);

arm_cmsis_nn_status arm_depthwise_conv_s8_opt_3x3(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_context *weight_sum_ctx,
                                                  const cmsis_nn_dw_conv_params *dw_conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int8_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  int8_t *output_data);

arm_cmsis_nn_status arm_depthwise_conv_s8_opt_3x3_c64_s1(const cmsis_nn_context *ctx,
                                                         const cmsis_nn_context *weight_sum_ctx,
                                                         const cmsis_nn_dw_conv_params *dw_conv_params,
                                                         const cmsis_nn_per_channel_quant_params *quant_params,
                                                         const cmsis_nn_dims *input_dims,
                                                         const int8_t *input_data,
                                                         const cmsis_nn_dims *filter_dims,
                                                         const int8_t *filter_data,
                                                         const cmsis_nn_dims *bias_dims,
                                                         const int32_t *bias_data,
                                                         const cmsis_nn_dims *output_dims,
                                                         int8_t *output_data);

int32_t arm_depthwise_conv_s8_opt_3x3_get_buffer_size(const cmsis_nn_dims *input_dims);

arm_cmsis_nn_status arm_depthwise_conv_s8_opt_channelwise(const cmsis_nn_context *ctx,
                                                          const cmsis_nn_context *weight_sum_ctx,
                                                          const cmsis_nn_dw_conv_params *dw_conv_params,
                                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const int8_t *input_data,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const int8_t *filter_data,
                                                          const cmsis_nn_dims *bias_dims,
                                                          const int32_t *bias_data,
                                                          const cmsis_nn_dims *output_dims,
                                                          int8_t *output_data);

int32_t arm_depthwise_conv_s8_opt_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_depthwise_conv_s8_opt_planar(const cmsis_nn_context *ctx,
                                                     const cmsis_nn_context *weight_sum_ctx,
                                                     const cmsis_nn_dw_conv_params *dw_conv_params,
                                                     const cmsis_nn_per_channel_quant_params *quant_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const int8_t *input_data,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const int8_t *filter_data,
                                                     const cmsis_nn_dims *bias_dims,
                                                     const int32_t *bias_data,
                                                     const cmsis_nn_dims *output_dims,
                                                     int8_t *output_data);

int32_t arm_depthwise_conv_s8_opt_planar_supported(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_depthwise_conv_wrapper_s16(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_dw_conv_params *dw_conv_params,
                                                   const cmsis_nn_per_channel_quant_params *quant_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const int16_t *input_data,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const int8_t *filter_data,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const int64_t *bias_data,
                                                   const cmsis_nn_dims *output_dims,
                                                   int16_t *output_data);

int32_t arm_depthwise_conv_wrapper_s16_get_buffer_size(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const cmsis_nn_dims *output_dims);

int32_t arm_depthwise_conv_wrapper_s16_get_buffer_size_dsp(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const cmsis_nn_dims *output_dims);

int32_t arm_depthwise_conv_wrapper_s16_get_buffer_size_mve(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_depthwise_conv_wrapper_s4(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_dw_conv_params *dw_conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int8_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  int8_t *output_data);

int32_t arm_depthwise_conv_wrapper_s4_get_buffer_size(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const cmsis_nn_dims *output_dims);

int32_t arm_depthwise_conv_wrapper_s4_get_buffer_size_dsp(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims);

int32_t arm_depthwise_conv_wrapper_s4_get_buffer_size_mve(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_depthwise_conv_wrapper_s8(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_context *weight_sum_ctx,
                                                  const cmsis_nn_dw_conv_params *dw_conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int8_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  int8_t *output_data);

int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const cmsis_nn_dims *output_dims);

int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims);

int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_depthwise_convolve_weight_sum(int32_t *vector_sum_buf,
                                                      int8_t *scratch_buf,
                                                      const int8_t *rhs,
                                                      const cmsis_nn_dw_conv_params *dw_conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const cmsis_nn_dims *output_dims,
                                                      const int32_t lhs_offset,
                                                      const int32_t *bias_data);

arm_cmsis_nn_status
arm_dequantize_s16_f32(const int16_t *input, float *output, int32_t size, int32_t zero_point, float scale);

arm_cmsis_nn_status
arm_dequantize_s8_f32(const int8_t *input, float *output, int32_t size, int32_t zero_point, float scale);

arm_cmsis_nn_status arm_dynamic_update_slice_s16(const int16_t *operand,
                                                 const int16_t *update,
                                                 const int32_t *start_indices,
                                                 const cmsis_nn_dynamic_update_slice_params *params,
                                                 int16_t *output);

arm_cmsis_nn_status arm_dynamic_update_slice_s8(const int8_t *operand,
                                                const int8_t *update,
                                                const int32_t *start_indices,
                                                const cmsis_nn_dynamic_update_slice_params *params,
                                                int8_t *output);

arm_cmsis_nn_status arm_elementwise_add_s16(const int16_t *input_1_vect,
                                            const int16_t *input_2_vect,
                                            const int32_t input_1_offset,
                                            const int32_t input_1_mult,
                                            const int32_t input_1_shift,
                                            const int32_t input_2_offset,
                                            const int32_t input_2_mult,
                                            const int32_t input_2_shift,
                                            const int32_t left_shift,
                                            int16_t *output,
                                            const int32_t out_offset,
                                            const int32_t out_mult,
                                            const int32_t out_shift,
                                            const int32_t out_activation_min,
                                            const int32_t out_activation_max,
                                            const int32_t block_size);

arm_cmsis_nn_status arm_elementwise_add_s8(const int8_t *input_1_vect,
                                           const int8_t *input_2_vect,
                                           const int32_t input_1_offset,
                                           const int32_t input_1_mult,
                                           const int32_t input_1_shift,
                                           const int32_t input_2_offset,
                                           const int32_t input_2_mult,
                                           const int32_t input_2_shift,
                                           const int32_t left_shift,
                                           int8_t *output,
                                           const int32_t out_offset,
                                           const int32_t out_mult,
                                           const int32_t out_shift,
                                           const int32_t out_activation_min,
                                           const int32_t out_activation_max,
                                           const int32_t block_size);

arm_cmsis_nn_status arm_elementwise_mul_s16(const int16_t *input_1_vect,
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

arm_cmsis_nn_status arm_elementwise_mul_s8(const int8_t *input_1_vect,
                                           const int8_t *input_2_vect,
                                           const int32_t input_1_offset,
                                           const int32_t input_2_offset,
                                           int8_t *output,
                                           const int32_t out_offset,
                                           const int32_t out_mult,
                                           const int32_t out_shift,
                                           const int32_t out_activation_min,
                                           const int32_t out_activation_max,
                                           const int32_t block_size);

arm_cmsis_nn_status arm_elementwise_prelu_s16(const int16_t *input,
                                              const int16_t *alpha,
                                              const int32_t input_offset,
                                              const int32_t alpha_offset,
                                              const int32_t out_offset,
                                              const int32_t output_multiplier_identity,
                                              const int32_t output_shift_identity,
                                              const int32_t output_multiplier_alpha,
                                              const int32_t output_shift_alpha,
                                              int16_t *output,
                                              const int32_t block_size);

arm_cmsis_nn_status arm_elementwise_prelu_s8(const int8_t *input,
                                             const int8_t *alpha,
                                             const int32_t input_offset,
                                             const int32_t alpha_offset,
                                             const int32_t out_offset,
                                             const int32_t output_multiplier_identity,
                                             const int32_t output_shift_identity,
                                             const int32_t output_multiplier_alpha,
                                             const int32_t output_shift_alpha,
                                             int8_t *output,
                                             const int32_t block_size);

arm_cmsis_nn_status arm_elementwise_squared_difference_s16(const int16_t *input_1_vect,
                                                           const int16_t *input_2_vect,
                                                           const int32_t input_1_offset,
                                                           const int32_t input_1_mult,
                                                           const int32_t input_1_shift,
                                                           const int32_t input_2_offset,
                                                           const int32_t input_2_mult,
                                                           const int32_t input_2_shift,
                                                           const int32_t left_shift,
                                                           int16_t *output,
                                                           const int32_t out_offset,
                                                           const int32_t out_mult,
                                                           const int32_t out_shift,
                                                           const int32_t out_activation_min,
                                                           const int32_t out_activation_max,
                                                           const int32_t block_size);

arm_cmsis_nn_status arm_elementwise_squared_difference_s8(const int8_t *input_1_vect,
                                                          const int8_t *input_2_vect,
                                                          const int32_t input_1_offset,
                                                          const int32_t input_1_mult,
                                                          const int32_t input_1_shift,
                                                          const int32_t input_2_offset,
                                                          const int32_t input_2_mult,
                                                          const int32_t input_2_shift,
                                                          const int32_t left_shift,
                                                          int8_t *output,
                                                          const int32_t out_offset,
                                                          const int32_t out_mult,
                                                          const int32_t out_shift,
                                                          const int32_t out_activation_min,
                                                          const int32_t out_activation_max,
                                                          const int32_t block_size);

arm_cmsis_nn_status arm_elementwise_sub_s16(const int16_t *input_1_vect,
                                            const int16_t *input_2_vect,
                                            const int32_t input_1_offset,
                                            const int32_t input_1_mult,
                                            const int32_t input_1_shift,
                                            const int32_t input_2_offset,
                                            const int32_t input_2_mult,
                                            const int32_t input_2_shift,
                                            const int32_t left_shift,
                                            int16_t *output,
                                            const int32_t out_offset,
                                            const int32_t out_mult,
                                            const int32_t out_shift,
                                            const int32_t out_activation_min,
                                            const int32_t out_activation_max,
                                            const int32_t block_size);

arm_cmsis_nn_status arm_elementwise_sub_s8(const int8_t *input_1_vect,
                                           const int8_t *input_2_vect,
                                           const int32_t input_1_offset,
                                           const int32_t input_1_mult,
                                           const int32_t input_1_shift,
                                           const int32_t input_2_offset,
                                           const int32_t input_2_mult,
                                           const int32_t input_2_shift,
                                           const int32_t left_shift,
                                           int8_t *output,
                                           const int32_t out_offset,
                                           const int32_t out_mult,
                                           const int32_t out_shift,
                                           const int32_t out_activation_min,
                                           const int32_t out_activation_max,
                                           const int32_t block_size);

arm_cmsis_nn_status arm_equal_s16(const cmsis_nn_context *ctx,
                                  const int16_t *input_1_data,
                                  const cmsis_nn_dims *input_1_dims,
                                  const int16_t *input_2_data,
                                  const cmsis_nn_dims *input_2_dims,
                                  bool *output_data,
                                  const cmsis_nn_dims *output_dims,
                                  const int32_t input_1_offset,
                                  const int32_t input_1_mult,
                                  const int32_t input_1_shift,
                                  const int32_t input_2_offset,
                                  const int32_t input_2_mult,
                                  const int32_t input_2_shift,
                                  const int32_t left_shift);

arm_cmsis_nn_status arm_equal_s8(const cmsis_nn_context *ctx,
                                 const int8_t *input_1_data,
                                 const cmsis_nn_dims *input_1_dims,
                                 const int8_t *input_2_data,
                                 const cmsis_nn_dims *input_2_dims,
                                 bool *output_data,
                                 const cmsis_nn_dims *output_dims,
                                 const int32_t input_1_offset,
                                 const int32_t input_1_mult,
                                 const int32_t input_1_shift,
                                 const int32_t input_2_offset,
                                 const int32_t input_2_mult,
                                 const int32_t input_2_shift,
                                 const int32_t left_shift);

arm_cmsis_nn_status arm_fully_connected_per_channel_s16(const cmsis_nn_context *ctx,
                                                        const cmsis_nn_fc_params *fc_params,
                                                        const cmsis_nn_per_channel_quant_params *quant_params,
                                                        const cmsis_nn_dims *input_dims,
                                                        const int16_t *input_data,
                                                        const cmsis_nn_dims *filter_dims,
                                                        const int8_t *kernel,
                                                        const cmsis_nn_dims *bias_dims,
                                                        const int64_t *bias_data,
                                                        const cmsis_nn_dims *output_dims,
                                                        int16_t *output_data);

int32_t arm_fully_connected_per_channel_s16_get_buffer_size(const cmsis_nn_dims *filter_dims);

int32_t arm_fully_connected_per_channel_s16_get_buffer_size_dsp(const cmsis_nn_dims *filter_dims);

int32_t arm_fully_connected_per_channel_s16_get_buffer_size_mve(const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_fully_connected_per_channel_s8(const cmsis_nn_context *ctx,
                                                       const cmsis_nn_fc_params *fc_params,
                                                       const cmsis_nn_per_channel_quant_params *quant_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const int8_t *input_data,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const int8_t *filter_data,
                                                       const cmsis_nn_dims *bias_dims,
                                                       const int32_t *bias_data,
                                                       const cmsis_nn_dims *output_dims,
                                                       int8_t *output_data);

arm_cmsis_nn_status arm_fully_connected_s16(const cmsis_nn_context *ctx,
                                            const cmsis_nn_fc_params *fc_params,
                                            const cmsis_nn_per_tensor_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int16_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int64_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            int16_t *output_data);

int32_t arm_fully_connected_s16_get_buffer_size(const cmsis_nn_dims *filter_dims);

int32_t arm_fully_connected_s16_get_buffer_size_dsp(const cmsis_nn_dims *filter_dims);

int32_t arm_fully_connected_s16_get_buffer_size_mve(const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_fully_connected_s4(const cmsis_nn_context *ctx,
                                           const cmsis_nn_fc_params *fc_params,
                                           const cmsis_nn_per_tensor_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int8_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const int32_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           int8_t *output_data);

arm_cmsis_nn_status arm_fully_connected_s8(const cmsis_nn_context *ctx,
                                           const cmsis_nn_fc_params *fc_params,
                                           const cmsis_nn_per_tensor_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int8_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const int32_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           int8_t *output_data);

int32_t arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims *filter_dims);

int32_t arm_fully_connected_s8_get_buffer_size_dsp(const cmsis_nn_dims *filter_dims);

int32_t arm_fully_connected_s8_get_buffer_size_mve(const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_fully_connected_wrapper_s16(const cmsis_nn_context *ctx,
                                                    const cmsis_nn_fc_params *fc_params,
                                                    const cmsis_nn_quant_params *quant_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const int16_t *input_data,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const int8_t *filter_data,
                                                    const cmsis_nn_dims *bias_dims,
                                                    const int64_t *bias_data,
                                                    const cmsis_nn_dims *output_dims,
                                                    int16_t *output_data);

arm_cmsis_nn_status arm_fully_connected_wrapper_s8(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_fc_params *fc_params,
                                                   const cmsis_nn_quant_params *quant_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const int8_t *input_data,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const int8_t *filter_data,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const int32_t *bias_data,
                                                   const cmsis_nn_dims *output_dims,
                                                   int8_t *output_data);

arm_cmsis_nn_status arm_gather_nd_s16(const int16_t *params_data,
                                      const cmsis_nn_dims *params_dims,
                                      const int32_t *indices_data,
                                      const cmsis_nn_dims *indices_dims,
                                      const cmsis_nn_gather_nd_params *params,
                                      int16_t *output_data,
                                      const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_gather_nd_s8(const int8_t *params_data,
                                     const cmsis_nn_dims *params_dims,
                                     const int32_t *indices_data,
                                     const cmsis_nn_dims *indices_dims,
                                     const cmsis_nn_gather_nd_params *params,
                                     int8_t *output_data,
                                     const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_gather_s16(const int16_t *input_data,
                                   const cmsis_nn_dims *input_dims,
                                   const int32_t *indices_data,
                                   const cmsis_nn_dims *indices_dims,
                                   const cmsis_nn_gather_params *params,
                                   int16_t *output_data,
                                   const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_gather_s8(const int8_t *input_data,
                                  const cmsis_nn_dims *input_dims,
                                  const int32_t *indices_data,
                                  const cmsis_nn_dims *indices_dims,
                                  const cmsis_nn_gather_params *params,
                                  int8_t *output_data,
                                  const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_greater_equal_s16(const cmsis_nn_context *ctx,
                                          const int16_t *input_1_data,
                                          const cmsis_nn_dims *input_1_dims,
                                          const int16_t *input_2_data,
                                          const cmsis_nn_dims *input_2_dims,
                                          bool *output_data,
                                          const cmsis_nn_dims *output_dims,
                                          const int32_t input_1_offset,
                                          const int32_t input_1_mult,
                                          const int32_t input_1_shift,
                                          const int32_t input_2_offset,
                                          const int32_t input_2_mult,
                                          const int32_t input_2_shift,
                                          const int32_t left_shift);

arm_cmsis_nn_status arm_greater_equal_s8(const cmsis_nn_context *ctx,
                                         const int8_t *input_1_data,
                                         const cmsis_nn_dims *input_1_dims,
                                         const int8_t *input_2_data,
                                         const cmsis_nn_dims *input_2_dims,
                                         bool *output_data,
                                         const cmsis_nn_dims *output_dims,
                                         const int32_t input_1_offset,
                                         const int32_t input_1_mult,
                                         const int32_t input_1_shift,
                                         const int32_t input_2_offset,
                                         const int32_t input_2_mult,
                                         const int32_t input_2_shift,
                                         const int32_t left_shift);

arm_cmsis_nn_status arm_greater_s16(const cmsis_nn_context *ctx,
                                    const int16_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const int16_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    bool *output_data,
                                    const cmsis_nn_dims *output_dims,
                                    const int32_t input_1_offset,
                                    const int32_t input_1_mult,
                                    const int32_t input_1_shift,
                                    const int32_t input_2_offset,
                                    const int32_t input_2_mult,
                                    const int32_t input_2_shift,
                                    const int32_t left_shift);

arm_cmsis_nn_status arm_greater_s8(const cmsis_nn_context *ctx,
                                   const int8_t *input_1_data,
                                   const cmsis_nn_dims *input_1_dims,
                                   const int8_t *input_2_data,
                                   const cmsis_nn_dims *input_2_dims,
                                   bool *output_data,
                                   const cmsis_nn_dims *output_dims,
                                   const int32_t input_1_offset,
                                   const int32_t input_1_mult,
                                   const int32_t input_1_shift,
                                   const int32_t input_2_offset,
                                   const int32_t input_2_mult,
                                   const int32_t input_2_shift,
                                   const int32_t left_shift);

arm_cmsis_nn_status arm_hard_swish_compat_s8(const int8_t *input,
                                             const int32_t input_offset,
                                             const int32_t output_offset,
                                             const int32_t output_multiplier_fp,
                                             const int32_t output_multiplier_exp,
                                             const int32_t relu_multiplier_fp,
                                             const int32_t relu_multiplier_exp,
                                             int8_t *output,
                                             const int32_t output_size);

arm_cmsis_nn_status arm_hard_swish_precise_s16(const int16_t *input,
                                               const int32_t input_offset,
                                               const int32_t output_offset,
                                               const int32_t output_multiplier,
                                               const int32_t output_shift,
                                               const int32_t relu_q3,
                                               const int32_t relu_q6,
                                               const int32_t prescale,
                                               int16_t *output,
                                               const int32_t output_size);

arm_cmsis_nn_status arm_hard_swish_precise_s8(const int8_t *input,
                                              const int32_t input_offset,
                                              const int32_t output_offset,
                                              const int32_t output_multiplier,
                                              const int32_t output_shift,
                                              const int32_t relu_q3,
                                              const int32_t relu_q6,
                                              const int32_t prescale,
                                              int8_t *output,
                                              const int32_t output_size);

arm_cmsis_nn_status arm_leaky_relu_s16(const int16_t *input,
                                       const int32_t input_offset,
                                       const int32_t output_offset,
                                       const int32_t output_multiplier_alpha,
                                       const int32_t output_shift_alpha,
                                       const int32_t output_multiplier_identity,
                                       const int32_t output_shift_identity,
                                       int16_t *output,
                                       const int32_t output_size);

arm_cmsis_nn_status arm_leaky_relu_s8(const int8_t *input,
                                      const int32_t input_offset,
                                      const int32_t output_offset,
                                      const int32_t output_multiplier_alpha,
                                      const int32_t output_shift_alpha,
                                      const int32_t output_multiplier_identity,
                                      const int32_t output_shift_identity,
                                      int8_t *output,
                                      const int32_t output_size);

arm_cmsis_nn_status arm_less_equal_s16(const cmsis_nn_context *ctx,
                                       const int16_t *input_1_data,
                                       const cmsis_nn_dims *input_1_dims,
                                       const int16_t *input_2_data,
                                       const cmsis_nn_dims *input_2_dims,
                                       bool *output_data,
                                       const cmsis_nn_dims *output_dims,
                                       const int32_t input_1_offset,
                                       const int32_t input_1_mult,
                                       const int32_t input_1_shift,
                                       const int32_t input_2_offset,
                                       const int32_t input_2_mult,
                                       const int32_t input_2_shift,
                                       const int32_t left_shift);

arm_cmsis_nn_status arm_less_equal_s8(const cmsis_nn_context *ctx,
                                      const int8_t *input_1_data,
                                      const cmsis_nn_dims *input_1_dims,
                                      const int8_t *input_2_data,
                                      const cmsis_nn_dims *input_2_dims,
                                      bool *output_data,
                                      const cmsis_nn_dims *output_dims,
                                      const int32_t input_1_offset,
                                      const int32_t input_1_mult,
                                      const int32_t input_1_shift,
                                      const int32_t input_2_offset,
                                      const int32_t input_2_mult,
                                      const int32_t input_2_shift,
                                      const int32_t left_shift);

arm_cmsis_nn_status arm_less_s16(const cmsis_nn_context *ctx,
                                 const int16_t *input_1_data,
                                 const cmsis_nn_dims *input_1_dims,
                                 const int16_t *input_2_data,
                                 const cmsis_nn_dims *input_2_dims,
                                 bool *output_data,
                                 const cmsis_nn_dims *output_dims,
                                 const int32_t input_1_offset,
                                 const int32_t input_1_mult,
                                 const int32_t input_1_shift,
                                 const int32_t input_2_offset,
                                 const int32_t input_2_mult,
                                 const int32_t input_2_shift,
                                 const int32_t left_shift);

arm_cmsis_nn_status arm_less_s8(const cmsis_nn_context *ctx,
                                const int8_t *input_1_data,
                                const cmsis_nn_dims *input_1_dims,
                                const int8_t *input_2_data,
                                const cmsis_nn_dims *input_2_dims,
                                bool *output_data,
                                const cmsis_nn_dims *output_dims,
                                const int32_t input_1_offset,
                                const int32_t input_1_mult,
                                const int32_t input_1_shift,
                                const int32_t input_2_offset,
                                const int32_t input_2_mult,
                                const int32_t input_2_shift,
                                const int32_t left_shift);

arm_cmsis_nn_status arm_logistic_s16(const int16_t *input,
                                     int16_t *output,
                                     const int32_t input_size,
                                     int32_t input_multiplier,
                                     int32_t input_left_shift);

arm_cmsis_nn_status arm_lstm_unidirectional_s16(const int16_t *input,
                                                int16_t *output,
                                                const cmsis_nn_lstm_params *params,
                                                cmsis_nn_lstm_context *buffers);

int32_t arm_lstm_unidirectional_s16_temp1_get_buffer_size(const cmsis_nn_lstm_params *lstm_params);

int32_t arm_lstm_unidirectional_s16_temp2_get_buffer_size(const cmsis_nn_lstm_params *lstm_params);

arm_cmsis_nn_status arm_lstm_unidirectional_s8(const int8_t *input,
                                               int8_t *output,
                                               const cmsis_nn_lstm_params *params,
                                               cmsis_nn_lstm_context *buffers);

int32_t arm_lstm_unidirectional_s8_temp1_get_buffer_size(const cmsis_nn_lstm_params *lstm_params);

int32_t arm_lstm_unidirectional_s8_temp2_get_buffer_size(const cmsis_nn_lstm_params *lstm_params);

arm_cmsis_nn_status arm_max_pool_s16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const int16_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     int16_t *dst);

arm_cmsis_nn_status arm_max_pool_s8(const cmsis_nn_context *ctx,
                                    const cmsis_nn_pool_params *pool_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int8_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const cmsis_nn_dims *output_dims,
                                    int8_t *output_data);

arm_cmsis_nn_status arm_maximum_s16(const cmsis_nn_context *ctx,
                                    const int16_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const int16_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    int16_t *output_data,
                                    const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_maximum_s8(const cmsis_nn_context *ctx,
                                   const int8_t *input_1_data,
                                   const cmsis_nn_dims *input_1_dims,
                                   const int8_t *input_2_data,
                                   const cmsis_nn_dims *input_2_dims,
                                   int8_t *output_data,
                                   const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_mean_s16(const int16_t *input_data,
                                 const cmsis_nn_dims *input_dims,
                                 const int32_t input_offset,
                                 const cmsis_nn_dims *axis_dims,
                                 int16_t *output_data,
                                 const cmsis_nn_dims *output_dims,
                                 const int32_t out_offset,
                                 const int32_t out_mult,
                                 const int32_t out_shift);

arm_cmsis_nn_status arm_mean_s8(const int8_t *input_data,
                                const cmsis_nn_dims *input_dims,
                                const int32_t input_offset,
                                const cmsis_nn_dims *axis_dims,
                                int8_t *output_data,
                                const cmsis_nn_dims *output_dims,
                                const int32_t out_offset,
                                const int32_t out_mult,
                                const int32_t out_shift);

arm_cmsis_nn_status arm_minimum_s16(const cmsis_nn_context *ctx,
                                    const int16_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const int16_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    int16_t *output_data,
                                    const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_minimum_s8(const cmsis_nn_context *ctx,
                                   const int8_t *input_1_data,
                                   const cmsis_nn_dims *input_1_dims,
                                   const int8_t *input_2_data,
                                   const cmsis_nn_dims *input_2_dims,
                                   int8_t *output_data,
                                   const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_mirror_pad_s16(const int16_t *input, const cmsis_nn_mirror_pad_params *params, int16_t *output);

arm_cmsis_nn_status arm_mirror_pad_s8(const int8_t *input, const cmsis_nn_mirror_pad_params *params, int8_t *output);

arm_cmsis_nn_status arm_mul_s16(const int16_t *input1_data,
                                const cmsis_nn_dims *input1_dims,
                                const int16_t *input2_data,
                                const cmsis_nn_dims *input2_dims,
                                const int32_t input1_offset,
                                const int32_t input2_offset,
                                int16_t *output_data,
                                const cmsis_nn_dims *output_dims,
                                const int32_t out_offset,
                                const int32_t out_mult,
                                const int32_t out_shift,
                                const int32_t out_activation_min,
                                const int32_t out_activation_max);

arm_cmsis_nn_status arm_mul_s8(const int8_t *input1_data,
                               const cmsis_nn_dims *input1_dims,
                               const int8_t *input2_data,
                               const cmsis_nn_dims *input2_dims,
                               const int32_t input1_offset,
                               const int32_t input2_offset,
                               int8_t *output_data,
                               const cmsis_nn_dims *output_dims,
                               const int32_t out_offset,
                               const int32_t out_mult,
                               const int32_t out_shift,
                               const int32_t out_activation_min,
                               const int32_t out_activation_max);

arm_cmsis_nn_status arm_mul_scalar_s16(const int16_t *input_1_vect,
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

arm_cmsis_nn_status arm_mul_scalar_s8(const int8_t *input_1_vect,
                                      const int8_t *input_2_vect,
                                      const int32_t input_1_offset,
                                      const int32_t input_2_offset,
                                      int8_t *output,
                                      const int32_t out_offset,
                                      const int32_t out_mult,
                                      const int32_t out_shift,
                                      const int32_t out_activation_min,
                                      const int32_t out_activation_max,
                                      const int32_t block_size);

arm_cmsis_nn_status arm_nn_activation_s16(const int16_t *input,
                                          int16_t *output,
                                          const int32_t size,
                                          const int32_t left_shift,
                                          const arm_nn_activation_type type);

arm_cmsis_nn_status arm_not_equal_s16(const cmsis_nn_context *ctx,
                                      const int16_t *input_1_data,
                                      const cmsis_nn_dims *input_1_dims,
                                      const int16_t *input_2_data,
                                      const cmsis_nn_dims *input_2_dims,
                                      bool *output_data,
                                      const cmsis_nn_dims *output_dims,
                                      const int32_t input_1_offset,
                                      const int32_t input_1_mult,
                                      const int32_t input_1_shift,
                                      const int32_t input_2_offset,
                                      const int32_t input_2_mult,
                                      const int32_t input_2_shift,
                                      const int32_t left_shift);

arm_cmsis_nn_status arm_not_equal_s8(const cmsis_nn_context *ctx,
                                     const int8_t *input_1_data,
                                     const cmsis_nn_dims *input_1_dims,
                                     const int8_t *input_2_data,
                                     const cmsis_nn_dims *input_2_dims,
                                     bool *output_data,
                                     const cmsis_nn_dims *output_dims,
                                     const int32_t input_1_offset,
                                     const int32_t input_1_mult,
                                     const int32_t input_1_shift,
                                     const int32_t input_2_offset,
                                     const int32_t input_2_mult,
                                     const int32_t input_2_shift,
                                     const int32_t left_shift);

arm_cmsis_nn_status arm_pad_s16(const int16_t *input,
                                int16_t *output,
                                const int16_t pad_value,
                                const cmsis_nn_dims *input_size,
                                const cmsis_nn_dims *pre_pad,
                                const cmsis_nn_dims *post_pad);

arm_cmsis_nn_status arm_pad_s8(const int8_t *input,
                               int8_t *output,
                               const int8_t pad_value,
                               const cmsis_nn_dims *input_size,
                               const cmsis_nn_dims *pre_pad,
                               const cmsis_nn_dims *post_pad);

arm_cmsis_nn_status arm_prelu_s16(const cmsis_nn_dims *input_dims,
                                  const int16_t *input,
                                  const cmsis_nn_dims *alpha_dims,
                                  const int16_t *alpha,
                                  const int32_t input_offset,
                                  const int32_t alpha_offset,
                                  const int32_t output_offset,
                                  const int32_t output_multiplier_identity,
                                  const int32_t output_shift_identity,
                                  const int32_t output_multiplier_alpha,
                                  const int32_t output_shift_alpha,
                                  const cmsis_nn_dims *output_dims,
                                  int16_t *output);

arm_cmsis_nn_status arm_prelu_s8(const cmsis_nn_dims *input_dims,
                                 const int8_t *input,
                                 const cmsis_nn_dims *alpha_dims,
                                 const int8_t *alpha,
                                 const int32_t input_offset,
                                 const int32_t alpha_offset,
                                 const int32_t output_offset,
                                 const int32_t output_multiplier_identity,
                                 const int32_t output_shift_identity,
                                 const int32_t output_multiplier_alpha,
                                 const int32_t output_shift_alpha,
                                 const cmsis_nn_dims *output_dims,
                                 int8_t *output);

arm_cmsis_nn_status arm_prelu_scalar_s16(const int16_t *scalar_vect,
                                         const int16_t *non_scalar_vect,
                                         const bool scalar_is_input,
                                         const int32_t input_offset,
                                         const int32_t alpha_offset,
                                         const int32_t output_offset,
                                         const int32_t output_multiplier_identity,
                                         const int32_t output_shift_identity,
                                         const int32_t output_multiplier_alpha,
                                         const int32_t output_shift_alpha,
                                         int16_t *output,
                                         const int32_t block_size);

arm_cmsis_nn_status arm_prelu_scalar_s8(const int8_t *scalar_vect,
                                        const int8_t *non_scalar_vect,
                                        const bool scalar_is_input,
                                        const int32_t input_offset,
                                        const int32_t alpha_offset,
                                        const int32_t output_offset,
                                        const int32_t output_multiplier_identity,
                                        const int32_t output_shift_identity,
                                        const int32_t output_multiplier_alpha,
                                        const int32_t output_shift_alpha,
                                        int8_t *output,
                                        const int32_t block_size);

arm_cmsis_nn_status
arm_quantize_f32_s16(const float *input, int16_t *output, int32_t size, int32_t zero_point, float scale);

arm_cmsis_nn_status
arm_quantize_f32_s8(const float *input, int8_t *output, int32_t size, int32_t zero_point, float scale);

arm_cmsis_nn_status arm_reduce_max_s16(const int16_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *axis_dims,
                                       int16_t *output_data,
                                       const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_reduce_max_s8(const int8_t *input_data,
                                      const cmsis_nn_dims *input_dims,
                                      const cmsis_nn_dims *axis_dims,
                                      int8_t *output_data,
                                      const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_reduce_min_s16(const int16_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *axis_dims,
                                       int16_t *output_data,
                                       const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_reduce_min_s8(const int8_t *input_data,
                                      const cmsis_nn_dims *input_dims,
                                      const cmsis_nn_dims *axis_dims,
                                      int8_t *output_data,
                                      const cmsis_nn_dims *output_dims);

void arm_relu6_q7(int8_t *data, uint16_t size);

arm_cmsis_nn_status arm_relu_generic_s16(const int16_t *input,
                                         const int32_t input_offset,
                                         const int32_t output_offset,
                                         const int32_t output_multiplier,
                                         const int32_t output_shift,
                                         const int32_t act_min,
                                         const int32_t act_max,
                                         int16_t *output,
                                         const int32_t output_size);

arm_cmsis_nn_status arm_relu_generic_s8(const int8_t *input,
                                        const int32_t input_offset,
                                        const int32_t output_offset,
                                        const int32_t output_multiplier,
                                        const int32_t output_shift,
                                        const int32_t act_min,
                                        const int32_t act_max,
                                        int8_t *output,
                                        const int32_t output_size);

void arm_relu_q15(int16_t *data, uint16_t size);

void arm_relu_q7(int8_t *data, uint16_t size);

arm_cmsis_nn_status arm_relu_s16(const int16_t *input,
                                 const int32_t input_offset,
                                 const int32_t output_offset,
                                 const int32_t output_multiplier,
                                 const int32_t output_shift,
                                 int16_t *output,
                                 const int32_t output_size);

arm_cmsis_nn_status arm_relu_s8(const int8_t *input,
                                const int32_t input_offset,
                                const int32_t output_offset,
                                const int32_t output_multiplier,
                                const int32_t output_shift,
                                int8_t *output,
                                const int32_t output_size);

arm_cmsis_nn_status arm_requantize_s16_s16(const int16_t *input,
                                           int16_t *output,
                                           int32_t size,
                                           int32_t effective_scale_multiplier,
                                           int32_t effective_scale_shift,
                                           int32_t input_zeropoint,
                                           int32_t output_zeropoint);

arm_cmsis_nn_status arm_requantize_s8_s8(const int8_t *input,
                                         int8_t *output,
                                         int32_t size,
                                         int32_t effective_scale_multiplier,
                                         int32_t effective_scale_shift,
                                         int32_t input_zeropoint,
                                         int32_t output_zeropoint);

void arm_reshape_s8(const int8_t *input, int8_t *output, const uint32_t total_size);

arm_cmsis_nn_status arm_resize_nearest_neighbor_s16(const cmsis_nn_context *ctx,
                                                    const cmsis_nn_resize_params *resize_params,
                                                    const cmsis_nn_dims *input_shape,
                                                    const int16_t *input_data,
                                                    const cmsis_nn_dims *output_size_shape,
                                                    const int32_t *output_size_data,
                                                    const cmsis_nn_dims *output_shape,
                                                    int16_t *output_data);

arm_cmsis_nn_status arm_resize_nearest_neighbor_s8(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_resize_params *resize_params,
                                                   const cmsis_nn_dims *input_shape,
                                                   const int8_t *input_data,
                                                   const cmsis_nn_dims *output_size_shape,
                                                   const int32_t *output_size_data,
                                                   const cmsis_nn_dims *output_shape,
                                                   int8_t *output_data);

arm_cmsis_nn_status arm_reverse_sequence_s16(const int16_t *input,
                                             const int32_t *seq_lengths,
                                             const cmsis_nn_reverse_sequence_params *params,
                                             int16_t *output);

arm_cmsis_nn_status arm_reverse_sequence_s8(const int8_t *input,
                                            const int32_t *seq_lengths,
                                            const cmsis_nn_reverse_sequence_params *params,
                                            int8_t *output);

arm_cmsis_nn_status arm_rsqrt_s16_per_op(const int16_t *input,
                                         const int32_t input_offset,
                                         int16_t *output,
                                         const int32_t out_offset,
                                         const int32_t out_activation_min,
                                         const int32_t out_activation_max,
                                         const int32_t block_size,
                                         const int16_t *lut);

arm_cmsis_nn_status arm_rsqrt_s16_universal(const int16_t *input,
                                            const int32_t input_offset,
                                            int16_t *output,
                                            const int32_t out_offset,
                                            const int32_t out_mult,
                                            const int32_t out_shift,
                                            const bool needs_rescale,
                                            const int32_t out_activation_min,
                                            const int32_t out_activation_max,
                                            const int32_t block_size,
                                            const int32_t *lut);

arm_cmsis_nn_status arm_scatter_nd_s16(const int32_t *indices,
                                       const int16_t *updates,
                                       const cmsis_nn_scatter_nd_params *params,
                                       int16_t *output);

arm_cmsis_nn_status arm_scatter_nd_s8(const int32_t *indices,
                                      const int8_t *updates,
                                      const cmsis_nn_scatter_nd_params *params,
                                      int8_t *output);

arm_cmsis_nn_status arm_select_v2_s16(const bool *condition,
                                      const int16_t *x,
                                      const int16_t *y,
                                      const cmsis_nn_select_v2_params *params,
                                      int16_t *output);

arm_cmsis_nn_status arm_select_v2_s8(const bool *condition,
                                     const int8_t *x,
                                     const int8_t *y,
                                     const cmsis_nn_select_v2_params *params,
                                     int8_t *output);

arm_cmsis_nn_status arm_softmax_s16(const int16_t *input,
                                    const int32_t num_rows,
                                    const int32_t row_size,
                                    const int32_t mult,
                                    const int32_t shift,
                                    const cmsis_nn_softmax_lut_s16 *softmax_params,
                                    int16_t *output);

void arm_softmax_s8(const int8_t *input,
                    const int32_t num_rows,
                    const int32_t row_size,
                    const int32_t mult,
                    const int32_t shift,
                    const int32_t diff_min,
                    int8_t *output);

void arm_softmax_s8_s16(const int8_t *input,
                        const int32_t num_rows,
                        const int32_t row_size,
                        const int32_t mult,
                        const int32_t shift,
                        const int32_t diff_min,
                        int16_t *output);

void arm_softmax_u8(const uint8_t *input,
                    const int32_t num_rows,
                    const int32_t row_size,
                    const int32_t mult,
                    const int32_t shift,
                    const int32_t diff_min,
                    uint8_t *output);

arm_cmsis_nn_status arm_space_to_batch_nd_s16(const int16_t *input_data,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_tile *block_shape,
                                              const cmsis_nn_dims *pad, // n->top, h->left, w->bottom, c->right
                                              int16_t *output_data,
                                              const cmsis_nn_dims *output_dims,
                                              const int32_t output_offset);

arm_cmsis_nn_status arm_space_to_batch_nd_s8(const int8_t *input_data,
                                             const cmsis_nn_dims *input_dims,
                                             const cmsis_nn_tile *block_shape,
                                             const cmsis_nn_dims *pad, // n->top, h->left, w->bottom, c->right
                                             int8_t *output_data,
                                             const cmsis_nn_dims *output_dims,
                                             const int32_t output_offset);

arm_cmsis_nn_status arm_space_to_depth_s16(const int16_t *input_data,
                                           const cmsis_nn_dims *input_dims,
                                           const int32_t block_size,
                                           int16_t *output_data,
                                           const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_space_to_depth_s8(const int8_t *input_data,
                                          const cmsis_nn_dims *input_dims,
                                          const int32_t block_size,
                                          int8_t *output_data,
                                          const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_split_s16(const int16_t *input_data,
                                  const int32_t input_dims,
                                  const int32_t *input_shape,
                                  const int32_t axis,
                                  const int32_t num_splits,
                                  const int32_t *split_dims,
                                  int16_t *const *output_data);

arm_cmsis_nn_status arm_split_s8(const int8_t *input_data,
                                 const int32_t input_dims,
                                 const int32_t *input_shape,
                                 const int32_t axis,
                                 const int32_t num_splits,
                                 const int32_t *split_dims,
                                 int8_t *const *output_data);

arm_cmsis_nn_status
arm_sqrt_s16(const int16_t *input, const cmsis_nn_dims *input_dims, int16_t *output, const int16_t *sqrt_lut);

arm_cmsis_nn_status
arm_sqrt_s16_tablefree(const int16_t *input, const cmsis_nn_dims *input_dims, int16_t *output, const float scale);

arm_cmsis_nn_status
arm_sqrt_s8(const int8_t *input, const cmsis_nn_dims *input_dims, int8_t *output, const int8_t *sqrt_lut);

arm_cmsis_nn_status arm_squared_difference_s16(const int16_t *input1_data,
                                               const cmsis_nn_dims *input1_dims,
                                               const int16_t *input2_data,
                                               const cmsis_nn_dims *input2_dims,
                                               const int32_t input1_offset,
                                               const int32_t input1_mult,
                                               const int32_t input1_shift,
                                               const int32_t input2_offset,
                                               const int32_t input2_mult,
                                               const int32_t input2_shift,
                                               const int32_t left_shift,
                                               int16_t *output_data,
                                               const cmsis_nn_dims *output_dims,
                                               const int32_t out_offset,
                                               const int32_t out_mult,
                                               const int32_t out_shift,
                                               const int32_t out_activation_min,
                                               const int32_t out_activation_max);

arm_cmsis_nn_status arm_squared_difference_s8(const int8_t *input1_data,
                                              const cmsis_nn_dims *input1_dims,
                                              const int8_t *input2_data,
                                              const cmsis_nn_dims *input2_dims,
                                              const int32_t input1_offset,
                                              const int32_t input1_mult,
                                              const int32_t input1_shift,
                                              const int32_t input2_offset,
                                              const int32_t input2_mult,
                                              const int32_t input2_shift,
                                              const int32_t left_shift,
                                              int8_t *output_data,
                                              const cmsis_nn_dims *output_dims,
                                              const int32_t out_offset,
                                              const int32_t out_mult,
                                              const int32_t out_shift,
                                              const int32_t out_activation_min,
                                              const int32_t out_activation_max);

arm_cmsis_nn_status arm_squared_difference_scalar_s16(const int16_t *input_1_vect,
                                                      const int16_t *input_2_vect,
                                                      const int32_t input_1_offset,
                                                      const int32_t input_1_mult,
                                                      const int32_t input_1_shift,
                                                      const int32_t input_2_offset,
                                                      const int32_t input_2_mult,
                                                      const int32_t input_2_shift,
                                                      const int32_t left_shift,
                                                      int16_t *output,
                                                      const int32_t out_offset,
                                                      const int32_t out_mult,
                                                      const int32_t out_shift,
                                                      const int32_t out_activation_min,
                                                      const int32_t out_activation_max,
                                                      const int32_t block_size);

arm_cmsis_nn_status arm_squared_difference_scalar_s8(const int8_t *input_1_vect,
                                                     const int8_t *input_2_vect,
                                                     const int32_t input_1_offset,
                                                     const int32_t input_1_mult,
                                                     const int32_t input_1_shift,
                                                     const int32_t input_2_offset,
                                                     const int32_t input_2_mult,
                                                     const int32_t input_2_shift,
                                                     const int32_t left_shift,
                                                     int8_t *output,
                                                     const int32_t out_offset,
                                                     const int32_t out_mult,
                                                     const int32_t out_shift,
                                                     const int32_t out_activation_min,
                                                     const int32_t out_activation_max,
                                                     const int32_t block_size);

arm_cmsis_nn_status arm_strided_slice_s16(const int16_t *input_data,
                                          int16_t *output_data,
                                          const cmsis_nn_dims *const input_dims,
                                          const cmsis_nn_dims *const begin_dims,
                                          const cmsis_nn_dims *const stride_dims,
                                          const cmsis_nn_dims *const output_dims);

arm_cmsis_nn_status arm_strided_slice_s32(const int32_t *input_data,
                                          int32_t *output_data,
                                          const cmsis_nn_dims *const input_dims,
                                          const cmsis_nn_dims *const begin_dims,
                                          const cmsis_nn_dims *const stride_dims,
                                          const cmsis_nn_dims *const output_dims);

arm_cmsis_nn_status arm_strided_slice_s8(const int8_t *input_data,
                                         int8_t *output_data,
                                         const cmsis_nn_dims *const input_dims,
                                         const cmsis_nn_dims *const begin_dims,
                                         const cmsis_nn_dims *const stride_dims,
                                         const cmsis_nn_dims *const output_dims);

arm_cmsis_nn_status arm_sub_s16(const int16_t *input1_data,
                                const cmsis_nn_dims *input1_dims,
                                const int16_t *input2_data,
                                const cmsis_nn_dims *input2_dims,
                                const int32_t input1_offset,
                                const int32_t input1_mult,
                                const int32_t input1_shift,
                                const int32_t input2_offset,
                                const int32_t input2_mult,
                                const int32_t input2_shift,
                                const int32_t left_shift,
                                int16_t *output_data,
                                const cmsis_nn_dims *output_dims,
                                const int32_t out_offset,
                                const int32_t out_mult,
                                const int32_t out_shift,
                                const int32_t out_activation_min,
                                const int32_t out_activation_max);

arm_cmsis_nn_status arm_sub_s8(const int8_t *input1_data,
                               const cmsis_nn_dims *input1_dims,
                               const int8_t *input2_data,
                               const cmsis_nn_dims *input2_dims,
                               const int32_t input1_offset,
                               const int32_t input1_mult,
                               const int32_t input1_shift,
                               const int32_t input2_offset,
                               const int32_t input2_mult,
                               const int32_t input2_shift,
                               const int32_t left_shift,
                               int8_t *output_data,
                               const cmsis_nn_dims *output_dims,
                               const int32_t out_offset,
                               const int32_t out_mult,
                               const int32_t out_shift,
                               const int32_t out_activation_min,
                               const int32_t out_activation_max);

arm_cmsis_nn_status arm_sub_scalar_s16(const int16_t *input_1_vect,
                                       const int16_t *input_2_vect,
                                       const int32_t input_1_offset,
                                       const int32_t input_1_mult,
                                       const int32_t input_1_shift,
                                       const int32_t input_2_offset,
                                       const int32_t input_2_mult,
                                       const int32_t input_2_shift,
                                       const int32_t left_shift,
                                       int16_t *output,
                                       const int32_t out_offset,
                                       const int32_t out_mult,
                                       const int32_t out_shift,
                                       const int32_t out_activation_min,
                                       const int32_t out_activation_max,
                                       const int32_t block_size);

arm_cmsis_nn_status arm_sub_scalar_s8(const int8_t *input_1_vect,
                                      const int8_t *input_2_vect,
                                      const int32_t input_1_offset,
                                      const int32_t input_1_mult,
                                      const int32_t input_1_shift,
                                      const int32_t input_2_offset,
                                      const int32_t input_2_mult,
                                      const int32_t input_2_shift,
                                      const int32_t left_shift,
                                      int8_t *output,
                                      const int32_t out_offset,
                                      const int32_t out_mult,
                                      const int32_t out_shift,
                                      const int32_t out_activation_min,
                                      const int32_t out_activation_max,
                                      const int32_t block_size);

arm_cmsis_nn_status arm_svdf_s8(const cmsis_nn_context *ctx,
                                const cmsis_nn_context *input_ctx,
                                const cmsis_nn_context *output_ctx,
                                const cmsis_nn_svdf_params *svdf_params,
                                const cmsis_nn_per_tensor_quant_params *input_quant_params,
                                const cmsis_nn_per_tensor_quant_params *output_quant_params,
                                const cmsis_nn_dims *input_dims,
                                const int8_t *input_data,
                                const cmsis_nn_dims *state_dims,
                                int8_t *state_data,
                                const cmsis_nn_dims *weights_feature_dims,
                                const int8_t *weights_feature_data,
                                const cmsis_nn_dims *weights_time_dims,
                                const int8_t *weights_time_data,
                                const cmsis_nn_dims *bias_dims,
                                const int32_t *bias_data,
                                const cmsis_nn_dims *output_dims,
                                int8_t *output_data);

int32_t arm_svdf_s8_get_buffer_size(const cmsis_nn_dims *weights_feature_dims);

int32_t arm_svdf_s8_get_buffer_size_dsp(const cmsis_nn_dims *weights_feature_dims);

int32_t arm_svdf_s8_get_buffer_size_mve(const cmsis_nn_dims *weights_feature_dims);

int32_t arm_svdf_s8_input_ctx_get_buffer_size(const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *weights_feature_dims);

int32_t arm_svdf_s8_output_ctx_get_buffer_size(const cmsis_nn_svdf_params *svdf_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *weights_feature_dims);

arm_cmsis_nn_status arm_svdf_state_s16_s8(const cmsis_nn_context *input_ctx,
                                          const cmsis_nn_context *output_ctx,
                                          const cmsis_nn_svdf_params *svdf_params,
                                          const cmsis_nn_per_tensor_quant_params *input_quant_params,
                                          const cmsis_nn_per_tensor_quant_params *output_quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *state_dims,
                                          int16_t *state_data,
                                          const cmsis_nn_dims *weights_feature_dims,
                                          const int8_t *weights_feature_data,
                                          const cmsis_nn_dims *weights_time_dims,
                                          const int16_t *weights_time_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data);

int32_t arm_svdf_state_s16_s8_input_ctx_get_buffer_size(const cmsis_nn_dims *input_dims,
                                                        const cmsis_nn_dims *weights_feature_dims);

int32_t arm_svdf_state_s16_s8_output_ctx_get_buffer_size(const cmsis_nn_svdf_params *svdf_params,
                                                         const cmsis_nn_dims *input_dims,
                                                         const cmsis_nn_dims *weights_feature_dims);

arm_cmsis_nn_status arm_tanh_s16(const int16_t *input,
                                 int16_t *output,
                                 const int32_t input_size,
                                 int32_t input_multiplier,
                                 int32_t input_left_shift);

arm_cmsis_nn_status arm_tile_s16(const int16_t *input, const cmsis_nn_tile_params *params, int16_t *output);

arm_cmsis_nn_status arm_tile_s8(const int8_t *input, const cmsis_nn_tile_params *params, int8_t *output);

arm_cmsis_nn_status arm_transpose_conv_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_context *output_ctx,
                                          const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data);

int32_t arm_transpose_conv_s8_get_buffer_size(const cmsis_nn_transpose_conv_params *transposed_conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *out_dims);

int32_t arm_transpose_conv_s8_get_buffer_size_mve(const cmsis_nn_transpose_conv_params *transposed_conv_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const cmsis_nn_dims *out_dims);

int32_t arm_transpose_conv_s8_get_reverse_conv_buffer_size(const cmsis_nn_transpose_conv_params *transposed_conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_transpose_conv_wrapper_s8(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_context *weight_sum_ctx,
                                                  const cmsis_nn_context *reverse_conv_ctx,
                                                  const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int8_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  int8_t *output_data);

arm_cmsis_nn_status arm_transpose_s16(const int16_t *input_data,
                                      int16_t *const output_data,
                                      const cmsis_nn_dims *const input_dims,
                                      const cmsis_nn_dims *const output_dims,
                                      const cmsis_nn_transpose_params *const transpose_params);

arm_cmsis_nn_status arm_transpose_s8(const int8_t *input_data,
                                     int8_t *const output_data,
                                     const cmsis_nn_dims *const input_dims,
                                     const cmsis_nn_dims *const output_dims,
                                     const cmsis_nn_transpose_params *const transpose_params);

arm_cmsis_nn_status
arm_where_s16(const int16_t *condition, const cmsis_nn_where_params *params, int64_t *output, int32_t *num_true);

arm_cmsis_nn_status
arm_where_s8(const int8_t *condition, const cmsis_nn_where_params *params, int64_t *output, int32_t *num_true);

