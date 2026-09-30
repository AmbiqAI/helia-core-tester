/* Verbatim declarations of every function of the contract-bound operators (arm_convolve_*, arm_depthwise_*, arm_fully_connected_*, arm_batch_matmul_*, arm_avgpool_*, arm_avg_pool_*, arm_max_pool_*, arm_relu*, arm_clamp_*, arm_hard_swish_*, arm_leaky_relu_*, arm_logistic_*, arm_tanh_*, arm_nn_activation_*, arm_prelu_*, arm_abs_*, arm_nn_abs_*, arm_mean_*, arm_nn_mean_*, arm_reduce_*, arm_add_*, arm_sub_*, arm_mul_*, arm_elementwise_*, arm_squared_difference_*, arm_maximum_*, arm_minimum_*, arm_argmax_*, arm_argmin_*, arm_nn_fill_*, arm_sqrt_*, arm_equal_*, arm_not_equal_*, arm_greater_*, arm_less_*, arm_comparison_*, arm_broadcast_to_*, arm_batch_to_space_*, arm_space_to_batch_*, arm_depth_to_space_*, arm_space_to_depth_*, arm_strided_slice_*, arm_pad_*, arm_transpose_*, arm_gather_*, arm_resize_nearest_neighbor_*, arm_pack_*, arm_mirror_pad_*, arm_tile_*, arm_reverse_sequence_*, arm_select_v2_*, arm_scatter_nd_*, arm_dynamic_update_slice_*, arm_where_*, arm_requantize_*, arm_batch_norm_*, arm_softmax_*, arm_split_*, arm_unpack_*, arm_quantize_*, arm_dequantize_*, arm_concatenation_*)
 * from ns-cmsis-nn arm_nnfunctions_flt.h; the fallback contract for unit tests without a checkout.
 * test_bound_operator_fixture_matches_the_real_tree keeps it equal to the export. */

#if ARM_NN_ENABLE_F16
arm_cmsis_nn_status
arm_argmax_f16(const float16_t *input_data, const cmsis_nn_dims *input_dims, int32_t axis, int32_t *output_data);

arm_cmsis_nn_status
arm_argmin_f16(const float16_t *input_data, const cmsis_nn_dims *input_dims, int32_t axis, int32_t *output_data);

arm_cmsis_nn_status arm_avg_pool_f16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params_f16 *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float16_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     float16_t *dst);

arm_cmsis_nn_status arm_batch_matmul_f16(const cmsis_nn_context *ctx,
                                         const cmsis_nn_bmm_params_f16 *bmm_params,
                                         const cmsis_nn_dims *input_lhs_dims,
                                         const float16_t *input_lhs,
                                         const cmsis_nn_dims *input_rhs_dims,
                                         const float16_t *input_rhs,
                                         const cmsis_nn_dims *output_dims,
                                         float16_t *output);

int32_t arm_batch_matmul_f16_get_buffer_size(const cmsis_nn_bmm_params_f16 *bmm_params,
                                             const cmsis_nn_dims *input_lhs_dims,
                                             const cmsis_nn_dims *input_rhs_dims,
                                             const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_batch_norm_f16(const float16_t *input,
                                       float16_t *output,
                                       const float16_t *scale,
                                       const float16_t *bias,
                                       const cmsis_nn_dims *input_dims,
                                       arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_concatenation_f16(const float16_t *const *input_data,
                                          int32_t num_inputs,
                                          const int32_t *axis_sizes,
                                          int32_t output_dims,
                                          const int32_t *output_shape,
                                          int32_t axis,
                                          float16_t *output_data);

void arm_concatenation_f16_w(const float16_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float16_t *output,
                             uint32_t offset_w);

void arm_concatenation_f16_x(const float16_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float16_t *output,
                             int32_t output_x,
                             uint32_t offset_x);

void arm_concatenation_f16_y(const float16_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float16_t *output,
                             int32_t output_y,
                             uint32_t offset_y);

void arm_concatenation_f16_z(const float16_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float16_t *output,
                             int32_t output_z,
                             uint32_t offset_z);

arm_cmsis_nn_status arm_convolve_1_x_n_f16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_conv_params_f16 *conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float16_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const float16_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const float16_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           float16_t *output_data,
                                           arm_nn_tensor_layout layout);

int32_t arm_convolve_1_x_n_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_f16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_conv_params_f16 *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float16_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const float16_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const float16_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                float16_t *output_data);

arm_cmsis_nn_status arm_convolve_1x1_f16(const cmsis_nn_context *ctx,
                                         const cmsis_nn_conv_params_f16 *conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const float16_t *input_data,
                                         const cmsis_nn_dims *filter_dims,
                                         const float16_t *filter_data,
                                         const cmsis_nn_dims *bias_dims,
                                         const float16_t *bias_data,
                                         const cmsis_nn_dims *output_dims,
                                         float16_t *output_data,
                                         arm_nn_tensor_layout layout);

int32_t arm_convolve_1x1_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const cmsis_nn_dims *filter_dims,
                                             const cmsis_nn_dims *output_dims,
                                             arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_convolve_1x1_nhwc_f16(const cmsis_nn_context *ctx,
                                              const cmsis_nn_conv_params_f16 *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const float16_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const float16_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const float16_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              float16_t *output_data);

arm_cmsis_nn_status arm_convolve_f16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params_f16 *conv_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float16_t *input_data,
                                     const cmsis_nn_dims *filter_dims,
                                     const float16_t *filter_data,
                                     const cmsis_nn_dims *bias_dims,
                                     const float16_t *bias_data,
                                     const cmsis_nn_dims *output_dims,
                                     float16_t *output_data,
                                     arm_nn_tensor_layout layout);

int32_t arm_convolve_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const cmsis_nn_dims *filter_dims,
                                         const cmsis_nn_dims *output_dims,
                                         arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_convolve_nhwc_f16(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params_f16 *conv_params,
                                          const cmsis_nn_dims *input_dims,
                                          const float16_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const float16_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const float16_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          float16_t *output_data);

arm_cmsis_nn_status arm_convolve_wrapper_f16(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params_f16 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const float16_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const float16_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const float16_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             float16_t *output_data);

int32_t arm_convolve_wrapper_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_depthwise_conv_f16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float16_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const float16_t *kernel,
                                           const cmsis_nn_dims *bias_dims,
                                           const float16_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           float16_t *output,
                                           arm_nn_tensor_layout layout);

int32_t arm_depthwise_conv_f16_get_buffer_size(const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_depthwise_conv_wrapper_f16(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const float16_t *input,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const float16_t *kernel,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const float16_t *bias,
                                                   const cmsis_nn_dims *output_dims,
                                                   float16_t *output);

int32_t arm_depthwise_conv_wrapper_f16_get_buffer_size(const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_depthwise_nhwc_conv_f16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float16_t *input,
                                                const cmsis_nn_dims *filter_dims,
                                                const float16_t *kernel,
                                                const cmsis_nn_dims *bias_dims,
                                                const float16_t *bias,
                                                const cmsis_nn_dims *output_dims,
                                                float16_t *output);

arm_cmsis_nn_status arm_dequantize_f16_f32(const float16_t *input, float32_t *output, int32_t block_size);

arm_cmsis_nn_status arm_elementwise_add_broadcast_f16(const float16_t *input_1_data,
                                                      const cmsis_nn_dims *input_1_dims,
                                                      const float16_t *input_2_data,
                                                      const cmsis_nn_dims *input_2_dims,
                                                      float16_t *output_data,
                                                      const cmsis_nn_dims *output_dims,
                                                      float16_t out_activation_min,
                                                      float16_t out_activation_max);

arm_cmsis_nn_status arm_elementwise_add_f16(const float16_t *input_1_vect,
                                            const float16_t *input_2_vect,
                                            float16_t *output,
                                            float16_t out_activation_min,
                                            float16_t out_activation_max,
                                            int32_t block_size);

arm_cmsis_nn_status arm_elementwise_add_fp16(const float16_t *input_1_vect,
                                             const float16_t *input_2_vect,
                                             float16_t *output,
                                             const float16_t out_activation_min,
                                             const float16_t out_activation_max,
                                             const int32_t block_size);

arm_cmsis_nn_status arm_elementwise_mul_broadcast_f16(const float16_t *input_1_data,
                                                      const cmsis_nn_dims *input_1_dims,
                                                      const float16_t *input_2_data,
                                                      const cmsis_nn_dims *input_2_dims,
                                                      float16_t *output_data,
                                                      const cmsis_nn_dims *output_dims,
                                                      float16_t out_activation_min,
                                                      float16_t out_activation_max);

arm_cmsis_nn_status arm_elementwise_mul_f16(const float16_t *input_1_vect,
                                            const float16_t *input_2_vect,
                                            float16_t *output,
                                            float16_t out_activation_min,
                                            float16_t out_activation_max,
                                            int32_t block_size);

arm_cmsis_nn_status arm_elementwise_squared_difference_f16(const float16_t *input_1_vect,
                                                           const float16_t *input_2_vect,
                                                           float16_t *output,
                                                           int32_t block_size);

arm_cmsis_nn_status arm_elementwise_sub_broadcast_f16(const float16_t *input_1_data,
                                                      const cmsis_nn_dims *input_1_dims,
                                                      const float16_t *input_2_data,
                                                      const cmsis_nn_dims *input_2_dims,
                                                      float16_t *output_data,
                                                      const cmsis_nn_dims *output_dims,
                                                      float16_t out_activation_min,
                                                      float16_t out_activation_max);

arm_cmsis_nn_status arm_elementwise_sub_f16(const float16_t *input_1_vect,
                                            const float16_t *input_2_vect,
                                            float16_t *output,
                                            float16_t out_activation_min,
                                            float16_t out_activation_max,
                                            int32_t block_size);

arm_cmsis_nn_status arm_fully_connected_f16(const cmsis_nn_context *ctx,
                                            const cmsis_nn_fc_params_f16 *fc_params,
                                            const cmsis_nn_dims *input_dims,
                                            const float16_t *input,
                                            const cmsis_nn_dims *filter_dims,
                                            const float16_t *kernel,
                                            const cmsis_nn_dims *bias_dims,
                                            const float16_t *bias,
                                            const cmsis_nn_dims *output_dims,
                                            float16_t *output,
                                            arm_nn_tensor_layout layout);

int32_t arm_fully_connected_f16_get_buffer_size(const cmsis_nn_fc_params_f16 *fc_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims,
                                                arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_fully_connected_nhwc_f16(const cmsis_nn_context *ctx,
                                                 const cmsis_nn_fc_params_f16 *fc_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const float16_t *input,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const float16_t *kernel,
                                                 const cmsis_nn_dims *bias_dims,
                                                 const float16_t *bias,
                                                 const cmsis_nn_dims *output_dims,
                                                 float16_t *output);

arm_cmsis_nn_status arm_gather_f16(const float16_t *input_data,
                                   const cmsis_nn_dims *input_dims,
                                   const int32_t *indices_data,
                                   const cmsis_nn_dims *indices_dims,
                                   const cmsis_nn_gather_params *params,
                                   float16_t *output_data,
                                   const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_gather_nd_f16(const float16_t *params_data,
                                      const cmsis_nn_dims *params_dims,
                                      const int32_t *indices_data,
                                      const cmsis_nn_dims *indices_dims,
                                      const cmsis_nn_gather_nd_params *params,
                                      float16_t *output_data,
                                      const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_hard_swish_f16(const float16_t *input, float16_t *output, int32_t size);

arm_cmsis_nn_status arm_max_pool_f16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params_f16 *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float16_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     float16_t *dst);

arm_cmsis_nn_status arm_maximum_f16(const cmsis_nn_context *ctx,
                                    const float16_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const float16_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    float16_t *output_data,
                                    const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_minimum_f16(const cmsis_nn_context *ctx,
                                    const float16_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const float16_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    float16_t *output_data,
                                    const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_nn_abs_f16(const float16_t *input, float16_t *output, int32_t block_size);

arm_cmsis_nn_status arm_nn_activation_f16(const float16_t *input,
                                          float16_t *output,
                                          int32_t size,
                                          arm_nn_activation_type_flt type,
                                          float16_t act_param);

arm_cmsis_nn_status arm_nn_fill_f16(float16_t value, float16_t *output, int32_t block_size);

arm_cmsis_nn_status arm_nn_mean_f16(const float16_t *input_data,
                                    const cmsis_nn_dims *input_dims,
                                    const cmsis_nn_dims *axis_dims,
                                    float16_t *output_data,
                                    const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_pack_f16(const float16_t *const *input_data,
                                 int32_t num_inputs,
                                 int32_t input_dims,
                                 const int32_t *input_shape,
                                 int32_t axis,
                                 float16_t *output_data);

arm_cmsis_nn_status arm_pad_f16(const float16_t *input,
                                float16_t *output,
                                float16_t pad_value,
                                const cmsis_nn_dims *input_size,
                                const cmsis_nn_dims *pre_pad,
                                const cmsis_nn_dims *post_pad);

arm_cmsis_nn_status arm_prelu_f16(const cmsis_nn_dims *input_dims,
                                  const float16_t *input,
                                  const cmsis_nn_dims *alpha_dims,
                                  const float16_t *alpha,
                                  const cmsis_nn_dims *output_dims,
                                  float16_t *output);

arm_cmsis_nn_status arm_reduce_max_f16(const float16_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *axis_dims,
                                       float16_t *output_data,
                                       const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_reduce_min_f16(const float16_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *axis_dims,
                                       float16_t *output_data,
                                       const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_reduce_sum_f16(const float16_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *axis_dims,
                                       float16_t *output_data,
                                       const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_resize_nearest_neighbor_f16(const cmsis_nn_context *ctx,
                                                    const cmsis_nn_resize_params *resize_params,
                                                    const cmsis_nn_dims *input_shape,
                                                    const float16_t *input_data,
                                                    const cmsis_nn_dims *output_size_shape,
                                                    const int32_t *output_size_data,
                                                    const cmsis_nn_dims *output_shape,
                                                    float16_t *output_data);

int32_t arm_resize_nearest_neighbor_f16_get_buffer_size(const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_softmax_f16(const float16_t *input, int32_t num_rows, int32_t row_size, float16_t *output);

arm_cmsis_nn_status arm_split_f16(const float16_t *input_data,
                                  const int32_t input_dims,
                                  const int32_t *input_shape,
                                  const int32_t axis,
                                  const int32_t num_splits,
                                  const int32_t *split_dims,
                                  float16_t *const *output_data);

arm_cmsis_nn_status arm_strided_slice_f16(const float16_t *input_data,
                                          float16_t *output_data,
                                          const cmsis_nn_dims *const input_dims,
                                          const cmsis_nn_dims *const begin_dims,
                                          const cmsis_nn_dims *const stride_dims,
                                          const cmsis_nn_dims *const output_dims);

arm_cmsis_nn_status arm_transpose_conv_f16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_context *output_ctx,
                                           const cmsis_nn_transpose_conv_params_f16 *transpose_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float16_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const float16_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const float16_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           float16_t *output_data,
                                           arm_nn_tensor_layout layout);

int32_t arm_transpose_conv_f16_get_buffer_size(const cmsis_nn_transpose_conv_params_f16 *transpose_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *out_dims);

int32_t
arm_transpose_conv_f16_get_reverse_conv_buffer_size(const cmsis_nn_transpose_conv_params_f16 *transpose_conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_transpose_conv_nhwc_f16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_context *output_ctx,
                                                const cmsis_nn_transpose_conv_params_f16 *transpose_conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float16_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const float16_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const float16_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                float16_t *output_data);

arm_cmsis_nn_status arm_transpose_conv_wrapper_f16(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_context *output_ctx,
                                                   const cmsis_nn_transpose_conv_params_f16 *transpose_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const float16_t *input_data,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const float16_t *filter_data,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const float16_t *bias_data,
                                                   const cmsis_nn_dims *output_dims,
                                                   float16_t *output_data,
                                                   arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_transpose_f16(const cmsis_nn_context *ctx,
                                      const cmsis_nn_transpose_params_f16 *params,
                                      const cmsis_nn_dims *input_dims,
                                      const float16_t *input,
                                      const cmsis_nn_dims *output_dims,
                                      float16_t *output);

arm_cmsis_nn_status arm_unpack_f16(const float16_t *input_data,
                                   int32_t input_dims,
                                   const int32_t *input_shape,
                                   int32_t axis,
                                   float16_t *const *output_data);

#endif

#if ARM_NN_ENABLE_F32
arm_cmsis_nn_status
arm_argmax_f32(const float32_t *input_data, const cmsis_nn_dims *input_dims, int32_t axis, int32_t *output_data);

arm_cmsis_nn_status
arm_argmin_f32(const float32_t *input_data, const cmsis_nn_dims *input_dims, int32_t axis, int32_t *output_data);

arm_cmsis_nn_status arm_avg_pool_f32(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params_f32 *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float32_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     float32_t *dst);

arm_cmsis_nn_status arm_batch_matmul_f32(const cmsis_nn_context *ctx,
                                         const cmsis_nn_bmm_params_f32 *bmm_params,
                                         const cmsis_nn_dims *input_lhs_dims,
                                         const float32_t *input_lhs,
                                         const cmsis_nn_dims *input_rhs_dims,
                                         const float32_t *input_rhs,
                                         const cmsis_nn_dims *output_dims,
                                         float32_t *output);

int32_t arm_batch_matmul_f32_get_buffer_size(const cmsis_nn_bmm_params_f32 *bmm_params,
                                             const cmsis_nn_dims *input_lhs_dims,
                                             const cmsis_nn_dims *input_rhs_dims,
                                             const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_batch_norm_f32(const float32_t *input,
                                       float32_t *output,
                                       const float32_t *scale,
                                       const float32_t *bias,
                                       const cmsis_nn_dims *input_dims,
                                       arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_concatenation_f32(const float32_t *const *input_data,
                                          int32_t num_inputs,
                                          const int32_t *axis_sizes,
                                          int32_t output_dims,
                                          const int32_t *output_shape,
                                          int32_t axis,
                                          float32_t *output_data);

void arm_concatenation_f32_w(const float32_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float32_t *output,
                             uint32_t offset_w);

void arm_concatenation_f32_x(const float32_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float32_t *output,
                             int32_t output_x,
                             uint32_t offset_x);

void arm_concatenation_f32_y(const float32_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float32_t *output,
                             int32_t output_y,
                             uint32_t offset_y);

void arm_concatenation_f32_z(const float32_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float32_t *output,
                             int32_t output_z,
                             uint32_t offset_z);

arm_cmsis_nn_status arm_convolve_1_x_n_f32(const cmsis_nn_context *ctx,
                                           const cmsis_nn_conv_params_f32 *conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float32_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const float32_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const float32_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           float32_t *output_data,
                                           arm_nn_tensor_layout layout);

int32_t arm_convolve_1_x_n_f32_get_buffer_size(const cmsis_nn_conv_params_f32 *conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_f32(const cmsis_nn_context *ctx,
                                                const cmsis_nn_conv_params_f32 *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float32_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const float32_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const float32_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                float32_t *output_data);

arm_cmsis_nn_status arm_convolve_1x1_f32(const cmsis_nn_context *ctx,
                                         const cmsis_nn_conv_params_f32 *conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const float32_t *input_data,
                                         const cmsis_nn_dims *filter_dims,
                                         const float32_t *filter_data,
                                         const cmsis_nn_dims *bias_dims,
                                         const float32_t *bias_data,
                                         const cmsis_nn_dims *output_dims,
                                         float32_t *output_data,
                                         arm_nn_tensor_layout layout);

int32_t arm_convolve_1x1_f32_get_buffer_size(const cmsis_nn_conv_params_f32 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const cmsis_nn_dims *filter_dims,
                                             const cmsis_nn_dims *output_dims,
                                             arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_convolve_1x1_nhwc_f32(const cmsis_nn_context *ctx,
                                              const cmsis_nn_conv_params_f32 *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const float32_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const float32_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const float32_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              float32_t *output_data);

arm_cmsis_nn_status arm_convolve_f32(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params_f32 *conv_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float32_t *input_data,
                                     const cmsis_nn_dims *filter_dims,
                                     const float32_t *filter_data,
                                     const cmsis_nn_dims *bias_dims,
                                     const float32_t *bias_data,
                                     const cmsis_nn_dims *output_dims,
                                     float32_t *output_data,
                                     arm_nn_tensor_layout layout);

int32_t arm_convolve_f32_get_buffer_size(const cmsis_nn_conv_params_f32 *conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const cmsis_nn_dims *filter_dims,
                                         const cmsis_nn_dims *output_dims,
                                         arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_convolve_nhwc_f32(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params_f32 *conv_params,
                                          const cmsis_nn_dims *input_dims,
                                          const float32_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const float32_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const float32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          float32_t *output_data);

arm_cmsis_nn_status arm_convolve_wrapper_f32(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params_f32 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const float32_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const float32_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const float32_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             float32_t *output_data);

int32_t arm_convolve_wrapper_f32_get_buffer_size(const cmsis_nn_conv_params_f32 *conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_depthwise_conv_f32(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params_f32 *dw_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float32_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const float32_t *kernel,
                                           const cmsis_nn_dims *bias_dims,
                                           const float32_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           float32_t *output,
                                           arm_nn_tensor_layout layout);

int32_t arm_depthwise_conv_f32_get_buffer_size(const cmsis_nn_dw_conv_params_f32 *dw_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_depthwise_conv_wrapper_f32(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_dw_conv_params_f32 *dw_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const float32_t *input,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const float32_t *kernel,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const float32_t *bias,
                                                   const cmsis_nn_dims *output_dims,
                                                   float32_t *output);

int32_t arm_depthwise_conv_wrapper_f32_get_buffer_size(const cmsis_nn_dw_conv_params_f32 *dw_conv_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_depthwise_nhwc_conv_f32(const cmsis_nn_context *ctx,
                                                const cmsis_nn_dw_conv_params_f32 *dw_conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float32_t *input,
                                                const cmsis_nn_dims *filter_dims,
                                                const float32_t *kernel,
                                                const cmsis_nn_dims *bias_dims,
                                                const float32_t *bias,
                                                const cmsis_nn_dims *output_dims,
                                                float32_t *output);

arm_cmsis_nn_status arm_elementwise_add_broadcast_f32(const float32_t *input_1_data,
                                                      const cmsis_nn_dims *input_1_dims,
                                                      const float32_t *input_2_data,
                                                      const cmsis_nn_dims *input_2_dims,
                                                      float32_t *output_data,
                                                      const cmsis_nn_dims *output_dims,
                                                      float32_t out_activation_min,
                                                      float32_t out_activation_max);

arm_cmsis_nn_status arm_elementwise_add_f32(const float32_t *input_1_vect,
                                            const float32_t *input_2_vect,
                                            float32_t *output,
                                            float32_t out_activation_min,
                                            float32_t out_activation_max,
                                            int32_t block_size);

arm_cmsis_nn_status arm_elementwise_mul_broadcast_f32(const float32_t *input_1_data,
                                                      const cmsis_nn_dims *input_1_dims,
                                                      const float32_t *input_2_data,
                                                      const cmsis_nn_dims *input_2_dims,
                                                      float32_t *output_data,
                                                      const cmsis_nn_dims *output_dims,
                                                      float32_t out_activation_min,
                                                      float32_t out_activation_max);

arm_cmsis_nn_status arm_elementwise_mul_f32(const float32_t *input_1_vect,
                                            const float32_t *input_2_vect,
                                            float32_t *output,
                                            float32_t out_activation_min,
                                            float32_t out_activation_max,
                                            int32_t block_size);

arm_cmsis_nn_status arm_elementwise_sub_broadcast_f32(const float32_t *input_1_data,
                                                      const cmsis_nn_dims *input_1_dims,
                                                      const float32_t *input_2_data,
                                                      const cmsis_nn_dims *input_2_dims,
                                                      float32_t *output_data,
                                                      const cmsis_nn_dims *output_dims,
                                                      float32_t out_activation_min,
                                                      float32_t out_activation_max);

arm_cmsis_nn_status arm_elementwise_sub_f32(const float32_t *input_1_vect,
                                            const float32_t *input_2_vect,
                                            float32_t *output,
                                            float32_t out_activation_min,
                                            float32_t out_activation_max,
                                            int32_t block_size);

arm_cmsis_nn_status arm_fully_connected_f32(const cmsis_nn_context *ctx,
                                            const cmsis_nn_fc_params_f32 *fc_params,
                                            const cmsis_nn_dims *input_dims,
                                            const float32_t *input,
                                            const cmsis_nn_dims *filter_dims,
                                            const float32_t *kernel,
                                            const cmsis_nn_dims *bias_dims,
                                            const float32_t *bias,
                                            const cmsis_nn_dims *output_dims,
                                            float32_t *output,
                                            arm_nn_tensor_layout layout);

int32_t arm_fully_connected_f32_get_buffer_size(const cmsis_nn_fc_params_f32 *fc_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims,
                                                arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_fully_connected_nhwc_f32(const cmsis_nn_context *ctx,
                                                 const cmsis_nn_fc_params_f32 *fc_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const float32_t *input,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const float32_t *kernel,
                                                 const cmsis_nn_dims *bias_dims,
                                                 const float32_t *bias,
                                                 const cmsis_nn_dims *output_dims,
                                                 float32_t *output);

arm_cmsis_nn_status arm_gather_f32(const float32_t *input_data,
                                   const cmsis_nn_dims *input_dims,
                                   const int32_t *indices_data,
                                   const cmsis_nn_dims *indices_dims,
                                   const cmsis_nn_gather_params *params,
                                   float32_t *output_data,
                                   const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_gather_nd_f32(const float32_t *params_data,
                                      const cmsis_nn_dims *params_dims,
                                      const int32_t *indices_data,
                                      const cmsis_nn_dims *indices_dims,
                                      const cmsis_nn_gather_nd_params *params,
                                      float32_t *output_data,
                                      const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_hard_swish_f32(const float32_t *input, float32_t *output, int32_t size);

arm_cmsis_nn_status arm_max_pool_f32(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params_f32 *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float32_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     float32_t *dst);

arm_cmsis_nn_status arm_maximum_f32(const cmsis_nn_context *ctx,
                                    const float32_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const float32_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    float32_t *output_data,
                                    const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_minimum_f32(const cmsis_nn_context *ctx,
                                    const float32_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const float32_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    float32_t *output_data,
                                    const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_nn_abs_f32(const float32_t *input, float32_t *output, int32_t block_size);

arm_cmsis_nn_status arm_nn_activation_f32(const float32_t *input,
                                          float32_t *output,
                                          int32_t size,
                                          arm_nn_activation_type_flt type,
                                          float32_t act_param);

arm_cmsis_nn_status arm_nn_fill_f32(float32_t value, float32_t *output, int32_t block_size);

arm_cmsis_nn_status arm_nn_mean_f32(const float32_t *input_data,
                                    const cmsis_nn_dims *input_dims,
                                    const cmsis_nn_dims *axis_dims,
                                    float32_t *output_data,
                                    const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_pack_f32(const float32_t *const *input_data,
                                 int32_t num_inputs,
                                 int32_t input_dims,
                                 const int32_t *input_shape,
                                 int32_t axis,
                                 float32_t *output_data);

arm_cmsis_nn_status arm_pad_f32(const float32_t *input,
                                float32_t *output,
                                float32_t pad_value,
                                const cmsis_nn_dims *input_size,
                                const cmsis_nn_dims *pre_pad,
                                const cmsis_nn_dims *post_pad);

arm_cmsis_nn_status arm_prelu_f32(const cmsis_nn_dims *input_dims,
                                  const float32_t *input,
                                  const cmsis_nn_dims *alpha_dims,
                                  const float32_t *alpha,
                                  const cmsis_nn_dims *output_dims,
                                  float32_t *output);

arm_cmsis_nn_status arm_reduce_max_f32(const float32_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *axis_dims,
                                       float32_t *output_data,
                                       const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_reduce_min_f32(const float32_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *axis_dims,
                                       float32_t *output_data,
                                       const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_reduce_sum_f32(const float32_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *axis_dims,
                                       float32_t *output_data,
                                       const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_resize_nearest_neighbor_f32(const cmsis_nn_context *ctx,
                                                    const cmsis_nn_resize_params *resize_params,
                                                    const cmsis_nn_dims *input_shape,
                                                    const float32_t *input_data,
                                                    const cmsis_nn_dims *output_size_shape,
                                                    const int32_t *output_size_data,
                                                    const cmsis_nn_dims *output_shape,
                                                    float32_t *output_data);

int32_t arm_resize_nearest_neighbor_f32_get_buffer_size(const cmsis_nn_dims *output_dims);

arm_cmsis_nn_status arm_softmax_f32(const float32_t *input, int32_t num_rows, int32_t row_size, float32_t *output);

arm_cmsis_nn_status arm_split_f32(const float32_t *input_data,
                                  int32_t input_dims,
                                  const int32_t *input_shape,
                                  int32_t axis,
                                  int32_t num_splits,
                                  const int32_t *split_dims,
                                  float32_t *const *output_data);

arm_cmsis_nn_status arm_strided_slice_f32(const float32_t *input_data,
                                          float32_t *output_data,
                                          const cmsis_nn_dims *const input_dims,
                                          const cmsis_nn_dims *const begin_dims,
                                          const cmsis_nn_dims *const stride_dims,
                                          const cmsis_nn_dims *const output_dims);

arm_cmsis_nn_status arm_transpose_conv_f32(const cmsis_nn_context *ctx,
                                           const cmsis_nn_context *output_ctx,
                                           const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float32_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const float32_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const float32_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           float32_t *output_data,
                                           arm_nn_tensor_layout layout);

int32_t arm_transpose_conv_f32_get_buffer_size(const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *out_dims);

int32_t
arm_transpose_conv_f32_get_reverse_conv_buffer_size(const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_transpose_conv_nhwc_f32(const cmsis_nn_context *ctx,
                                                const cmsis_nn_context *output_ctx,
                                                const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float32_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const float32_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const float32_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                float32_t *output_data);

arm_cmsis_nn_status arm_transpose_conv_wrapper_f32(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_context *output_ctx,
                                                   const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const float32_t *input_data,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const float32_t *filter_data,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const float32_t *bias_data,
                                                   const cmsis_nn_dims *output_dims,
                                                   float32_t *output_data,
                                                   arm_nn_tensor_layout layout);

arm_cmsis_nn_status arm_transpose_f32(const cmsis_nn_context *ctx,
                                      const cmsis_nn_transpose_params_f32 *params,
                                      const cmsis_nn_dims *input_dims,
                                      const float32_t *input,
                                      const cmsis_nn_dims *output_dims,
                                      float32_t *output);

arm_cmsis_nn_status arm_unpack_f32(const float32_t *input_data,
                                   int32_t input_dims,
                                   const int32_t *input_shape,
                                   int32_t axis,
                                   float32_t *const *output_data);

#endif
