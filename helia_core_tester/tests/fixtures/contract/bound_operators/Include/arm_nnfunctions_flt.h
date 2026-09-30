/* Verbatim declarations of every function of the contract-bound operators (arm_convolve_*, arm_depthwise_*, arm_fully_connected_*, arm_batch_matmul_*, arm_transpose_conv_*, arm_avgpool_*, arm_avg_pool_*, arm_max_pool_*, arm_relu*, arm_clamp_*, arm_hard_swish_*, arm_leaky_relu_*, arm_logistic_*, arm_tanh_*, arm_nn_activation_*, arm_prelu_*)
 * from ns-cmsis-nn arm_nnfunctions_flt.h; the fallback contract for unit tests without a checkout.
 * test_bound_operator_fixture_matches_the_real_tree keeps it equal to the export. */

#if ARM_NN_ENABLE_F16
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

arm_cmsis_nn_status arm_hard_swish_f16(const float16_t *input, float16_t *output, int32_t size);

arm_cmsis_nn_status arm_max_pool_f16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params_f16 *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float16_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     float16_t *dst);

arm_cmsis_nn_status arm_nn_activation_f16(const float16_t *input,
                                          float16_t *output,
                                          int32_t size,
                                          arm_nn_activation_type_flt type,
                                          float16_t act_param);

arm_cmsis_nn_status arm_prelu_f16(const cmsis_nn_dims *input_dims,
                                  const float16_t *input,
                                  const cmsis_nn_dims *alpha_dims,
                                  const float16_t *alpha,
                                  const cmsis_nn_dims *output_dims,
                                  float16_t *output);

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

#endif

#if ARM_NN_ENABLE_F32
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

arm_cmsis_nn_status arm_hard_swish_f32(const float32_t *input, float32_t *output, int32_t size);

arm_cmsis_nn_status arm_max_pool_f32(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params_f32 *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float32_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     float32_t *dst);

arm_cmsis_nn_status arm_nn_activation_f32(const float32_t *input,
                                          float32_t *output,
                                          int32_t size,
                                          arm_nn_activation_type_flt type,
                                          float32_t act_param);

arm_cmsis_nn_status arm_prelu_f32(const cmsis_nn_dims *input_dims,
                                  const float32_t *input,
                                  const cmsis_nn_dims *alpha_dims,
                                  const float32_t *alpha,
                                  const cmsis_nn_dims *output_dims,
                                  float32_t *output);

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

#endif
