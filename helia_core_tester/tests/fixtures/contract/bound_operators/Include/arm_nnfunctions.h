/* Verbatim declarations of every function of the contract-bound operators (arm_convolve_*, arm_depthwise_*, arm_fully_connected_*, arm_batch_matmul_*, arm_transpose_conv_*, arm_avgpool_*, arm_avg_pool_*, arm_max_pool_*, arm_relu*, arm_clamp_*, arm_hard_swish_*, arm_leaky_relu_*, arm_logistic_*, arm_tanh_*, arm_nn_activation_*, arm_prelu_*)
 * from ns-cmsis-nn arm_nnfunctions.h; the fallback contract for unit tests without a checkout.
 * test_bound_operator_fixture_matches_the_real_tree keeps it equal to the export. */

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

arm_cmsis_nn_status arm_logistic_s16(const int16_t *input,
                                     int16_t *output,
                                     const int32_t input_size,
                                     int32_t input_multiplier,
                                     int32_t input_left_shift);

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

arm_cmsis_nn_status arm_nn_activation_s16(const int16_t *input,
                                          int16_t *output,
                                          const int32_t size,
                                          const int32_t left_shift,
                                          const arm_nn_activation_type type);

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

arm_cmsis_nn_status arm_tanh_s16(const int16_t *input,
                                 int16_t *output,
                                 const int32_t input_size,
                                 int32_t input_multiplier,
                                 int32_t input_left_shift);

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

