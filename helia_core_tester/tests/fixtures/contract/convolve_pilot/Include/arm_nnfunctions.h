/* Verbatim declarations of every arm_convolve_* function (kernels and scratch queries) from
 * ns-cmsis-nn arm_nnfunctions.h; the fallback contract for unit tests without a checkout.
 * test_convolve_contract_fixture_matches_the_real_tree keeps it equal to the export. */

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

