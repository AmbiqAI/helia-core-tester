/* Verbatim declarations of every function of the contract-bound operators (arm_convolve_*, arm_depthwise_*)
 * from ns-cmsis-nn arm_nnfunctions_flt.h; the fallback contract for unit tests without a checkout.
 * test_bound_operator_fixture_matches_the_real_tree keeps it equal to the export. */

#if ARM_NN_ENABLE_F16
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

#endif

#if ARM_NN_ENABLE_F32
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

#endif
