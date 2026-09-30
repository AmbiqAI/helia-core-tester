/* Verbatim declarations of every function of the contract-bound operators (arm_convolve_*, arm_depthwise_*, arm_fully_connected_*, arm_batch_matmul_*, arm_transpose_conv_*, arm_avgpool_*, arm_avg_pool_*, arm_max_pool_*)
 * from ns-cmsis-nn arm_nnsupportfunctions.h; the fallback contract for unit tests without a checkout.
 * test_bound_operator_fixture_matches_the_real_tree keeps it equal to the export. */

int32_t arm_depthwise_conv_s8_opt_get_buffer_size_dsp(const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims);

int32_t arm_depthwise_conv_s8_opt_get_buffer_size_mve(const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims);

