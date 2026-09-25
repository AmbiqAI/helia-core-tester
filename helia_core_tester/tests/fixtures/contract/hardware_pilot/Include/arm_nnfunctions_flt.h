/* Declarations copied verbatim from ns-cmsis-nn Include/arm_nnfunctions_flt.h for the hardware adapter
 * pilot; test_hardware_adapter_contract.py checks them against the real export. */

#if ARM_NN_ENABLE_F32
arm_cmsis_nn_status arm_elementwise_add_f32(const float32_t *input_1_vect,
                                            const float32_t *input_2_vect,
                                            float32_t *output,
                                            float32_t out_activation_min,
                                            float32_t out_activation_max,
                                            int32_t block_size);

arm_cmsis_nn_status arm_elementwise_sub_f32(const float32_t *input_1_vect,
                                            const float32_t *input_2_vect,
                                            float32_t *output,
                                            float32_t out_activation_min,
                                            float32_t out_activation_max,
                                            int32_t block_size);

arm_cmsis_nn_status arm_elementwise_mul_f32(const float32_t *input_1_vect,
                                            const float32_t *input_2_vect,
                                            float32_t *output,
                                            float32_t out_activation_min,
                                            float32_t out_activation_max,
                                            int32_t block_size);

#endif

#if ARM_NN_ENABLE_F16
arm_cmsis_nn_status arm_elementwise_add_f16(const float16_t *input_1_vect,
                                            const float16_t *input_2_vect,
                                            float16_t *output,
                                            float16_t out_activation_min,
                                            float16_t out_activation_max,
                                            int32_t block_size);

arm_cmsis_nn_status arm_elementwise_sub_f16(const float16_t *input_1_vect,
                                            const float16_t *input_2_vect,
                                            float16_t *output,
                                            float16_t out_activation_min,
                                            float16_t out_activation_max,
                                            int32_t block_size);

arm_cmsis_nn_status arm_elementwise_mul_f16(const float16_t *input_1_vect,
                                            const float16_t *input_2_vect,
                                            float16_t *output,
                                            float16_t out_activation_min,
                                            float16_t out_activation_max,
                                            int32_t block_size);

#endif
