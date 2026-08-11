#ifndef HCT_BENCHMARK_SERVER_ADAPTER_H
#define HCT_BENCHMARK_SERVER_ADAPTER_H

#include <stdint.h>

#include "arm_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    const int8_t *input;
    int32_t input_offset;
    int8_t *output;
    int32_t output_offset;
    int32_t output_multiplier;
    int32_t output_shift;
    int32_t activation_min;
    int32_t activation_max;
    int32_t block_size;
    uint8_t needs_rescale;
} hct_abs_s8_request_t;

typedef struct
{
    cmsis_nn_context *ctx;
    cmsis_nn_context *weight_sum_ctx;
    cmsis_nn_conv_params *conv_params;
    cmsis_nn_per_channel_quant_params *quant_params;
    cmsis_nn_dims *input_dims;
    const int8_t *input_data;
    cmsis_nn_dims *filter_dims;
    const int8_t *filter_data;
    cmsis_nn_dims *bias_dims;
    const int32_t *bias_data;
    cmsis_nn_dims *upscale_dims;
    cmsis_nn_dims *output_dims;
    int8_t *output_data;
} hct_convolve_s8_request_t;

arm_cmsis_nn_status hct_dispatch_abs_s8(const hct_abs_s8_request_t *request);
arm_cmsis_nn_status hct_dispatch_convolve_s8(const hct_convolve_s8_request_t *request);
arm_cmsis_nn_status hct_dispatch_kernel(uint32_t kernel_id, const void *request);
arm_cmsis_nn_status hct_link_smoke_invoke_abs_s8(void);
arm_cmsis_nn_status hct_link_smoke_invoke_convolve_s8(void);

#ifdef __cplusplus
}
#endif

#endif
