#include "benchmark_server_adapter.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

arm_cmsis_nn_status hct_dispatch_abs_s8(const hct_abs_s8_request_t *request)
{
    if (request == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_abs_s8(
        request->input,
        request->input_offset,
        request->output,
        request->output_offset,
        request->output_multiplier,
        request->output_shift,
        request->needs_rescale != 0u,
        request->activation_min,
        request->activation_max,
        request->block_size);
}

#ifndef HCT_HOST_ABS_ONLY
arm_cmsis_nn_status hct_dispatch_convolve_s8(const hct_convolve_s8_request_t *request)
{
    if (request == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_convolve_s8(request->ctx,
                           request->weight_sum_ctx,
                           request->conv_params,
                           request->quant_params,
                           request->input_dims,
                           request->input_data,
                           request->filter_dims,
                           request->filter_data,
                           request->bias_dims,
                           request->bias_data,
                           request->upscale_dims,
                           request->output_dims,
                           request->output_data);
}
#else
arm_cmsis_nn_status hct_dispatch_convolve_s8(const hct_convolve_s8_request_t *request)
{
    (void)request;
    return ARM_CMSIS_NN_ARG_ERROR;
}
#endif

arm_cmsis_nn_status hct_dispatch_kernel(uint32_t kernel_id, const void *request)
{
    switch (kernel_id)
    {
        case 1u:
            return hct_dispatch_abs_s8((const hct_abs_s8_request_t *)request);
        case 2u:
            return hct_dispatch_convolve_s8((const hct_convolve_s8_request_t *)request);
        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}

arm_cmsis_nn_status hct_link_smoke_invoke_abs_s8(void)
{
    static const int8_t kInput[8] = {0, -1, 2, -3, 4, -5, 6, -7};
    static int8_t kOutput[8];
    const hct_abs_s8_request_t request = {
        .input = kInput,
        .input_offset = 0,
        .output = kOutput,
        .output_offset = 0,
        .output_multiplier = 0x40000000,
        .output_shift = 1,
        .activation_min = -128,
        .activation_max = 127,
        .block_size = (int32_t)(sizeof(kInput) / sizeof(kInput[0])),
        .needs_rescale = 0u,
    };
    return hct_dispatch_kernel(1u, &request);
}

#ifndef HCT_HOST_ABS_ONLY
arm_cmsis_nn_status hct_link_smoke_invoke_convolve_s8(void)
{
    static int8_t kInput[3] = {1, -2, 3};
    static int8_t kFilter[1] = {1};
    static int32_t kBias[1] = {0};
    static int32_t kMultiplier[1] = {0x40000000};
    static int32_t kShift[1] = {1};
    static int8_t kOutput[3];
    static int16_t kScratch[8];
    static int32_t kWeightSum[1];
    cmsis_nn_context ctx = {.buf = kScratch, .size = sizeof(kScratch)};
    cmsis_nn_context weight_sum_ctx = {.buf = kWeightSum, .size = sizeof(kWeightSum)};
    cmsis_nn_conv_params conv_params = {
        .input_offset = 0,
        .output_offset = 0,
        .stride = {.w = 1, .h = 1},
        .padding = {.w = 0, .h = 0},
        .dilation = {.w = 1, .h = 1},
        .activation = {.min = -128, .max = 127},
    };
    cmsis_nn_per_channel_quant_params quant_params = {.multiplier = kMultiplier, .shift = kShift};
    cmsis_nn_dims input_dims = {.n = 1, .h = 1, .w = 3, .c = 1};
    cmsis_nn_dims filter_dims = {.n = 1, .h = 1, .w = 1, .c = 1};
    cmsis_nn_dims bias_dims = {.n = 0, .h = 0, .w = 0, .c = 1};
    cmsis_nn_dims output_dims = {.n = 1, .h = 1, .w = 3, .c = 1};
    if (arm_convolve_weight_sum(kWeightSum, kFilter, &input_dims, &filter_dims, &output_dims, 0, kBias) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    hct_convolve_s8_request_t request = {
        .ctx = &ctx,
        .weight_sum_ctx = &weight_sum_ctx,
        .conv_params = &conv_params,
        .quant_params = &quant_params,
        .input_dims = &input_dims,
        .input_data = kInput,
        .filter_dims = &filter_dims,
        .filter_data = kFilter,
        .bias_dims = &bias_dims,
        .bias_data = kBias,
        .upscale_dims = NULL,
        .output_dims = &output_dims,
        .output_data = kOutput,
    };
    return hct_dispatch_kernel(2u, &request);
}
#else
arm_cmsis_nn_status hct_link_smoke_invoke_convolve_s8(void)
{
    return ARM_CMSIS_NN_ARG_ERROR;
}
#endif
