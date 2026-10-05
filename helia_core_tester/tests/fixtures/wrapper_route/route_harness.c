/* Real s8 wrappers, recording callees. */
#include <stdio.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/* Headers stay plain; wrapper bodies see MVE. */
#ifdef HCT_ROUTE_MVE
#define ARM_MATH_MVEI 1
#endif

static const char *g_route;
static int g_to_conv;

#define HCT_CALL(name) (g_route = #name, ARM_CMSIS_NN_SUCCESS)
#define arm_convolve_1x1_s8_fast(...) HCT_CALL(arm_convolve_1x1_s8_fast)
#define arm_convolve_1x1_s8(...) HCT_CALL(arm_convolve_1x1_s8)
#define arm_convolve_1_x_n_s8(...) HCT_CALL(arm_convolve_1_x_n_s8)
#define arm_convolve_1x1_out_s8(...) HCT_CALL(arm_convolve_1x1_out_s8)
#define arm_convolve_s8_small_cin(...) HCT_CALL(arm_convolve_s8_small_cin)
#define arm_convolve_s8_3x3_c16_s1(...) HCT_CALL(arm_convolve_s8_3x3_c16_s1)
#define arm_convolve_s8(...) HCT_CALL(arm_convolve_s8)
#define arm_depthwise_conv_3x3_s8(...) HCT_CALL(arm_depthwise_conv_3x3_s8)
#define arm_depthwise_conv_s8_opt(...) HCT_CALL(arm_depthwise_conv_s8_opt)
#define arm_depthwise_conv_s8(...) HCT_CALL(arm_depthwise_conv_s8)
#define arm_fully_connected_per_channel_s8(...) HCT_CALL(arm_fully_connected_per_channel_s8)
#define arm_fully_connected_s8(...) HCT_CALL(arm_fully_connected_s8)
/* DW->conv: mark it, then route the conv. */
#define arm_transpose_s8(...) (g_to_conv = 1, ARM_CMSIS_NN_SUCCESS)
#define arm_convolve_wrapper_s8_get_buffer_size(...) 0

#include "ConvolutionFunctions/arm_convolve_wrapper_s8.c"
#include "ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c"
#include "FullyConnectedFunctions/arm_fully_connected_wrapper_s8.c"

/* Rows: kind, i, f, o dims, stride, pad, dil, ch_mult. */
int main(void)
{
    static int8_t buf[16];
    int32_t mult = 0, shift = 0;
    const cmsis_nn_context ctx = {buf, (int32_t)sizeof(buf)};
    const cmsis_nn_per_channel_quant_params pc = {&mult, &shift};
    char kind;
    cmsis_nn_dims i, f, o;
    const cmsis_nn_dims b = {0, 0, 0, 0};
    int32_t sh, sw, ph, pw, dh, dw, cm;

    while (scanf(" %c %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", &kind, &i.n, &i.h, &i.w, &i.c, &f.n,
                 &f.h, &f.w, &f.c, &o.n, &o.h, &o.w, &o.c, &sh, &sw, &ph, &pw, &dh, &dw, &cm) == 20)
    {
        const cmsis_nn_tile stride = {sw, sh}, pad = {pw, ph}, dil = {dw, dh};
        const cmsis_nn_activation act = {-128, 127};
        g_route = "none";
        g_to_conv = 0;
        if (kind == 'c')
        {
            const cmsis_nn_conv_params p = {0, 0, stride, pad, dil, act};
            arm_convolve_wrapper_s8(&ctx, &ctx, &p, &pc, &i, NULL, &f, NULL, &b, NULL, &o, NULL);
        }
        else if (kind == 'd')
        {
            const cmsis_nn_dw_conv_params p = {0, 0, cm, stride, pad, dil, act};
            arm_depthwise_conv_wrapper_s8(&ctx, &ctx, &p, &pc, &i, NULL, &f, NULL, &b, NULL, &o, NULL);
        }
        else
        {
            const cmsis_nn_fc_params p = {0, 0, 0, act};
            const cmsis_nn_quant_params q = {&mult, &shift, 1};
            arm_fully_connected_wrapper_s8(&ctx, &p, &q, &i, NULL, &f, NULL, &b, NULL, &o, NULL);
        }
        printf("%s%s\n", g_to_conv ? "to_conv:" : "", g_route);
    }
    return 0;
}
