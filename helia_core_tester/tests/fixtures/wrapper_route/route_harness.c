/* Real wrappers, recording callees. */
#include <stdio.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/* Plain C: the planar plane rule. */
#include "NNSupportFunctions/arm_nn_depthwise_conv_s8_planar.c"

/* Headers stay plain; wrapper bodies see MVE. */
#ifdef HCT_ROUTE_MVE
#define ARM_MATH_MVEI 1
#endif

/* Real sizers: the adapter's ctx size. */
#include "ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c"
#include "ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.c"

static const char *g_route;
static const char *g_variant;
static int g_to_conv;

/* arm_depthwise_conv_s8_opt's planar or channelwise pick. */
static arm_cmsis_nn_status hct_dw_opt(const cmsis_nn_context *ctx,
                                      const cmsis_nn_dw_conv_params *p,
                                      const cmsis_nn_dims *i,
                                      const cmsis_nn_dims *f,
                                      const cmsis_nn_dims *o)
{
    g_route = "arm_depthwise_conv_s8_opt";
    g_variant = "arm_depthwise_conv_s8_opt_channelwise";
#if defined(ARM_MATH_MVEI)
    if (arm_nn_depthwise_conv_s8_planar_candidate(p, i))
    {
        /* The decline guard of arm_nn_depthwise_conv_s8_planar. */
        const int32_t bytes = arm_nn_depthwise_conv_s8_planar_bytes(p, i, f, o);
        if (!(bytes < 0 || ctx->buf == NULL || bytes > ctx->size ||
              bytes > arm_depthwise_conv_s8_opt_get_buffer_size(i, f)))
        {
            g_variant = "arm_depthwise_conv_s8_opt_planar";
        }
    }
#else
    (void)ctx;
    (void)p;
    (void)i;
    (void)f;
    (void)o;
#endif
    return ARM_CMSIS_NN_SUCCESS;
}

#define HCT_CALL(name) (g_route = #name, ARM_CMSIS_NN_SUCCESS)
#define arm_convolve_1x1_s8_fast(...) HCT_CALL(arm_convolve_1x1_s8_fast)
#define arm_convolve_1x1_s8(...) HCT_CALL(arm_convolve_1x1_s8)
#define arm_convolve_1_x_n_s8(...) HCT_CALL(arm_convolve_1_x_n_s8)
#define arm_convolve_1x1_out_s8(...) HCT_CALL(arm_convolve_1x1_out_s8)
#define arm_convolve_s8_small_cin(...) HCT_CALL(arm_convolve_s8_small_cin)
#define arm_convolve_s8_3x3_c16_s1(...) HCT_CALL(arm_convolve_s8_3x3_c16_s1)
#define arm_convolve_s8(...) HCT_CALL(arm_convolve_s8)
#define arm_depthwise_conv_3x3_s8(...) HCT_CALL(arm_depthwise_conv_3x3_s8)
#define arm_depthwise_conv_s8_opt(c, ws, p, q, i, in, f, k, bd, b, o, out) hct_dw_opt(c, p, i, f, o)
#define arm_depthwise_conv_s8(...) HCT_CALL(arm_depthwise_conv_s8)
#define arm_fully_connected_per_channel_s8(...) HCT_CALL(arm_fully_connected_per_channel_s8)
#define arm_fully_connected_s8(...) HCT_CALL(arm_fully_connected_s8)
#define arm_convolve_1x1_s4_fast(...) HCT_CALL(arm_convolve_1x1_s4_fast)
#define arm_convolve_1x1_s4(...) HCT_CALL(arm_convolve_1x1_s4)
#define arm_convolve_1_x_n_s4(...) HCT_CALL(arm_convolve_1_x_n_s4)
#define arm_convolve_even_s4(...) HCT_CALL(arm_convolve_even_s4)
#define arm_convolve_s4(...) HCT_CALL(arm_convolve_s4)
#define arm_convolve_1x1_s16_ns_np_nd(...) HCT_CALL(arm_convolve_1x1_s16_ns_np_nd)
#define arm_convolve_s16_fast_small_kernel(...) HCT_CALL(arm_convolve_s16_fast_small_kernel)
#define arm_convolve_s16_group_ch_mult_1(...) HCT_CALL(arm_convolve_s16_group_ch_mult_1)
#define arm_convolve_s16(...) HCT_CALL(arm_convolve_s16)
#define arm_depthwise_conv_s4_opt(...) HCT_CALL(arm_depthwise_conv_s4_opt)
#define arm_depthwise_conv_s4(...) HCT_CALL(arm_depthwise_conv_s4)
#define arm_depthwise_conv_fast_s16(...) HCT_CALL(arm_depthwise_conv_fast_s16)
#define arm_depthwise_conv_s16(...) HCT_CALL(arm_depthwise_conv_s16)
/* DW->conv: mark it, then route the conv. */
#define arm_transpose_s8(...) (g_to_conv = 1, ARM_CMSIS_NN_SUCCESS)
#define arm_convolve_wrapper_s8_get_buffer_size(...) 0

#include "ConvolutionFunctions/arm_convolve_wrapper_s8.c"
#include "ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c"
#include "FullyConnectedFunctions/arm_fully_connected_wrapper_s8.c"
#include "ConvolutionFunctions/arm_convolve_wrapper_s4.c"
#include "ConvolutionFunctions/arm_depthwise_conv_wrapper_s4.c"
#include "ConvolutionFunctions/arm_convolve_wrapper_s16.c"
#include "ConvolutionFunctions/arm_depthwise_conv_wrapper_s16.c"

/* Rows: kind, i, f, o dims, stride, pad, dil, ch_mult.
   Kinds: c/d/f s8, p/q s4, r/s s16 (conv/dw). */
int main(void)
{
    static int8_t buf[16];
    int32_t mult = 0, shift = 0;
    const cmsis_nn_context ctx = {buf, (int32_t)sizeof(buf)};
    const cmsis_nn_per_channel_quant_params pc = {&mult, &shift};
    char kind;
    cmsis_nn_dims i, f, o;
    const cmsis_nn_dims b = {0, 0, 0, 0};
    const cmsis_nn_bias_data b16 = {NULL, false};
    int32_t sh, sw, ph, pw, dh, dw, cm;

    while (scanf(" %c %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", &kind, &i.n, &i.h, &i.w, &i.c, &f.n,
                 &f.h, &f.w, &f.c, &o.n, &o.h, &o.w, &o.c, &sh, &sw, &ph, &pw, &dh, &dw, &cm) == 20)
    {
        const cmsis_nn_tile stride = {sw, sh}, pad = {pw, ph}, dil = {dw, dh};
        const cmsis_nn_activation act = {-128, 127};
        const cmsis_nn_conv_params cp = {0, 0, stride, pad, dil, act};
        const cmsis_nn_dw_conv_params dp = {0, 0, cm, stride, pad, dil, act};
        g_route = "none";
        g_variant = NULL;
        g_to_conv = 0;
        switch (kind)
        {
        case 'c':
            arm_convolve_wrapper_s8(&ctx, &ctx, &cp, &pc, &i, NULL, &f, NULL, &b, NULL, &o, NULL);
            break;
        case 'd':
        {
            /* The adapter sizes ctx with the wrapper sizer. */
            const int32_t size = arm_depthwise_conv_wrapper_s8_get_buffer_size(&dp, &i, &f, &o);
            const cmsis_nn_context dw_ctx = {size > 0 ? buf : NULL, size > 0 ? size : 0};
            arm_depthwise_conv_wrapper_s8(&dw_ctx, &ctx, &dp, &pc, &i, NULL, &f, NULL, &b, NULL, &o, NULL);
            break;
        }
        case 'p':
            arm_convolve_wrapper_s4(&ctx, &cp, &pc, &i, NULL, &f, NULL, &b, NULL, &o, NULL);
            break;
        case 'q':
            arm_depthwise_conv_wrapper_s4(&ctx, &dp, &pc, &i, NULL, &f, NULL, &b, NULL, &o, NULL);
            break;
        case 'r':
            arm_convolve_wrapper_s16(&ctx, &cp, &pc, &i, NULL, &f, NULL, &b, &b16, &o, NULL);
            break;
        case 's':
            arm_depthwise_conv_wrapper_s16(&ctx, &dp, &pc, &i, NULL, &f, NULL, &b, NULL, &o, NULL);
            break;
        default:
        {
            const cmsis_nn_fc_params fp = {0, 0, 0, act};
            const cmsis_nn_quant_params q = {&mult, &shift, 1};
            arm_fully_connected_wrapper_s8(&ctx, &fp, &q, &i, NULL, &f, NULL, &b, NULL, &o, NULL);
        }
        }
        printf("%s%s%s%s\n", g_to_conv ? "to_conv:" : "", g_route, g_variant ? "+" : "", g_variant ? g_variant : "");
    }
    return 0;
}
