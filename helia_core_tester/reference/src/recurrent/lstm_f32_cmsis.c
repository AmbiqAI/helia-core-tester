/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variant: ns-cmsis-nn's arm_lstm_unidirectional_f32 arithmetic (arm_nn_lstm_step_f32), so a
 * float32 golden follows the kernel's activation approximations and operation order instead of exact
 * sigmoid/tanh. Inputs and layout as lstm_f32 (gates stacked forget, input, cell, output).
 *
 * Gate: bias, then + dot(input), then + dot(hidden). Sigmoid: e = exp(-|x|) by the 256-segment exp2
 * table, then 1 / (1 + e) or e / (1 + e). Gate tanh: the 384-segment table over [0, 6), linear
 * interpolation, saturating to +-1, the sign copied from x.
 * target 0 (Helium): dots in four float32 lanes (lane = k % 4, fused), reduced left to right;
 *   cell = fma(i, g, f * c) with f * c rounded first; hidden = o * tanh_table(cell) with a fused
 *   interpolation and the index floored.
 * target 1 (scalar with an FPU, as GCC -Ofast contracts it): fused dot steps, fused interpolations,
 *   cell = fma(f, c, i * g).
 * target 2 (soft float): nothing fused.
 * The scalar legs are compiled C, so GCC's contraction choice is modelled, not guaranteed; the
 * comparison tolerance (PyTorch's float32 assert_close) covers a different choice.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hct_cmsis_act_f32.h"
#include "hct_ref_internal.h"

typedef struct
{
    int fused;
    int target;
} Arith;

static float madd(const Arith *a, float x, float y, float z)
{
    return a->fused ? fmaf(x, y, z) : (float)(x * y) + z;
}

static float sigmoid(const Arith *a, float x)
{
    float y = 0.0f;
    (void)hct_cmsis_sigmoid_f32(x, a->target, &y);
    return y;
}

static float tanh_lut(const Arith *a, float x)
{
    float y = 0.0f;
    (void)hct_cmsis_tanh_f32(x, a->target, &y);
    return y;
}

static float dot(const Arith *a, const float *x, const float *w, int32_t n)
{
    if (a->target == 0)
    {
        float lane[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int32_t k = 0; k < n; ++k)
        {
            lane[k % 4] = fmaf(x[k], w[k], lane[k % 4]);
        }
        return 0.0f + (((lane[0] + lane[1]) + lane[2]) + lane[3]);
    }
    float acc = 0.0f;
    for (int32_t k = 0; k < n; ++k)
    {
        acc = madd(a, x[k], w[k], acc);
    }
    return acc;
}

typedef struct
{
    int32_t batch, steps, input_size, hidden;
} Shape;

static int32_t check(const HctLstmF32CmsisParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                     int32_t num_outputs, Shape *s, int *has_bias)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 4, outputs, num_outputs, 1));
    int64_t n = 0;
    int64_t n_bias = 0;
    for (int i = 0; i < 3; ++i)
    {
        HCT_TRY(hct_check_tensor(&inputs[i], HCT_FLOAT32, &n));
    }
    HCT_TRY(hct_check_tensor(&inputs[3], HCT_FLOAT32, &n_bias));
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT32, &n));
    if ((p->time_major != 0 && p->time_major != 1) || p->target < 0 || p->target > 2 || !(p->cell_clip >= 0.0f) ||
        isinf(p->cell_clip))
    {
        return HCT_E_PARAM;
    }
    const HctTensor *in = &inputs[0];
    const HctTensor *w = &inputs[1];
    const HctTensor *r = &inputs[2];
    const HctTensor *out = &outputs[0];
    if (in->rank != 3 || w->rank != 3 || r->rank != 3 || out->rank != 3)
    {
        return HCT_E_SHAPE;
    }
    s->batch = p->time_major ? in->dims[1] : in->dims[0];
    s->steps = p->time_major ? in->dims[0] : in->dims[1];
    s->input_size = in->dims[2];
    s->hidden = w->dims[1];
    *has_bias = n_bias != 0;
    if (s->hidden < 1 || w->dims[0] != 4 || w->dims[2] != s->input_size || r->dims[0] != 4 ||
        r->dims[1] != s->hidden || r->dims[2] != s->hidden || out->dims[0] != in->dims[0] ||
        out->dims[1] != in->dims[1] || out->dims[2] != s->hidden ||
        (*has_bias && (inputs[3].rank != 2 || inputs[3].dims[0] != 4 || inputs[3].dims[1] != s->hidden)))
    {
        return HCT_E_SHAPE;
    }
    return HCT_OK;
}

int32_t hct_ref_lstm_f32_cmsis(const HctLstmF32CmsisParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    Shape s;
    int has_bias = 0;
    HCT_TRY(check(params, inputs, num_inputs, outputs, num_outputs, &s, &has_bias));
    const Arith a = {params->target != 2, params->target};
    const int32_t I = s.input_size;
    const int32_t H = s.hidden;
    const size_t n_state = (size_t)s.batch * (size_t)H;
    float *cell = calloc(n_state, sizeof(float));
    float *hidden = calloc(n_state, sizeof(float));
    float *xrow = malloc((size_t)I * sizeof(float) + 1);
    float *wrow = malloc((size_t)(I > H ? I : H) * sizeof(float) + 1);
    if (cell == NULL || hidden == NULL || xrow == NULL || wrow == NULL)
    {
        free(cell);
        free(hidden);
        free(xrow);
        free(wrow);
        return HCT_E_SIZE;
    }
    const float clip = params->cell_clip;
    for (int32_t t = 0; t < s.steps; ++t)
    {
        for (int32_t b = 0; b < s.batch; ++b)
        {
            const int64_t xr = params->time_major ? (int64_t)t * s.batch + b : (int64_t)b * s.steps + t;
            for (int32_t k = 0; k < I; ++k)
            {
                xrow[k] = hct_load_f32(&inputs[0], xr * I + k);
            }
            /* The step reads the previous hidden state of this batch row; write after all units. */
            float *h_prev = &hidden[(size_t)b * H];
            float gates[4];
            float *h_new = malloc((size_t)H * sizeof(float));
            if (h_new == NULL)
            {
                free(cell);
                free(hidden);
                free(xrow);
                free(wrow);
                return HCT_E_SIZE;
            }
            for (int32_t u = 0; u < H; ++u)
            {
                for (int g = 0; g < 4; ++g)
                {
                    float acc = has_bias ? hct_load_f32(&inputs[3], (int64_t)g * H + u) : 0.0f;
                    for (int32_t k = 0; k < I; ++k)
                    {
                        wrow[k] = hct_load_f32(&inputs[1], ((int64_t)g * H + u) * I + k);
                    }
                    acc += dot(&a, xrow, wrow, I);
                    if (t > 0)
                    {
                        for (int32_t k = 0; k < H; ++k)
                        {
                            wrow[k] = hct_load_f32(&inputs[2], ((int64_t)g * H + u) * H + k);
                        }
                        acc += dot(&a, h_prev, wrow, H);
                    }
                    gates[g] = g == 2 ? tanh_lut(&a, acc) : sigmoid(&a, acc);
                }
                float *c = &cell[(size_t)b * H + u];
                float cv = 0.0f;
                if (params->target == 0)
                {
                    cv = fmaf(gates[1], gates[2], gates[0] * *c);
                }
                else if (params->target == 1)
                {
                    cv = fmaf(gates[0], *c, gates[1] * gates[2]);
                }
                else
                {
                    cv = (gates[0] * *c) + (gates[1] * gates[2]);
                }
                if (clip > 0.0f)
                {
                    cv = hct_clamp_f32(cv, -clip, clip);
                }
                *c = cv;
                h_new[u] = gates[3] * tanh_lut(&a, cv);
                hct_store_f32(&outputs[0], xr * H + u, h_new[u]);
            }
            memcpy(h_prev, h_new, (size_t)H * sizeof(float));
            free(h_new);
        }
    }
    free(cell);
    free(hidden);
    free(xrow);
    free(wrow);
    return HCT_OK;
}
