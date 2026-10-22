/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variant: ns-cmsis-nn's arm_gru_unidirectional_f32 arithmetic (arm_nn_gru_step_f32) per float32 target, so a
 * float32 golden follows the kernel's activation tables and operation order (read from the GCC -Ofast objects)
 * instead of exact sigmoid/tanh. Operands and layout as gru_f32 (gates stacked update z, reset r, candidate n).
 *
 * Projections: bias + dot; no recurrent dot on the first step (the state is zero, so a zero streaming state gives the
 * same values). z = sigmoid(ip_z + hp_z), r likewise; sigmoid and tanh are hct_cmsis_{sigmoid,tanh}_f32.
 * reset_after: n_pre = ip_n + r * hp_n. Otherwise s = sum_k Un[k] * (r[k] * h[k]) (r * h rounded),
 * n_pre = ip_n + (b_hn + s). h = fma(z, h_prev, (1 - z) * tanh(n_pre)) (an explicit fmaf on every target).
 * target 0 (Helium): dots in four float32 lanes (lane = k % 4, fused), reduced with the bias reassociated in as
 *   (((l0 + l1) + bias) + l2) + l3; n_pre = fma(r, hp_n, ip_n); the pre-reset sum as GCC vectorised it: for H >= 4
 *   the first 4 * (H / 4) terms in four lanes (product rounded, then added), lanes summed left to right, the rest
 *   fused one by one; for H < 4 a fused chain from zero. A NaN candidate becomes the all-ones NaN.
 * target 1 (scalar with an FPU): fused dot chains from zero, then bias + dot; n_pre = fma(r, hp_n, ip_n).
 * target 2 (soft float): nothing fused but the combine's fmaf (a libm call).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hct_cmsis_act_f32.h"
#include "hct_ref_internal.h"

#define LANES 4

typedef struct
{
    const HctTensor *w, *r, *bi, *bh;
    int has_bi, has_bh;
    int32_t I, H, target;
    float *x, *row;
} Gru;

static float madd(int32_t target, float a, float b, float c)
{
    return target != 2 ? fmaf(a, b, c) : (float)(a * b) + c;
}

static float proj(const Gru *g, const float *v, int32_t n, const HctTensor *w, int64_t row, float bias)
{
    for (int32_t k = 0; k < n; ++k)
    {
        g->row[k] = hct_load_f32(w, row * n + k);
    }
    if (g->target == 0)
    {
        float l[LANES] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int32_t k = 0; k < n; ++k)
        {
            l[k % LANES] = fmaf(v[k], g->row[k], l[k % LANES]);
        }
        return (((l[0] + l[1]) + bias) + l[2]) + l[3];
    }
    float acc = 0.0f;
    for (int32_t k = 0; k < n; ++k)
    {
        acc = madd(g->target, v[k], g->row[k], acc);
    }
    return bias + acc;
}

static float input_proj(const Gru *g, int gate, int32_t u)
{
    const int64_t row = (int64_t)gate * g->H + u;
    return proj(g, g->x, g->I, g->w, row, g->has_bi ? hct_load_f32(g->bi, row) : 0.0f);
}

static float hidden_proj(const Gru *g, const float *h_prev, int gate, int32_t u)
{
    const int64_t row = (int64_t)gate * g->H + u;
    const float bias = g->has_bh ? hct_load_f32(g->bh, row) : 0.0f;
    return h_prev == NULL ? bias : proj(g, h_prev, g->H, g->r, row, bias);
}

static float prereset_sum(const Gru *g, const float *rh, int64_t row)
{
    const int32_t H = g->H;
    for (int32_t k = 0; k < H; ++k)
    {
        g->row[k] = hct_load_f32(g->r, row * H + k);
    }
    float sum = 0.0f;
    int32_t k = 0;
    if (g->target == 0 && H >= LANES)
    {
        float l[LANES] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (; k + LANES <= H; k += LANES)
        {
            for (int j = 0; j < LANES; ++j)
            {
                l[j] = l[j] + g->row[k + j] * rh[k + j];
            }
        }
        sum = ((l[0] + l[1]) + l[2]) + l[3];
    }
    for (; k < H; ++k)
    {
        sum = madd(g->target, g->row[k], rh[k], sum);
    }
    return sum;
}

static int32_t sigmoid(const Gru *g, float x, float *out)
{
    return hct_cmsis_sigmoid_f32(x, g->target, out);
}

static int32_t step(const Gru *g, int32_t reset_after, const float *h_prev, float *stage, float *h_new)
{
    const int32_t H = g->H;
    if (!reset_after && h_prev != NULL)
    {
        for (int32_t u = 0; u < H; ++u)
        {
            float r = 0.0f;
            HCT_TRY(sigmoid(g, hidden_proj(g, h_prev, 1, u) + input_proj(g, 1, u), &r));
            stage[u] = r * h_prev[u];
        }
    }
    for (int32_t u = 0; u < H; ++u)
    {
        float z = 0.0f;
        HCT_TRY(sigmoid(g, hidden_proj(g, h_prev, 0, u) + input_proj(g, 0, u), &z));
        const float xh = input_proj(g, 2, u);
        float n_pre = 0.0f;
        if (reset_after)
        {
            float r = 0.0f;
            HCT_TRY(sigmoid(g, hidden_proj(g, h_prev, 1, u) + input_proj(g, 1, u), &r));
            const float hh = hidden_proj(g, h_prev, 2, u);
            n_pre = g->target == 2 ? xh + r * hh : fmaf(r, hh, xh);
        }
        else
        {
            float hh = g->has_bh ? hct_load_f32(g->bh, 2 * (int64_t)H + u) : 0.0f;
            if (h_prev != NULL)
            {
                hh = hh + prereset_sum(g, stage, 2 * (int64_t)H + u);
            }
            n_pre = xh + hh;
        }
        float cand = 0.0f;
        HCT_TRY(hct_cmsis_tanh_f32(n_pre, g->target, &cand));
        if (g->target == 0 && isnan(n_pre))
        {
            const uint32_t all_ones = 0xFFFFFFFFu;
            memcpy(&cand, &all_ones, sizeof(cand));
        }
        h_new[u] = fmaf(z, h_prev != NULL ? h_prev[u] : 0.0f, (1.0f - z) * cand);
    }
    return HCT_OK;
}

static int32_t check_bias(const HctTensor *b, int32_t hidden, int *present)
{
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(b, HCT_FLOAT32, &n));
    *present = n != 0;
    if (*present && (b->rank != 2 || b->dims[0] != 3 || b->dims[1] != hidden))
    {
        return HCT_E_SHAPE;
    }
    return HCT_OK;
}

int32_t hct_ref_gru_f32_cmsis(const HctGruF32CmsisParams *p, const HctTensor *inputs, int32_t num_inputs,
                              HctTensor *outputs, int32_t num_outputs)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 5, outputs, num_outputs, 1));
    int64_t n = 0;
    for (int i = 0; i < 3; ++i)
    {
        HCT_TRY(hct_check_tensor(&inputs[i], HCT_FLOAT32, &n));
    }
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT32, &n));
    if ((p->time_major != 0 && p->time_major != 1) || (p->reset_after != 0 && p->reset_after != 1) ||
        p->target < 0 || p->target > 2)
    {
        return HCT_E_PARAM;
    }
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    Gru g = {&inputs[1], &inputs[2], &inputs[3], &inputs[4], 0, 0, 0, 0, p->target, NULL, NULL};
    if (in->rank != 3 || g.w->rank != 3 || g.r->rank != 3 || out->rank != 3)
    {
        return HCT_E_SHAPE;
    }
    const int32_t B = p->time_major ? in->dims[1] : in->dims[0];
    const int32_t T = p->time_major ? in->dims[0] : in->dims[1];
    g.I = in->dims[2];
    g.H = g.w->dims[1];
    const int32_t H = g.H;
    if (H < 1 || g.w->dims[0] != 3 || g.w->dims[2] != g.I || g.r->dims[0] != 3 || g.r->dims[1] != H ||
        g.r->dims[2] != H || out->dims[0] != in->dims[0] || out->dims[1] != in->dims[1] || out->dims[2] != H)
    {
        return HCT_E_SHAPE;
    }
    HCT_TRY(check_bias(g.bi, H, &g.has_bi));
    HCT_TRY(check_bias(g.bh, H, &g.has_bh));
    const size_t widest = (size_t)(g.I > H ? g.I : H);
    float *hidden = calloc((size_t)B * (size_t)H + 1, sizeof(float));
    float *stage = malloc((size_t)H * sizeof(float));
    float *h_new = malloc((size_t)H * sizeof(float));
    g.x = malloc(((size_t)g.I + 1) * sizeof(float));
    g.row = malloc((widest + 1) * sizeof(float));
    int32_t status = HCT_OK;
    if (hidden == NULL || stage == NULL || h_new == NULL || g.x == NULL || g.row == NULL)
    {
        status = HCT_E_SIZE;
    }
    for (int32_t t = 0; t < T && status == HCT_OK; ++t)
    {
        for (int32_t b = 0; b < B && status == HCT_OK; ++b)
        {
            const int64_t xr = p->time_major ? (int64_t)t * B + b : (int64_t)b * T + t;
            for (int32_t k = 0; k < g.I; ++k)
            {
                g.x[k] = hct_load_f32(in, xr * g.I + k);
            }
            float *h = &hidden[(size_t)b * H];
            status = step(&g, p->reset_after, t > 0 ? h : NULL, stage, h_new);
            if (status == HCT_OK)
            {
                memcpy(h, h_new, (size_t)H * sizeof(float));
                for (int32_t u = 0; u < H; ++u)
                {
                    hct_store_f32(out, xr * H + u, h[u]);
                }
            }
        }
    }
    free(hidden);
    free(stage);
    free(h_new);
    free(g.x);
    free(g.row);
    return status;
}
