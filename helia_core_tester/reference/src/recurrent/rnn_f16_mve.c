/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Named variants: ns-cmsis-nn's float16 recurrent kernels on Helium (cortex-m55, GCC -Ofast), bit for bit,
 * as the shipped objects compute them (read from their disassembly). Operands and layouts as lstm_f16,
 * gru_f16 and svdf_f16.
 *
 * Shared pieces:
 *   dot: element k in float16 lane k % 8 (fused multiply-add, one rounding), lanes reduced as
 *     ((0+1)+(2+3))+((4+5)+(6+7)), every add rounded to float16.
 *   sigmoid (arm_nn_sigmoid_scalar_f16): e = exp(-|x|) by arm_nn_softmax_exp_lut_f16 in float32 (x clamped
 *     to [-80, 80] by minNum/maxNum, t = x log2e, n = floor of t rounded to float16, f = t - n, the
 *     256-segment exp2 table interpolated with one fused multiply-add, scaled by 2^n with n clamped to
 *     [-14, 15], rounded to float16), then 1 / (1 + e) or e / (1 + e) in float32, rounded to float16.
 *   tanh (arm_nn_tanh_scalar_ref_f16 and arm_nn_vtanh_lut_direct_mve_f16, which agree): the 256-segment
 *     table over [0, 4], diff = y1 - y0 rounded, y0 + diff * frac fused, saturating to 1 past 4, the sign
 *     copied from x; zero and NaN pass through.
 *
 * lstm_f16_mve (arm_nn_lstm_step_f16): gate = bias, + dot(x) rounded, + dot(h) rounded (no recurrent term
 *   on the first step); cell = fma(i, g, f * c) with f * c rounded, clipped by maxNum/minNum when
 *   cell_clip > 0; hidden = o * tanh(cell) rounded.
 * gru_f16_mve (arm_nn_gru_step_f16): projections are bias + dot, each rounded; z = sigmoid(ip + hp).
 *   reset_after: n_pre = fma(r, hp_n, ip_n). Otherwise r * h is staged (rounded) and its recurrent sum
 *   runs as GCC vectorised the scalar loop: for H >= 8 the first 8 * (H / 8) terms in eight float16 lanes
 *   (product rounded, then added), the lanes summed left to right, the rest fused one by one; for H < 8
 *   a fused chain from zero. n_pre = ip_n + (b_hn + sum). h = fma(z, h_prev, (1 - z) * tanh(n_pre)).
 * svdf_f16_mve (arm_svdf_f16), one kernel call per input step: the state shifts, the feature dot (no
 *   bias) fills the newest slot, each filter's time dot reduces its memory, rank 1 with bias adds the
 *   bias in float16, otherwise the rank sum runs in float32 from the bias (or zero) and rounds once; the
 *   output clamp is maxNum then minNum against the float16 bounds.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hct_ref_internal.h"

#define LANES 8

static float h16(float v)
{
    return hct_round_f16(v);
}

/* a * b + c in float16, rounded once: the double fma is exact in the product and rounds the sum at 53
 * bits, which is innocuous before the rounding to 11 bits (53 >= 2 * 22 + 2). */
static float fma16(float a, float b, float c)
{
    return hct_f16_to_f32(hct_f64_to_f16(fma((double)a, (double)b, (double)c)));
}

static float dot16(const float *a, const float *b, int32_t n)
{
    float l[LANES] = {0.0f};
    for (int32_t k = 0; k < n; ++k)
    {
        l[k % LANES] = fma16(a[k], b[k], l[k % LANES]);
    }
    const float lo = h16(h16(l[0] + l[1]) + h16(l[2] + l[3]));
    const float hi = h16(h16(l[4] + l[5]) + h16(l[6] + l[7]));
    return h16(lo + hi);
}

/* VCVT.S32.F32: truncation, saturating, NaN to 0. */
static int32_t cvt_s32(float v)
{
    if (isnan(v))
    {
        return 0;
    }
    if (v >= 2147483648.0f)
    {
        return INT32_MAX;
    }
    if (v < -2147483648.0f)
    {
        return INT32_MIN;
    }
    return (int32_t)v;
}

static float exp_lut16(float x)
{
    float xf = fmaxf(fminf(x, 80.0f), -80.0f);
    const float t = xf * 1.44269504088896341f;
    const float th = h16(t);
    int32_t n = cvt_s32(th);
    n = th < (float)n ? n - 1 : n;
    const float idx_f = (t - (float)n) * 256.0f;
    int32_t idx = cvt_s32(idx_f);
    idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
    const float frac = idx_f - (float)idx;
    const float y0 = hct_f16_to_f32(hct_cmsis_exp2_lut_f16[idx]);
    const float y1 = hct_f16_to_f32(hct_cmsis_exp2_lut_f16[idx + 1]);
    n = n > 15 ? 15 : (n < -14 ? -14 : n);
    return h16(fmaf(frac, y1 - y0, y0) * ldexpf(1.0f, n));
}

static float sigmoid16(float x)
{
    if (x >= 0.0f)
    {
        return h16(1.0f / (1.0f + exp_lut16(-x)));
    }
    const float e = exp_lut16(x);
    return h16(e / (1.0f + e));
}

static float tanh16(float x)
{
    if (x == 0.0f || isnan(x))
    {
        return x;
    }
    const float ax = fabsf(x);
    float y = 1.0f;
    if (!(ax > 4.0f))
    {
        const float t = ax * 64.0f;
        const int32_t idx = (int32_t)t < 255 ? (int32_t)t : 255;
        const float frac = t - (float)idx;
        const float y0 = hct_f16_to_f32(hct_cmsis_tanh_lut_f16[idx]);
        const float diff = h16(hct_f16_to_f32(hct_cmsis_tanh_lut_f16[idx + 1]) - y0);
        y = fma16(frac, diff, y0);
    }
    return copysignf(y, x);
}

static int32_t check_rnn(const HctTensor *inputs, const HctTensor *outputs, int32_t gates, int32_t time_major,
                         int32_t *B, int32_t *T, int32_t *I, int32_t *H)
{
    int64_t n = 0;
    for (int i = 0; i < 3; ++i)
    {
        HCT_TRY(hct_check_tensor(&inputs[i], HCT_FLOAT16, &n));
    }
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT16, &n));
    const HctTensor *in = &inputs[0];
    const HctTensor *w = &inputs[1];
    const HctTensor *r = &inputs[2];
    const HctTensor *out = &outputs[0];
    if (in->rank != 3 || w->rank != 3 || r->rank != 3 || out->rank != 3)
    {
        return HCT_E_SHAPE;
    }
    *B = time_major ? in->dims[1] : in->dims[0];
    *T = time_major ? in->dims[0] : in->dims[1];
    *I = in->dims[2];
    *H = w->dims[1];
    if (*H < 1 || w->dims[0] != gates || w->dims[2] != *I || r->dims[0] != gates || r->dims[1] != *H ||
        r->dims[2] != *H || out->dims[0] != in->dims[0] || out->dims[1] != in->dims[1] || out->dims[2] != *H)
    {
        return HCT_E_SHAPE;
    }
    return HCT_OK;
}

static int32_t check_bias(const HctTensor *b, int32_t gates, int32_t hidden, int *present)
{
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(b, HCT_FLOAT16, &n));
    *present = n != 0;
    if (*present && (b->rank != 2 || b->dims[0] != gates || b->dims[1] != hidden))
    {
        return HCT_E_SHAPE;
    }
    return HCT_OK;
}

static void load_row(const HctTensor *t, int64_t row, int32_t len, float *dst)
{
    for (int32_t k = 0; k < len; ++k)
    {
        dst[k] = hct_load_f32(t, row * len + k);
    }
}

typedef struct
{
    float *hidden, *cell, *x, *w, *h_new, *stage;
} Scratch;

static void free_scratch(Scratch *s)
{
    free(s->hidden);
    free(s->cell);
    free(s->x);
    free(s->w);
    free(s->h_new);
    free(s->stage);
}

static int32_t alloc_scratch(Scratch *s, int32_t B, int32_t I, int32_t H)
{
    const size_t n_state = (size_t)B * (size_t)H + 1;
    s->hidden = calloc(n_state, sizeof(float));
    s->cell = calloc(n_state, sizeof(float));
    s->x = malloc(((size_t)I + 1) * sizeof(float));
    s->w = malloc(((size_t)(I > H ? I : H) + 1) * sizeof(float));
    s->h_new = malloc(((size_t)H + 1) * sizeof(float));
    s->stage = malloc(((size_t)H + 1) * sizeof(float));
    if (s->hidden == NULL || s->cell == NULL || s->x == NULL || s->w == NULL || s->h_new == NULL ||
        s->stage == NULL)
    {
        free_scratch(s);
        return HCT_E_SIZE;
    }
    return HCT_OK;
}

/* ------------------------------------------------------------------ LSTM */

int32_t hct_ref_lstm_f16_mve(const HctLstmFloatParams *p, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 4, outputs, num_outputs, 1));
    if ((p->time_major != 0 && p->time_major != 1) || !(p->cell_clip >= 0.0f) || isinf(p->cell_clip))
    {
        return HCT_E_PARAM;
    }
    int32_t B = 0, T = 0, I = 0, H = 0;
    int has_bias = 0;
    HCT_TRY(check_rnn(inputs, outputs, 4, p->time_major, &B, &T, &I, &H));
    HCT_TRY(check_bias(&inputs[3], 4, H, &has_bias));
    Scratch s;
    HCT_TRY(alloc_scratch(&s, B, I, H));
    const float clip = h16(p->cell_clip);
    for (int32_t t = 0; t < T; ++t)
    {
        for (int32_t b = 0; b < B; ++b)
        {
            const int64_t xr = p->time_major ? (int64_t)t * B + b : (int64_t)b * T + t;
            load_row(&inputs[0], xr, I, s.x);
            float *h_prev = &s.hidden[(size_t)b * H];
            for (int32_t u = 0; u < H; ++u)
            {
                float g[4];
                for (int gate = 0; gate < 4; ++gate)
                {
                    const int64_t row = (int64_t)gate * H + u;
                    float acc = has_bias ? hct_load_f32(&inputs[3], row) : 0.0f;
                    load_row(&inputs[1], row, I, s.w);
                    acc = h16(acc + dot16(s.x, s.w, I));
                    if (t > 0)
                    {
                        load_row(&inputs[2], row, H, s.w);
                        acc = h16(acc + dot16(h_prev, s.w, H));
                    }
                    g[gate] = gate == 2 ? tanh16(acc) : sigmoid16(acc);
                }
                float *c = &s.cell[(size_t)b * H + u];
                float cv = fma16(g[1], g[2], h16(g[0] * *c));
                if (clip > 0.0f)
                {
                    cv = fminf(fmaxf(cv, -clip), clip);
                }
                *c = cv;
                s.h_new[u] = h16(g[3] * tanh16(cv));
                hct_store_f32(&outputs[0], xr * H + u, s.h_new[u]);
            }
            memcpy(h_prev, s.h_new, (size_t)H * sizeof(float));
        }
    }
    free_scratch(&s);
    return HCT_OK;
}

/* ------------------------------------------------------------------- GRU */

typedef struct
{
    const HctTensor *w, *r, *bi, *bh;
    int has_bi, has_bh;
    int32_t I, H;
} GruOps;

static float input_proj(const GruOps *o, Scratch *s, int gate, int32_t u)
{
    const int64_t row = (int64_t)gate * o->H + u;
    const float acc = o->has_bi ? hct_load_f32(o->bi, row) : 0.0f;
    load_row(o->w, row, o->I, s->w);
    return h16(acc + dot16(s->x, s->w, o->I));
}

static float hidden_proj(const GruOps *o, Scratch *s, const float *h_prev, int gate, int32_t u)
{
    const int64_t row = (int64_t)gate * o->H + u;
    const float acc = o->has_bh ? hct_load_f32(o->bh, row) : 0.0f;
    if (h_prev == NULL)
    {
        return acc;
    }
    load_row(o->r, row, o->H, s->w);
    return h16(acc + dot16(h_prev, s->w, o->H));
}

/* The pre-reset candidate's recurrent sum, as GCC -Ofast vectorised its scalar loop. */
static float prereset_sum(const float *w, const float *rh, int32_t H)
{
    float sum = 0.0f;
    int32_t k = 0;
    if (H >= LANES)
    {
        float l[LANES] = {0.0f};
        for (; k + LANES <= H; k += LANES)
        {
            for (int j = 0; j < LANES; ++j)
            {
                l[j] = h16(l[j] + h16(w[k + j] * rh[k + j]));
            }
        }
        sum = l[0];
        for (int j = 1; j < LANES; ++j)
        {
            sum = h16(sum + l[j]);
        }
    }
    for (; k < H; ++k)
    {
        sum = fma16(w[k], rh[k], sum);
    }
    return sum;
}

int32_t hct_ref_gru_f16_mve(const HctGruParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                            int32_t num_outputs)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 5, outputs, num_outputs, 1));
    if ((p->time_major != 0 && p->time_major != 1) || (p->reset_after != 0 && p->reset_after != 1))
    {
        return HCT_E_PARAM;
    }
    GruOps o = {&inputs[1], &inputs[2], &inputs[3], &inputs[4], 0, 0, 0, 0};
    int32_t B = 0, T = 0;
    HCT_TRY(check_rnn(inputs, outputs, 3, p->time_major, &B, &T, &o.I, &o.H));
    HCT_TRY(check_bias(o.bi, 3, o.H, &o.has_bi));
    HCT_TRY(check_bias(o.bh, 3, o.H, &o.has_bh));
    const int32_t H = o.H;
    Scratch s;
    HCT_TRY(alloc_scratch(&s, B, o.I, H));
    for (int32_t t = 0; t < T; ++t)
    {
        for (int32_t b = 0; b < B; ++b)
        {
            const int64_t xr = p->time_major ? (int64_t)t * B + b : (int64_t)b * T + t;
            load_row(&inputs[0], xr, o.I, s.x);
            const float *h_prev = t > 0 ? &s.hidden[(size_t)b * H] : NULL;
            if (!p->reset_after && h_prev != NULL)
            {
                for (int32_t u = 0; u < H; ++u)
                {
                    const float r_pre = h16(input_proj(&o, &s, 1, u) + hidden_proj(&o, &s, h_prev, 1, u));
                    s.stage[u] = h16(sigmoid16(r_pre) * h_prev[u]);
                }
            }
            for (int32_t u = 0; u < H; ++u)
            {
                const float z = sigmoid16(h16(input_proj(&o, &s, 0, u) + hidden_proj(&o, &s, h_prev, 0, u)));
                const float xh = input_proj(&o, &s, 2, u);
                float n_pre = 0.0f;
                if (p->reset_after)
                {
                    const float r = sigmoid16(h16(input_proj(&o, &s, 1, u) + hidden_proj(&o, &s, h_prev, 1, u)));
                    n_pre = fma16(r, hidden_proj(&o, &s, h_prev, 2, u), xh);
                }
                else
                {
                    float hh = o.has_bh ? hct_load_f32(o.bh, 2 * (int64_t)H + u) : 0.0f;
                    if (h_prev != NULL)
                    {
                        load_row(o.r, 2 * (int64_t)H + u, H, s.w);
                        hh = h16(hh + prereset_sum(s.w, s.stage, H));
                    }
                    n_pre = h16(xh + hh);
                }
                const float p_term = h16(h16(1.0f - z) * tanh16(n_pre));
                s.h_new[u] = fma16(z, h_prev != NULL ? h_prev[u] : 0.0f, p_term);
                hct_store_f32(&outputs[0], xr * H + u, s.h_new[u]);
            }
            memcpy(&s.hidden[(size_t)b * H], s.h_new, (size_t)H * sizeof(float));
        }
    }
    free_scratch(&s);
    return HCT_OK;
}

/* ------------------------------------------------------------------ SVDF */

int32_t hct_ref_svdf_f16_mve(const HctSvdfFloatParams *p, const HctTensor *inputs, int32_t num_inputs,
                             HctTensor *outputs, int32_t num_outputs)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 5, outputs, num_outputs, 2));
    int64_t n = 0;
    int64_t n_bias = 0;
    for (int i = 0; i < 5; ++i)
    {
        HCT_TRY(hct_check_tensor(&inputs[i], HCT_FLOAT16, i == 3 ? &n_bias : &n));
    }
    HCT_TRY(hct_check_tensor(&outputs[0], HCT_FLOAT16, &n));
    HCT_TRY(hct_check_tensor(&outputs[1], HCT_FLOAT16, &n));
    HCT_TRY(hct_check_float_activation(p->activation_min, p->activation_max));
    const HctTensor *in = &inputs[0];
    const HctTensor *wf = &inputs[1];
    const HctTensor *wt = &inputs[2];
    const HctTensor *st = &inputs[4];
    if (in->rank != 3 || wf->rank != 2 || wt->rank != 2 || st->rank != 3 || outputs[0].rank != 3)
    {
        return HCT_E_SHAPE;
    }
    const int32_t S = in->dims[0], B = in->dims[1], I = in->dims[2], F = wf->dims[0], M = wt->dims[1];
    if (p->rank < 1 || F % p->rank != 0)
    {
        return HCT_E_PARAM;
    }
    const int32_t rank = p->rank;
    const int32_t U = F / rank;
    const int has_bias = n_bias != 0;
    if (wf->dims[1] != I || wt->dims[0] != F || M < 1 || st->dims[0] != B || st->dims[1] != F || st->dims[2] != M ||
        outputs[0].dims[0] != S || outputs[0].dims[1] != B || outputs[0].dims[2] != U ||
        (has_bias && (inputs[3].rank != 1 || inputs[3].dims[0] != U)))
    {
        return HCT_E_SHAPE;
    }
    HCT_TRY(hct_check_same_shape(st, &outputs[1]));
    const size_t n_state = (size_t)B * (size_t)F * (size_t)M;
    float *state = malloc((n_state + 1) * sizeof(float));
    float *x = malloc(((size_t)I + 1) * sizeof(float));
    float *w = malloc(((size_t)(I > M ? I : M) + 1) * sizeof(float));
    float *a = malloc(((size_t)F + 1) * sizeof(float));
    if (state == NULL || x == NULL || w == NULL || a == NULL)
    {
        free(state);
        free(x);
        free(w);
        free(a);
        return HCT_E_SIZE;
    }
    for (size_t i = 0; i < n_state; ++i)
    {
        state[i] = hct_load_f32(st, (int64_t)i);
    }
    const float lo = h16(p->activation_min);
    const float hi = h16(p->activation_max);
    for (int32_t step = 0; step < S; ++step)
    {
        for (int32_t b = 0; b < B; ++b)
        {
            const int64_t x_row = (int64_t)step * B + b;
            load_row(in, x_row, I, x);
            for (int32_t f = 0; f < F; ++f)
            {
                float *mem = &state[((size_t)b * F + f) * M];
                memmove(mem, mem + 1, (size_t)(M - 1) * sizeof(float));
                load_row(wf, f, I, w);
                mem[M - 1] = dot16(x, w, I);
            }
            for (int32_t f = 0; f < F; ++f)
            {
                load_row(wt, f, M, w);
                a[f] = dot16(w, &state[((size_t)b * F + f) * M], M);
            }
            for (int32_t u = 0; u < U; ++u)
            {
                float v = 0.0f;
                if (has_bias && rank == 1)
                {
                    v = h16(a[u] + hct_load_f32(&inputs[3], u));
                }
                else
                {
                    float sum = has_bias ? hct_load_f32(&inputs[3], u) : 0.0f;
                    for (int32_t r = 0; r < rank; ++r)
                    {
                        sum += a[u * rank + r];
                    }
                    v = h16(sum);
                }
                hct_store_f32(&outputs[0], x_row * U + u, fminf(fmaxf(v, lo), hi));
            }
        }
    }
    for (size_t i = 0; i < n_state; ++i)
    {
        hct_store_f32(&outputs[1], (int64_t)i, state[i]);
    }
    free(state);
    free(x);
    free(w);
    free(a);
    return HCT_OK;
}
