/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * GRU (Keras conventions; gates update z, reset r, candidate n; zero initial state;
 * h = z*h + (1-z)*n). reset_after: n = tanh(Wn x + b_in + r * (Un h + b_hn)); otherwise
 * n = tanh(Wn x + b_in + Un (r*h) + b_hn). Every matmul-plus-bias is an exact sum rounded once
 * to the element type; sigmoid/tanh are binary64 rounded once; the remaining elementwise
 * steps are IEEE operations in the element type.
 *
 * SVDF: input [S, B, I] runs S steps of TFLM's EvalIntegerSvdfReference / EvalFloatSvdfReference,
 * output [S, B, U] holds every step's output and state_out the final state. Each step: the state
 * [B, F, M] shifts left by one step, the feature projection fills the newest slot, the time
 * weights reduce each filter's memory, rank filters sum per unit with the bias. Integer: the
 * projection is rescaled into the state range (int16 state is TFLite's; int8 state with int8
 * time weights is ns-cmsis-nn's arm_svdf_s8, the same arithmetic), sums are int32 as in TFLM.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hct_ref_internal.h"

static float round_to(int32_t dtype, double v)
{
    return dtype == HCT_FLOAT16 ? hct_f16_to_f32(hct_f64_to_f16(v)) : (float)v;
}

/* --------------------------------------------------------------------- GRU */

static int32_t check_bias(const HctTensor *b, int32_t dtype, int32_t hidden, int *present)
{
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(b, dtype, &n));
    *present = n != 0;
    if (*present && (b->rank != 2 || b->dims[0] != 3 || b->dims[1] != hidden))
    {
        return HCT_E_SHAPE;
    }
    return HCT_OK;
}

/* Exact x[0..len) . w[w_row] (+ bias b[bias_index] when present) rounded once. */
static int32_t dot(int32_t dtype, const float *x, int32_t len, const HctTensor *w, int64_t w_row, const HctTensor *b1,
                   int has_b1, const HctTensor *b2, int has_b2, int64_t bias_index, HctXsum *acc_in, float *out)
{
    HctXsum acc;
    if (acc_in)
    {
        acc = *acc_in;
    }
    else
    {
        hct_xsum_init(&acc);
    }
    for (int32_t k = 0; k < len; ++k)
    {
        hct_xsum_add_product(&acc, x[k], hct_load_f32(w, w_row * len + k));
    }
    if (has_b1)
    {
        hct_xsum_add(&acc, (double)hct_load_f32(b1, bias_index));
    }
    if (has_b2)
    {
        hct_xsum_add(&acc, (double)hct_load_f32(b2, bias_index));
    }
    if (out == NULL)
    {
        *acc_in = acc;
        return HCT_OK;
    }
    double v = 0.0;
    HCT_TRY(hct_xsum_round(&acc, 1, dtype, &v));
    *out = (float)v;
    return HCT_OK;
}

static int32_t gru(const HctGruParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                   int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 5, outputs, num_outputs, 1));
    int64_t n = 0;
    for (int i = 0; i < 3; ++i)
    {
        HCT_TRY(hct_check_tensor(&inputs[i], dtype, &n));
    }
    HCT_TRY(hct_check_tensor(&outputs[0], dtype, &n));
    if ((p->time_major != 0 && p->time_major != 1) || (p->reset_after != 0 && p->reset_after != 1))
    {
        return HCT_E_PARAM;
    }
    const HctTensor *in = &inputs[0];
    const HctTensor *w = &inputs[1];
    const HctTensor *r = &inputs[2];
    HctTensor *out = &outputs[0];
    if (in->rank != 3 || w->rank != 3 || r->rank != 3 || out->rank != 3)
    {
        return HCT_E_SHAPE;
    }
    const int32_t B = p->time_major ? in->dims[1] : in->dims[0];
    const int32_t T = p->time_major ? in->dims[0] : in->dims[1];
    const int32_t I = in->dims[2];
    const int32_t H = w->dims[1];
    if (H < 1 || w->dims[0] != 3 || w->dims[2] != I || r->dims[0] != 3 || r->dims[1] != H || r->dims[2] != H ||
        out->dims[0] != in->dims[0] || out->dims[1] != in->dims[1] || out->dims[2] != H)
    {
        return HCT_E_SHAPE;
    }
    int has_bi = 0;
    int has_bh = 0;
    HCT_TRY(check_bias(&inputs[3], dtype, H, &has_bi));
    HCT_TRY(check_bias(&inputs[4], dtype, H, &has_bh));
    float *hidden = calloc((size_t)B * (size_t)H + 1, sizeof(float));
    float *x = malloc(((size_t)I + 1) * sizeof(float));
    float *z = malloc((size_t)H * sizeof(float));
    float *rg = malloc((size_t)H * sizeof(float));
    float *rh = malloc((size_t)H * sizeof(float));
    if (hidden == NULL || x == NULL || z == NULL || rg == NULL || rh == NULL)
    {
        free(hidden);
        free(x);
        free(z);
        free(rg);
        free(rh);
        return HCT_E_SIZE;
    }
    int32_t status = HCT_OK;
    for (int32_t t = 0; t < T && status == HCT_OK; ++t)
    {
        for (int32_t b = 0; b < B && status == HCT_OK; ++b)
        {
            const int64_t xr = p->time_major ? (int64_t)t * B + b : (int64_t)b * T + t;
            for (int32_t k = 0; k < I; ++k)
            {
                x[k] = hct_load_f32(in, xr * I + k);
            }
            float *h = &hidden[(size_t)b * H];
            for (int32_t u = 0; u < H && status == HCT_OK; ++u)
            {
                HctXsum acc;
                float pre = 0.0f;
                for (int g = 0; g < 2 && status == HCT_OK; ++g)
                {
                    hct_xsum_init(&acc);
                    status = dot(dtype, x, I, w, (int64_t)g * H + u, &inputs[3], has_bi, NULL, 0, (int64_t)g * H + u,
                                 &acc, NULL);
                    if (status == HCT_OK)
                    {
                        status = dot(dtype, h, H, r, (int64_t)g * H + u, &inputs[4], has_bh, NULL, 0,
                                     (int64_t)g * H + u, &acc, &pre);
                    }
                    (g == 0 ? z : rg)[u] = round_to(dtype, 1.0 / (1.0 + exp(-(double)pre)));
                }
            }
            for (int32_t u = 0; u < H; ++u)
            {
                rh[u] = round_to(dtype, (double)rg[u] * h[u]);
            }
            float *nv = malloc((size_t)H * sizeof(float));
            if (nv == NULL)
            {
                status = HCT_E_SIZE;
                break;
            }
            for (int32_t u = 0; u < H && status == HCT_OK; ++u)
            {
                const int64_t gi = 2 * (int64_t)H + u;
                float pre = 0.0f;
                if (p->reset_after)
                {
                    float a = 0.0f;
                    float c = 0.0f;
                    status = dot(dtype, x, I, w, gi, &inputs[3], has_bi, NULL, 0, gi, NULL, &a);
                    if (status == HCT_OK)
                    {
                        status = dot(dtype, h, H, r, gi, &inputs[4], has_bh, NULL, 0, gi, NULL, &c);
                    }
                    pre = round_to(dtype, (double)a + round_to(dtype, (double)rg[u] * c));
                }
                else
                {
                    HctXsum acc;
                    hct_xsum_init(&acc);
                    status = dot(dtype, x, I, w, gi, &inputs[3], has_bi, NULL, 0, gi, &acc, NULL);
                    if (status == HCT_OK)
                    {
                        status = dot(dtype, rh, H, r, gi, &inputs[4], has_bh, NULL, 0, gi, &acc, &pre);
                    }
                }
                nv[u] = round_to(dtype, tanh((double)pre));
            }
            for (int32_t u = 0; u < H && status == HCT_OK; ++u)
            {
                const float keep = round_to(dtype, (double)z[u] * h[u]);
                const float one_minus = round_to(dtype, 1.0 - (double)z[u]);
                h[u] = round_to(dtype, (double)keep + round_to(dtype, (double)one_minus * nv[u]));
                hct_store_f32(out, xr * H + u, h[u]);
            }
            free(nv);
        }
    }
    free(hidden);
    free(x);
    free(z);
    free(rg);
    free(rh);
    return status;
}

int32_t hct_ref_gru_f32(const HctGruParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs)
{
    return gru(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_gru_f16(const HctGruParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs)
{
    return gru(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}

/* -------------------------------------------------------------------- SVDF */

typedef struct
{
    int32_t S, B, I, F, M, U, rank;
    int has_bias;
} SvdfShape;

static int32_t svdf_shape(const HctTensor *inputs, const HctTensor *outputs, int32_t rank, int32_t in_dtype,
                          int32_t wf_dtype, int32_t state_dtype, int32_t bias_dtype, int32_t out_dtype, SvdfShape *s)
{
    int64_t n = 0;
    int64_t n_bias = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], in_dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], wf_dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[2], state_dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[3], bias_dtype, &n_bias));
    HCT_TRY(hct_check_tensor(&inputs[4], state_dtype, &n));
    HCT_TRY(hct_check_tensor(&outputs[0], out_dtype, &n));
    HCT_TRY(hct_check_tensor(&outputs[1], state_dtype, &n));
    const HctTensor *in = &inputs[0];
    const HctTensor *wf = &inputs[1];
    const HctTensor *wt = &inputs[2];
    const HctTensor *st = &inputs[4];
    if (in->rank != 3 || wf->rank != 2 || wt->rank != 2 || st->rank != 3 || outputs[0].rank != 3)
    {
        return HCT_E_SHAPE;
    }
    s->S = in->dims[0];
    s->B = in->dims[1];
    s->I = in->dims[2];
    s->F = wf->dims[0];
    s->M = wt->dims[1];
    if (rank < 1 || s->F % rank != 0)
    {
        return HCT_E_PARAM;
    }
    s->rank = rank;
    s->U = s->F / rank;
    s->has_bias = n_bias != 0;
    if (wf->dims[1] != s->I || wt->dims[0] != s->F || s->M < 1 || st->dims[0] != s->B || st->dims[1] != s->F ||
        st->dims[2] != s->M || outputs[0].dims[0] != s->S || outputs[0].dims[1] != s->B || outputs[0].dims[2] != s->U ||
        (s->has_bias && (inputs[3].rank != 1 || inputs[3].dims[0] != s->U)))
    {
        return HCT_E_SHAPE;
    }
    return hct_check_same_shape(st, &outputs[1]);
}

static int32_t svdf_int(const HctSvdfParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs, int32_t state_dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 5, outputs, num_outputs, 2));
    SvdfShape s;
    HCT_TRY(svdf_shape(inputs, outputs, p->rank, HCT_INT8, HCT_INT8, state_dtype, HCT_INT32, HCT_INT8, &s));
    int32_t smin = 0;
    int32_t smax = 0;
    HCT_TRY(hct_dtype_range(state_dtype, &smin, &smax));
    if (p->input_offset < -127 || p->input_offset > 128 || p->output_offset < -128 || p->output_offset > 127 ||
        p->input_multiplier < 0 || p->output_multiplier < 0 || p->input_shift < -31 || p->input_shift > 30 ||
        p->output_shift < -31 || p->output_shift > 30 || p->input_activation_min > p->input_activation_max ||
        p->input_activation_min < smin || p->input_activation_max > smax ||
        p->output_activation_min > p->output_activation_max || p->output_activation_min < -128 ||
        p->output_activation_max > 127)
    {
        return HCT_E_PARAM;
    }
    const HctTensor *st = &inputs[4];
    HctTensor *so = &outputs[1];
    const int64_t n_state = (int64_t)s.B * s.F * s.M;
    for (int64_t i = 0; i < n_state; ++i)
    {
        hct_store_i32(so, i, hct_load_i32(st, i));
    }
    for (int32_t step = 0; step < s.S; ++step)
    {
        for (int32_t b = 0; b < s.B; ++b)
        {
            const int64_t x_row = (int64_t)step * s.B + b;
            for (int32_t f = 0; f < s.F; ++f)
            {
                const int64_t base = ((int64_t)b * s.F + f) * s.M;
                for (int32_t m = 0; m + 1 < s.M; ++m)
                {
                    hct_store_i32(so, base + m, hct_load_i32(so, base + m + 1));
                }
                int64_t acc = 0;
                for (int32_t i = 0; i < s.I; ++i)
                {
                    acc += (int64_t)hct_load_i32(&inputs[1], (int64_t)f * s.I + i) *
                           ((int64_t)hct_load_i32(&inputs[0], x_row * s.I + i) + p->input_offset);
                }
                if (acc < INT32_MIN || acc > INT32_MAX)
                {
                    return HCT_E_PARAM;
                }
                int32_t v = hct_multiply_by_quantized_multiplier((int32_t)acc, p->input_multiplier, p->input_shift);
                v = v < p->input_activation_min ? p->input_activation_min
                                                : (v > p->input_activation_max ? p->input_activation_max : v);
                hct_store_i32(so, base + s.M - 1, v);
            }
            for (int32_t u = 0; u < s.U; ++u)
            {
                int64_t total = s.has_bias ? hct_load_i32(&inputs[3], u) : 0;
                for (int32_t r = 0; r < s.rank; ++r)
                {
                    const int32_t f = u * s.rank + r;
                    int64_t t = 0;
                    for (int32_t m = 0; m < s.M; ++m)
                    {
                        t += (int64_t)hct_load_i32(&inputs[2], (int64_t)f * s.M + m) *
                             hct_load_i32(so, ((int64_t)b * s.F + f) * s.M + m);
                    }
                    if (t < INT32_MIN || t > INT32_MAX)
                    {
                        return HCT_E_PARAM;
                    }
                    total += t;
                }
                if (total < INT32_MIN || total > INT32_MAX)
                {
                    return HCT_E_PARAM;
                }
                int64_t o = (int64_t)hct_multiply_by_quantized_multiplier((int32_t)total, p->output_multiplier,
                                                                          p->output_shift) +
                            p->output_offset;
                o = o < p->output_activation_min ? p->output_activation_min
                                                 : (o > p->output_activation_max ? p->output_activation_max : o);
                hct_store_i32(&outputs[0], x_row * s.U + u, (int32_t)o);
            }
        }
    }
    return HCT_OK;
}

int32_t hct_ref_svdf_s8(const HctSvdfParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs)
{
    return svdf_int(params, inputs, num_inputs, outputs, num_outputs, HCT_INT8);
}

int32_t hct_ref_svdf_state_s16_s8(const HctSvdfParams *params, const HctTensor *inputs, int32_t num_inputs,
                                  HctTensor *outputs, int32_t num_outputs)
{
    return svdf_int(params, inputs, num_inputs, outputs, num_outputs, HCT_INT16);
}

static int32_t svdf_float(const HctSvdfFloatParams *p, const HctTensor *inputs, int32_t num_inputs,
                          HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 5, outputs, num_outputs, 2));
    SvdfShape s;
    HCT_TRY(svdf_shape(inputs, outputs, p->rank, dtype, dtype, dtype, dtype, dtype, &s));
    HCT_TRY(hct_check_float_activation(p->activation_min, p->activation_max));
    const HctTensor *st = &inputs[4];
    HctTensor *so = &outputs[1];
    const int64_t n_state = (int64_t)s.B * s.F * s.M;
    for (int64_t i = 0; i < n_state; ++i)
    {
        hct_store_f32(so, i, hct_load_f32(st, i));
    }
    for (int32_t step = 0; step < s.S; ++step)
    {
        for (int32_t b = 0; b < s.B; ++b)
        {
            const int64_t x_row = (int64_t)step * s.B + b;
            for (int32_t f = 0; f < s.F; ++f)
            {
                const int64_t base = ((int64_t)b * s.F + f) * s.M;
                for (int32_t m = 0; m + 1 < s.M; ++m)
                {
                    hct_store_f32(so, base + m, hct_load_f32(so, base + m + 1));
                }
                HctXsum acc;
                hct_xsum_init(&acc);
                for (int32_t i = 0; i < s.I; ++i)
                {
                    hct_xsum_add_product(&acc, hct_load_f32(&inputs[1], (int64_t)f * s.I + i),
                                         hct_load_f32(&inputs[0], x_row * s.I + i));
                }
                HCT_TRY(hct_store_xsum(so, base + s.M - 1, &acc, 1, -INFINITY, INFINITY));
            }
            for (int32_t u = 0; u < s.U; ++u)
            {
                HctXsum total;
                hct_xsum_init(&total);
                if (s.has_bias)
                {
                    hct_xsum_add(&total, (double)hct_load_f32(&inputs[3], u));
                }
                for (int32_t r = 0; r < s.rank; ++r)
                {
                    const int32_t f = u * s.rank + r;
                    HctXsum t;
                    hct_xsum_init(&t);
                    for (int32_t m = 0; m < s.M; ++m)
                    {
                        hct_xsum_add_product(&t, hct_load_f32(&inputs[2], (int64_t)f * s.M + m),
                                             hct_load_f32(so, ((int64_t)b * s.F + f) * s.M + m));
                    }
                    double tv = 0.0;
                    HCT_TRY(hct_xsum_round(&t, 1, dtype, &tv)); /* each filter's time reduction is a tensor */
                    hct_xsum_add(&total, tv);
                }
                HCT_TRY(hct_store_xsum(&outputs[0], x_row * s.U + u, &total, 1, p->activation_min,
                                       p->activation_max));
            }
        }
    }
    return HCT_OK;
}

int32_t hct_ref_svdf_f32(const HctSvdfFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return svdf_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_svdf_f16(const HctSvdfFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return svdf_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
