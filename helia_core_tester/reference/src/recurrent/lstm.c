/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unidirectional LSTM, no peephole/projection/layer norm, zero initial cell state and a hidden
 * state starting at the output zero point (real zero), output = every step's hidden state.
 *
 * Integer: TFLM's lstm_eval for int8x8_16 (int8 activations, int32 bias) and int16x8_64 (int16
 * activations, int64 bias). Per gate, the input and recurrent matmuls are FullyConnected to
 * int16 at the gate scale 2^-12 (int32 accumulator with double-rounding
 * MultiplyByQuantizedMultiplier, or int64 with the reduced-multiplier one), added saturating,
 * then Logistic or Tanh (reference_integer_ops on int16, input multiplier 3). The cell is
 * f*c + i*g through int16 Mul/Add with clipping; the hidden state is o * tanh(c) through Mul to
 * the output type, tanh taking the cell scale power.
 *
 * Float: gate pre-activations are exact sums rounded once to the element type, sigmoid/tanh
 * are evaluated in binary64 and rounded once, and the cell/hidden updates are IEEE operations
 * in the element type in the order f*c + i*g, o * tanh(c) (no contraction).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hct_ref_internal.h"

typedef struct
{
    int32_t batch, steps, input_size, hidden;
} LstmShape;

/* input [B,T,I] (or [T,B,I] time-major), weights [G,H,I], recurrent [G,H,H], bias [G,H] or empty,
 * output [B,T,H] / [T,B,H]. */
static int32_t lstm_shape(const HctTensor *inputs, const HctTensor *out, int32_t gates, int32_t time_major,
                          int32_t in_dtype, int32_t w_dtype, int32_t bias_dtype, int *has_bias, LstmShape *s)
{
    int64_t n = 0;
    HCT_TRY(hct_check_tensor(&inputs[0], in_dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[1], w_dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[2], w_dtype, &n));
    HCT_TRY(hct_check_tensor(&inputs[3], bias_dtype, &n));
    HCT_TRY(hct_check_tensor(out, in_dtype, &n));
    if (time_major != 0 && time_major != 1)
    {
        return HCT_E_PARAM;
    }
    const HctTensor *in = &inputs[0];
    const HctTensor *w = &inputs[1];
    const HctTensor *r = &inputs[2];
    if (in->rank != 3 || w->rank != 3 || r->rank != 3 || out->rank != 3)
    {
        return HCT_E_SHAPE;
    }
    s->batch = time_major ? in->dims[1] : in->dims[0];
    s->steps = time_major ? in->dims[0] : in->dims[1];
    s->input_size = in->dims[2];
    s->hidden = w->dims[1];
    if (w->dims[0] != gates || w->dims[2] != s->input_size || r->dims[0] != gates || r->dims[1] != s->hidden ||
        r->dims[2] != s->hidden || out->dims[0] != in->dims[0] || out->dims[1] != in->dims[1] ||
        out->dims[2] != s->hidden || s->hidden < 1)
    {
        return HCT_E_SHAPE;
    }
    const HctTensor *b = &inputs[3];
    *has_bias = b->rank != 1 || b->dims[0] != 0;
    if (*has_bias && (b->rank != 2 || b->dims[0] != gates || b->dims[1] != s->hidden))
    {
        return HCT_E_SHAPE;
    }
    return HCT_OK;
}

/* Row offset of (batch b, step t) in a [B,T,X] or [T,B,X] tensor. */
static int64_t row(const LstmShape *s, int32_t time_major, int32_t b, int32_t t)
{
    return time_major ? (int64_t)t * s->batch + b : (int64_t)b * s->steps + t;
}

static int32_t sat16(int64_t v)
{
    return v < INT16_MIN ? INT16_MIN : (v > INT16_MAX ? INT16_MAX : (int32_t)v);
}

/* ------------------------------------------------------------------ integer */

static int32_t check_mult(int32_t m, int32_t shift, int32_t max_shift)
{
    return (m >= 0 && shift >= -31 && shift <= max_shift) ? HCT_OK : HCT_E_PARAM;
}

/* FullyConnected to int16: sum (x + offset) * w (+ bias), requantized, saturated to int16. */
static int32_t gate_fc(int is16, const HctTensor *x, int64_t x_row, int32_t x_len, int32_t offset, const int8_t *w,
                       const HctTensor *bias, int has_bias, int64_t bias_index, int32_t mult, int32_t shift,
                       int32_t *out)
{
    int64_t acc = 0;
    for (int32_t k = 0; k < x_len; ++k)
    {
        acc += (int64_t)w[k] * ((int64_t)hct_load_i32(x, x_row * x_len + k) + offset);
    }
    if (has_bias)
    {
        acc += hct_load_i64(bias, bias_index);
    }
    int64_t v = 0;
    if (is16)
    {
        if (acc < -((int64_t)1 << 47) || acc >= ((int64_t)1 << 47))
        {
            return HCT_E_PARAM;
        }
        v = hct_multiply_by_quantized_multiplier_64(acc, mult, shift);
    }
    else
    {
        if (acc < INT32_MIN || acc > INT32_MAX)
        {
            return HCT_E_PARAM; /* TFLM accumulates in int32 */
        }
        v = hct_multiply_by_quantized_multiplier((int32_t)acc, mult, shift);
    }
    *out = sat16(v);
    return HCT_OK;
}

static int32_t lstm_int(const HctLstmParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs, int is16)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 4, outputs, num_outputs, 1));
    const int32_t act = is16 ? HCT_INT16 : HCT_INT8;
    LstmShape s;
    int has_bias = 0;
    HCT_TRY(lstm_shape(inputs, &outputs[0], 4, p->time_major, act, HCT_INT8, is16 ? HCT_INT64 : HCT_INT32, &has_bias,
                       &s));
    int32_t qmin = 0;
    int32_t qmax = 0;
    HCT_TRY(hct_dtype_range(act, &qmin, &qmax));
    const int32_t max_shift = is16 ? 7 : 30;
    /* 4 gates x {input multiplier, input shift, hidden multiplier, hidden shift}. */
    const int32_t gm[16] = {p->forget_input_multiplier, p->forget_input_shift, p->forget_hidden_multiplier,
                            p->forget_hidden_shift,     p->input_input_multiplier, p->input_input_shift,
                            p->input_hidden_multiplier, p->input_hidden_shift,     p->cell_input_multiplier,
                            p->cell_input_shift,        p->cell_hidden_multiplier, p->cell_hidden_shift,
                            p->output_input_multiplier, p->output_input_shift,     p->output_hidden_multiplier,
                            p->output_hidden_shift};
    for (int g = 0; g < 4; ++g)
    {
        HCT_TRY(check_mult(gm[4 * g], gm[4 * g + 1], max_shift));
        HCT_TRY(check_mult(gm[4 * g + 2], gm[4 * g + 3], max_shift));
    }
    HCT_TRY(check_mult(p->forget_to_cell_multiplier, p->forget_to_cell_shift, 30));
    HCT_TRY(check_mult(p->input_to_cell_multiplier, p->input_to_cell_shift, 30));
    HCT_TRY(check_mult(p->hidden_multiplier, p->hidden_shift, 30));
    if (p->output_offset < qmin || p->output_offset > qmax || p->input_offset < -qmax || p->input_offset > -qmin ||
        p->cell_clip < 0 || p->cell_clip > INT16_MAX || p->cell_scale_power < -30 || p->cell_scale_power > 0)
    {
        return HCT_E_PARAM;
    }
    /* TFLM's Tanh(cell_state_scale_power): left shift (15 + power) - 3, or a right shift with
     * multiplier 3 when negative. */
    int32_t cell_mult = 0;
    int32_t cell_shift = 15 + p->cell_scale_power - 3;
    if (cell_shift < 0)
    {
        cell_shift = -cell_shift;
        cell_mult = 3;
    }
    else
    {
        if (cell_shift > 13)
        {
            return HCT_E_PARAM;
        }
        cell_mult = 3 << cell_shift;
        cell_shift = 0;
    }
    const int32_t H = s.hidden;
    const size_t n_state = (size_t)s.batch * (size_t)H;
    int32_t *cell = calloc(n_state ? n_state : 1, sizeof(int32_t));
    int32_t *hidden = malloc((n_state ? n_state : 1) * sizeof(int32_t));
    int32_t *gate = malloc((size_t)(4 * (H ? H : 1)) * sizeof(int32_t));
    if (cell == NULL || hidden == NULL || gate == NULL)
    {
        free(cell);
        free(hidden);
        free(gate);
        return HCT_E_SIZE;
    }
    for (size_t i = 0; i < n_state; ++i)
    {
        hidden[i] = p->output_offset;
    }
    HctTensor hidden_t = {HCT_INT32, 1, {(int32_t)n_state}, hidden};
    const int8_t *w = (const int8_t *)inputs[1].data;
    const int8_t *rw = (const int8_t *)inputs[2].data;
    int32_t status = HCT_OK;
    for (int32_t t = 0; t < s.steps && status == HCT_OK; ++t)
    {
        for (int32_t b = 0; b < s.batch && status == HCT_OK; ++b)
        {
            const int64_t xr = row(&s, p->time_major, b, t);
            for (int g = 0; g < 4 && status == HCT_OK; ++g)
            {
                for (int32_t h = 0; h < H && status == HCT_OK; ++h)
                {
                    int32_t in_part = 0;
                    int32_t rec_part = 0;
                    status = gate_fc(is16, &inputs[0], xr, s.input_size, p->input_offset,
                                     w + ((int64_t)g * H + h) * s.input_size, &inputs[3], has_bias, (int64_t)g * H + h,
                                     gm[4 * g], gm[4 * g + 1], &in_part);
                    if (status == HCT_OK)
                    {
                        /* The recurrent matmul has no bias; its input is the stored hidden state. */
                        status = gate_fc(is16, &hidden_t, b, H, -p->output_offset, rw + ((int64_t)g * H + h) * H,
                                         NULL, 0, 0, gm[4 * g + 2], gm[4 * g + 3], &rec_part);
                    }
                    /* Sigmoid for forget, input, output; tanh (power -12: multiplier 3) for the cell gate. */
                    gate[g * H + h] = hct_tanh_logistic_s16_value(sat16((int64_t)in_part + rec_part), 3, 0, g == 2);
                }
            }
            for (int32_t h = 0; h < H && status == HCT_OK; ++h)
            {
                int32_t *c = &cell[(size_t)b * H + h];
                const int32_t fc = sat16(hct_multiply_by_quantized_multiplier(gate[h] * *c, p->forget_to_cell_multiplier,
                                                                              p->forget_to_cell_shift));
                const int32_t ig = sat16(hct_multiply_by_quantized_multiplier(
                    gate[H + h] * gate[2 * H + h], p->input_to_cell_multiplier, p->input_to_cell_shift));
                int32_t cv = sat16((int64_t)fc + ig);
                if (p->cell_clip > 0)
                {
                    cv = cv < -p->cell_clip ? -p->cell_clip : (cv > p->cell_clip ? p->cell_clip : cv);
                }
                *c = cv;
                const int32_t tc = hct_tanh_logistic_s16_value(cv, cell_mult, cell_shift, 1);
                int64_t hv =
                    (int64_t)hct_multiply_by_quantized_multiplier(gate[3 * H + h] * tc, p->hidden_multiplier,
                                                                  p->hidden_shift) +
                    p->output_offset;
                hv = hv < qmin ? qmin : (hv > qmax ? qmax : hv);
                hidden[(size_t)b * H + h] = (int32_t)hv;
                hct_store_i32(&outputs[0], xr * H + h, (int32_t)hv);
            }
        }
    }
    free(cell);
    free(hidden);
    free(gate);
    return status;
}

int32_t hct_ref_lstm_s8(const HctLstmParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs)
{
    return lstm_int(params, inputs, num_inputs, outputs, num_outputs, 0);
}

int32_t hct_ref_lstm_s16(const HctLstmParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return lstm_int(params, inputs, num_inputs, outputs, num_outputs, 1);
}

/* -------------------------------------------------------------------- float */

/* One element-type rounding of a binary64 value (binary16 directly, not through binary32). */
static float round_to(int32_t dtype, double v)
{
    return dtype == HCT_FLOAT16 ? hct_f16_to_f32(hct_f64_to_f16(v)) : (float)v;
}

static float sigmoid_r(int32_t dtype, float x)
{
    return round_to(dtype, 1.0 / (1.0 + exp(-(double)x)));
}

static float tanh_r(int32_t dtype, float x)
{
    return round_to(dtype, tanh((double)x));
}

static int32_t lstm_float(const HctLstmFloatParams *p, const HctTensor *inputs, int32_t num_inputs,
                          HctTensor *outputs, int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 4, outputs, num_outputs, 1));
    LstmShape s;
    int has_bias = 0;
    HCT_TRY(lstm_shape(inputs, &outputs[0], 4, p->time_major, dtype, dtype, dtype, &has_bias, &s));
    if (!(p->cell_clip >= 0.0f) || isinf(p->cell_clip))
    {
        return HCT_E_PARAM;
    }
    const float clip = round_to(dtype, p->cell_clip);
    const int32_t H = s.hidden;
    const size_t n_state = (size_t)s.batch * (size_t)H;
    float *cell = calloc(n_state ? n_state : 1, sizeof(float));
    float *hidden = calloc(n_state ? n_state : 1, sizeof(float));
    float *gate = malloc((size_t)(4 * H) * sizeof(float));
    if (cell == NULL || hidden == NULL || gate == NULL)
    {
        free(cell);
        free(hidden);
        free(gate);
        return HCT_E_SIZE;
    }
    int32_t status = HCT_OK;
    for (int32_t t = 0; t < s.steps && status == HCT_OK; ++t)
    {
        for (int32_t b = 0; b < s.batch && status == HCT_OK; ++b)
        {
            const int64_t xr = row(&s, p->time_major, b, t);
            for (int g = 0; g < 4 && status == HCT_OK; ++g)
            {
                for (int32_t h = 0; h < H && status == HCT_OK; ++h)
                {
                    /* One exact sum of both matmuls and the bias. */
                    HctXsum acc;
                    hct_xsum_init(&acc);
                    for (int32_t k = 0; k < s.input_size; ++k)
                    {
                        hct_xsum_add_product(&acc, hct_load_f32(&inputs[0], xr * s.input_size + k),
                                             hct_load_f32(&inputs[1], ((int64_t)g * H + h) * s.input_size + k));
                    }
                    for (int32_t k = 0; k < H; ++k)
                    {
                        hct_xsum_add_product(&acc, hidden[(size_t)b * H + k],
                                             hct_load_f32(&inputs[2], ((int64_t)g * H + h) * H + k));
                    }
                    if (has_bias)
                    {
                        hct_xsum_add(&acc, (double)hct_load_f32(&inputs[3], (int64_t)g * H + h));
                    }
                    double pre = 0.0;
                    status = hct_xsum_round(&acc, 1, dtype, &pre);
                    gate[g * H + h] = g == 2 ? tanh_r(dtype, (float)pre) : sigmoid_r(dtype, (float)pre);
                }
            }
            for (int32_t h = 0; h < H && status == HCT_OK; ++h)
            {
                float *c = &cell[(size_t)b * H + h];
                const float fc = round_to(dtype, (double)gate[h] * *c);
                const float ig = round_to(dtype, (double)gate[H + h] * gate[2 * H + h]);
                float cv = round_to(dtype, (double)fc + ig);
                if (clip > 0.0f)
                {
                    cv = hct_clamp_f32(cv, -clip, clip);
                }
                *c = cv;
                const float hv = round_to(dtype, (double)gate[3 * H + h] * tanh_r(dtype, cv));
                hidden[(size_t)b * H + h] = hv;
                hct_store_f32(&outputs[0], xr * H + h, hv);
            }
        }
    }
    free(cell);
    free(hidden);
    free(gate);
    return status;
}

int32_t hct_ref_lstm_f32(const HctLstmFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return lstm_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_lstm_f16(const HctLstmFloatParams *params, const HctTensor *inputs, int32_t num_inputs,
                         HctTensor *outputs, int32_t num_outputs)
{
    return lstm_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
