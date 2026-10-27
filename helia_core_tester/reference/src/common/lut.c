/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * TFLite's LUTPopulate<int16_t>, in float as TFLM evaluates it. Only operations IEEE 754
 * rounds correctly (+, -, *, /, sqrtf, roundf) go into a table generated here; a table
 * that needs a transcendental is stored instead, so it cannot drift with the host libm.
 */
#include <math.h>

#include "hct_ref_internal.h"

void hct_lut_populate_s16(float input_scale, int32_t input_zero_point, float output_scale, int32_t output_zero_point,
                          float (*transform)(float value, const void *params), const void *params,
                          int16_t lut[HCT_LUT_S16_SIZE])
{
    const float input_min = input_scale * (float)(INT16_MIN - input_zero_point);
    const float input_max = input_scale * (float)(INT16_MAX - input_zero_point);
    const float output_min = output_scale * (float)(INT16_MIN - output_zero_point);
    const float output_max = output_scale * (float)(INT16_MAX - output_zero_point);
    const int nb_steps = 512;
    const float step = (input_max - input_min) / (float)nb_steps;
    const float half_step = step / 2.0f;
    const float output_scaling_inv = (float)(INT16_MAX - INT16_MIN + 1) / (output_max - output_min);
    const float table_min = (float)INT16_MIN;
    const float table_max = (float)INT16_MAX;
    for (int i = 0; i < nb_steps; i++)
    {
        const float val = transform(input_min + (float)i * step, params);
        const float val_midpoint = transform(input_min + (float)i * step + half_step, params);
        const float val_next = transform(input_min + (float)(i + 1) * step, params);
        const float sample_val = roundf(val * output_scaling_inv);
        const float midpoint_interp_val = roundf((val_next * output_scaling_inv + roundf(val * output_scaling_inv)) / 2.0f);
        const float midpoint_val = roundf(val_midpoint * output_scaling_inv);
        const float midpoint_err = midpoint_interp_val - midpoint_val;
        const float bias = roundf(midpoint_err / 2.0f);
        const float v = sample_val - bias;
        lut[i] = (int16_t)(v < table_min ? table_min : (v > table_max ? table_max : v));
    }
    const float last = roundf(transform(input_max, params) * output_scaling_inv);
    lut[nb_steps] = (int16_t)(last < table_min ? table_min : (last > table_max ? table_max : last));
}
