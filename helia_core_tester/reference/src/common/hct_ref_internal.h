/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Helpers shared by the hct_ref entries. Not part of the ABI: the library is
 * built -fvisibility=hidden and only hct_ref_abi.h declarations are exported.
 */
#ifndef HCT_REF_INTERNAL_H
#define HCT_REF_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "hct_ref_abi.h"

/* Upper bound on any tensor's element count; generation never comes close. */
#define HCT_MAX_ELEMENTS (1LL << 26)

#define HCT_TRY(expr)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        const int32_t hct_status_ = (expr);                                                                            \
        if (hct_status_ != HCT_OK)                                                                                     \
        {                                                                                                              \
            return hct_status_;                                                                                        \
        }                                                                                                              \
    } while (0)

/* ---- tensors ---- */

/* Checks pointer, dtype, rank and dims; *count receives the element count. A
 * zero-element tensor may have a NULL data pointer. */
int32_t hct_check_tensor(const HctTensor *t, int32_t dtype, int64_t *count);

/* Checks the tensor-array pointers and counts of a kernel entry. */
int32_t hct_check_io(
    const HctTensor *inputs, int32_t num_inputs, int32_t want_inputs, const HctTensor *outputs, int32_t num_outputs,
    int32_t want_outputs);

/* Two operands broadcast numpy-style (trailing dims aligned, 1 stretches) onto
 * an output whose shape must equal the broadcast shape. */
typedef struct
{
    int32_t rank;
    int32_t dims[HCT_MAX_RANK];
    int64_t a_strides[HCT_MAX_RANK]; /* 0 along a broadcast dim */
    int64_t b_strides[HCT_MAX_RANK];
    int64_t count;
} HctBroadcast2;

int32_t hct_broadcast2_init(HctBroadcast2 *bc, const HctTensor *a, const HctTensor *b, const HctTensor *out);

/* Element offsets into a and b of output element `index` (row-major). */
void hct_broadcast2_offsets(const HctBroadcast2 *bc, int64_t index, int64_t *a_offset, int64_t *b_offset);

/* ---- fixed point (TFLite reference semantics) ---- */

int32_t hct_saturating_rounding_doubling_high_mul(int32_t a, int32_t b);

/* Round-half-away-from-zero division by 2^exponent, exponent in [0, 31]. */
int32_t hct_rounding_divide_by_pot(int32_t x, int32_t exponent);

/* MultiplyByQuantizedMultiplier, double rounding: SRDHM then RDBPOT. shift in
 * [-31, 30]; a positive shift scales x up first, wrapping as int32 arithmetic does. */
int32_t hct_multiply_by_quantized_multiplier(int32_t x, int32_t multiplier, int32_t shift);

/* CMSIS-NN arm_nn_requantize without CMSIS_NN_USE_SINGLE_ROUNDING (a named variant, not
 * TFLite): the high multiply rounds ties up with no sign-dependent nudge and no saturation,
 * then a rounding right shift. shift in [-31, 30]; a positive shift wraps as int32 math does. */
int32_t hct_cmsis_requantize(int32_t x, int32_t multiplier, int32_t shift);

/* QuantizeMultiplier: real in [0, inf) -> (multiplier in [2^30, 2^31) or 0, shift). */
int32_t hct_quantize_multiplier_impl(double real, int32_t *multiplier, int32_t *shift);

/* As above, for 0 < real < 1 (shift <= 0). */
int32_t hct_quantize_multiplier_smaller_than_one(double real, int32_t *multiplier, int32_t *shift);

/* CalculateActivationRangeQuantized for an int8/int16 tensor. */
int32_t hct_activation_range_quantized(
    int32_t activation, int32_t dtype, float scale, int32_t zero_point, int32_t *act_min, int32_t *act_max);

/* Inclusive integer range of an int8/int16 dtype; HCT_E_DTYPE otherwise. */
int32_t hct_dtype_range(int32_t dtype, int32_t *qmin, int32_t *qmax);

/* ---- float ---- */

/* binary16 <-> binary32. Widening is exact; narrowing rounds to nearest even,
 * overflows to infinity, and keeps NaNs NaN (quieted, sign kept). */
float hct_f16_to_f32(uint16_t h);
uint16_t hct_f32_to_f16(float f);

/* binary64 -> binary16, rounded once to nearest even. */
uint16_t hct_f64_to_f16(double d);

/* Clamp that propagates NaN (std::min(std::max(x, lo), hi) as TFLite applies it). */
static inline float hct_clamp_f32(float x, float lo, float hi)
{
    if (x < lo)
    {
        return lo;
    }
    if (x > hi)
    {
        return hi;
    }
    return x;
}

/* Activation bounds must be ordered and not NaN; infinities leave a side open. */
int32_t hct_check_float_activation(float lo, float hi);

/* ---- element access (dtype already checked by the caller) ---- */

static inline int32_t hct_load_i32(const HctTensor *t, int64_t i)
{
    switch (t->dtype)
    {
    case HCT_INT8:
        return ((const int8_t *)t->data)[i];
    case HCT_INT16:
        return ((const int16_t *)t->data)[i];
    case HCT_BOOL:
        return ((const uint8_t *)t->data)[i];
    default:
        return ((const int32_t *)t->data)[i];
    }
}

/* Stores v, which the caller has already clamped to the dtype's range. */
static inline void hct_store_i32(HctTensor *t, int64_t i, int32_t v)
{
    switch (t->dtype)
    {
    case HCT_INT8:
        ((int8_t *)t->data)[i] = (int8_t)v;
        break;
    case HCT_INT16:
        ((int16_t *)t->data)[i] = (int16_t)v;
        break;
    case HCT_BOOL:
        ((uint8_t *)t->data)[i] = (uint8_t)(v != 0);
        break;
    default:
        ((int32_t *)t->data)[i] = v;
        break;
    }
}

/* float32 element, or a binary16 element widened exactly. */
static inline float hct_load_f32(const HctTensor *t, int64_t i)
{
    return t->dtype == HCT_FLOAT16 ? hct_f16_to_f32(((const uint16_t *)t->data)[i]) : ((const float *)t->data)[i];
}

/* Stores f, rounding once to binary16 for a float16 tensor. */
static inline void hct_store_f32(HctTensor *t, int64_t i, float f)
{
    if (t->dtype == HCT_FLOAT16)
    {
        ((uint16_t *)t->data)[i] = hct_f32_to_f16(f);
    }
    else
    {
        ((float *)t->data)[i] = f;
    }
}

/* One binary16 arithmetic result: the binary32 value rounded to binary16 and back.
 * binary32 has >= 2p + 2 bits for p = 11, so one +, -, * or / computed in binary32
 * and rounded here is exactly the IEEE binary16 operation. */
static inline float hct_round_f16(float f)
{
    return hct_f16_to_f32(hct_f32_to_f16(f));
}

/* ---- int16 lookup tables (TFLite LUTPopulate<int16_t> / LUTLookup) ---- */

#define HCT_LUT_S16_SIZE 513

/* detail::LUTPopulateInt16<float>: 512 segments over the int16 input range, each anchor
 * biased by half the midpoint interpolation error, plus a closing anchor for the last slope. */
void hct_lut_populate_s16(float input_scale, int32_t input_zero_point, float output_scale, int32_t output_zero_point,
                          float (*transform)(float value, const void *params), const void *params,
                          int16_t lut[HCT_LUT_S16_SIZE]);

/* LUTLookup(int16_t): linear interpolation between lut[256 + (v >> 7)] and the next anchor. */
static inline int32_t hct_lut_lookup_s16(int32_t value, const int16_t *lut)
{
    const int32_t index = 256 + (value >> 7);
    const int32_t offset = value & 0x7f;
    const int32_t base = lut[index];
    const int32_t slope = lut[index + 1] - lut[index];
    return (int16_t)(base + ((slope * offset + 64) >> 7));
}

/* Checks a two-input, one-output broadcasting entry and sets up its iteration. */
int32_t hct_binary_setup(const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs,
                         int32_t in_dtype, int32_t out_dtype, HctBroadcast2 *bc);

/* Checks a one-input, one-output entry whose output has the input's shape; *count gets the size. */
int32_t hct_unary_setup(const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs,
                        int32_t in_dtype, int32_t out_dtype, int64_t *count);

#endif /* HCT_REF_INTERNAL_H */
