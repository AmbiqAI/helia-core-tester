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

#endif /* HCT_REF_INTERNAL_H */
