/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Building blocks of the kernel-faithful float16 variants (ns-cmsis-nn on Helium). Every float16 operation is
 * rounded once: a fused multiply-add through the exact accumulator, an add or multiply as a binary32 operation
 * rounded to binary16 (exact, binary32 has >= 2 * 11 + 2 bits). Float32 steps are plain binary32 arithmetic (the
 * library is built without contraction or excess precision). Not part of the ABI.
 */
#ifndef HCT_F16_MVE_H
#define HCT_F16_MVE_H

#include <stdint.h>

#include "hct_ref_internal.h"

#define HCT_F16_LANES 8
/* ARM_NN_F16_ACC_BLOCK: taps a float16 lane sums before it folds into float32. */
#define HCT_F16_ACC_BLOCK 32
/* One fold block of a row dot product: ARM_NN_F16_ACC_BLOCK taps in each of the eight lanes. */
#define HCT_F16_FOLD_SPAN (HCT_F16_ACC_BLOCK * HCT_F16_LANES)
/* ARM_NN_MVE_F16_MAX_GATHER_STRIDE_{16,8,4}: the largest row length a 16/8/4-lane gather can stride. */
#define HCT_F16_GATHER_MAX_16 (65535 / 15)
#define HCT_F16_GATHER_MAX_8 (65535 / 7)
#define HCT_F16_GATHER_MAX_4 (65535 / 3)

/* arm_nn_mat_mult_nt_n_packed_f16 numerical policies. */
enum
{
    HCT_F16_PACKED_FOLD = 0,  /* the plain entry: from the bias, 32-tap blocks folded past ARM_NN_F16_ACC_BLOCK */
    HCT_F16_PACKED_ACC16 = 1, /* _acc16: from the bias, float16 throughout */
    HCT_F16_PACKED_PRECISE = 2 /* _precise: from zero, 4-tap blocks, bias added in float32 */
};

static inline float hct_h16(float v)
{
    return hct_round_f16(v);
}

/* a * b + c rounded once to binary16. */
int32_t hct_f16_fma(float a, float b, float c, float *out);

/* Elements [k0, k0 + n) of a dot product into fresh float16 lanes, element k0 + k in lane k % 8. */
int32_t hct_f16_lanes(const float *a, const float *b, int32_t k0, int32_t n, float lane[HCT_F16_LANES]);

/* arm_nn_vec_reduce_add_f16: ((0+1)+(2+3))+((4+5)+(6+7)) in float16. */
float hct_f16_reduce8(const float lane[HCT_F16_LANES]);

/* arm_nn_f16_fold_pairs_f32: lanes 2j and 2j+1 widened and added in float32 into pair accumulator j (set when first). */
void hct_f16_fold_pairs(float pairs[4], const float lane[HCT_F16_LANES], int first);

/* arm_nn_f16_pairs_sum_f32: (0+1)+(2+3) in float32. */
float hct_f16_pairs_sum(const float pairs[4]);

/* Row dot product in float16 lanes, reduced in float16, then bias + sum in float16. */
int32_t hct_f16_dot_lanes16(const float *a, const float *b, int32_t K, float bias, float *out);

/* Row dot product, fresh lanes every HCT_F16_FOLD_SPAN elements folded into float32 pairs, summed, bias added in
 * float32, one rounding. */
int32_t hct_f16_dot_fold(const float *a, const float *b, int32_t K, float bias, float *out);

/* One column of a lane-per-column kernel: from `start`, one fused multiply-add per element in order. With fold and
 * K > ARM_NN_F16_ACC_BLOCK, the first 32 elements run from `start`, every further 32 from zero, each block's float16
 * result is widened and added in float32 (the first sets it), one rounding at the end. */
int32_t hct_f16_chain(const float *a, const float *b, int32_t K, int fold, float start, float *out);

/* Columns [0, result) the 16/8/4-lane gather kernels of arm_nn_mat_mult_nt_t_f16 cover; 0 past the eight-lane
 * gather stride. */
int32_t hct_f16_gather_columns(int32_t K, int32_t N);

/* Output column c (of N) of one row of arm_nn_mat_mult_nt_t_f16 (acc16: the _acc16 entry): K < 32 gather columns
 * chain from the bias, everything else is one row dot product, folded when K > 32 and not acc16. */
int32_t hct_f16_nt_t_column(const float *lhs, const float *rhs, int32_t K, int32_t c, int32_t N, float bias,
                            int acc16, float *out);

/* One output column of arm_nn_mat_mult_nt_n_packed_f16 / _acc16 / _precise on Helium (policy above). */
int32_t hct_f16_nt_n_packed_column(const float *lhs, const float *rhs, int32_t K, float bias, int32_t policy,
                                   float *out);

/* The activation clamp on non-NaN values (the generator masks nonfinite lanes). */
float hct_f16_clamp_finite(float v, float lo, float hi);

#endif /* HCT_F16_MVE_H */
