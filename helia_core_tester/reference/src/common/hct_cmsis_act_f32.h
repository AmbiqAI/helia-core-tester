/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * ns-cmsis-nn's float32 sigmoid and tanh approximations per target, shared by the kernel-faithful variants.
 * target 0 (Helium) and 1 (scalar with an FPU, as GCC -Ofast contracts it) fuse the table interpolation;
 * target 2 (soft float) fuses nothing. Not part of the ABI.
 */
#ifndef HCT_CMSIS_ACT_F32_H
#define HCT_CMSIS_ACT_F32_H

#include <stdint.h>

/* arm_nn_sigmoid_scalar_f32: e = exp(-|x|) from the 256-segment exp2 table, then 1 / (1 + e) or e / (1 + e). */
int32_t hct_cmsis_sigmoid_f32(float x, int32_t target, float *out);

/* arm_nn_tanh_scalar_ref_f32 / arm_nn_vtanh_lut_direct_mve_f32: the 384-segment table over [0, 6), linear
 * interpolation, +-1 from |x| >= 6, the sign copied from x; a NaN is returned as is. */
int32_t hct_cmsis_tanh_f32(float x, int32_t target, float *out);

#endif /* HCT_CMSIS_ACT_F32_H */
