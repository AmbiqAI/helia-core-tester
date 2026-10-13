/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated from spec/entries.yaml by helia_core_tester.generation.reference.abi.
 * Do not edit; run `python -m helia_core_tester.generation.reference.abi --write`.
 */
#ifndef HCT_REF_ABI_H
#define HCT_REF_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#define HCT_REF_ABI_VERSION 1
#define HCT_MAX_RANK 8

typedef enum
{
    HCT_INT8 = 1,
    HCT_INT16 = 2,
    HCT_INT32 = 3,
    HCT_INT64 = 4,
    HCT_FLOAT32 = 5,
    HCT_FLOAT16 = 6,
    HCT_BOOL = 7
} HctDtype;

typedef enum
{
    HCT_OK = 0,
    HCT_E_NULL = 1,
    HCT_E_COUNT = 2,
    HCT_E_DTYPE = 3,
    HCT_E_SHAPE = 4,
    HCT_E_PARAM = 5,
    HCT_E_SIZE = 6
} HctStatus;

typedef enum
{
    HCT_ACT_NONE = 0,
    HCT_ACT_RELU = 1,
    HCT_ACT_RELU6 = 2,
    HCT_ACT_RELU_N1_TO_1 = 3
} HctActivation;

/* A tensor: dtype is an HctDtype, dims[0..rank-1] row-major (NHWC for 4-D), data
 * points at rank-product elements (1 for rank 0). */
typedef struct
{
    int32_t dtype;
    int32_t rank;
    int32_t dims[HCT_MAX_RANK];
    void *data;
} HctTensor;

/* A real multiplier, as TFLite's QuantizeMultiplier takes it. */
typedef struct
{
    double real_multiplier;
} HctQuantizeMultiplierIn;

typedef struct
{
    int32_t multiplier;
    int32_t shift;
} HctQuantizeMultiplierOut;

/* CalculateActivationRangeQuantized. activation is an HctActivation value. */
typedef struct
{
    int32_t activation;
    int32_t dtype;
    float scale;
    int32_t zero_point;
} HctActivationRangeIn;

typedef struct
{
    int32_t min;
    int32_t max;
} HctActivationRangeOut;

/* Per-tensor quantization of an int8/int16 Add, scales as float32 as TFLite stores them. */
typedef struct
{
    int32_t dtype;
    int32_t activation;
    float input1_scale;
    int32_t input1_zero_point;
    float input2_scale;
    int32_t input2_zero_point;
    float output_scale;
    int32_t output_zero_point;
} HctAddQuant;

/* Offsets follow CMSIS-NN; input offsets are -zero_point, output offset is +zero_point. */
typedef struct
{
    int32_t left_shift;
    int32_t input1_offset;
    int32_t input1_multiplier;
    int32_t input1_shift;
    int32_t input2_offset;
    int32_t input2_multiplier;
    int32_t input2_shift;
    int32_t output_offset;
    int32_t output_multiplier;
    int32_t output_shift;
    int32_t activation_min;
    int32_t activation_max;
} HctAddParams;

/* Float clamp; +/-INFINITY leaves the side unbounded. */
typedef struct
{
    float activation_min;
    float activation_max;
} HctFloatActivationParams;

int32_t hct_ref_abi_version(void);

int32_t hct_ref_quantize_multiplier(const HctQuantizeMultiplierIn *in, HctQuantizeMultiplierOut *out);
int32_t hct_ref_activation_range_quantized(const HctActivationRangeIn *in, HctActivationRangeOut *out);
int32_t hct_ref_add_prepare(const HctAddQuant *in, HctAddParams *out);

/* inputs [input1:int8, input2:int8] -> outputs [output:int8] */
int32_t hct_ref_add_s8(const HctAddParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:int16, input2:int16] -> outputs [output:int16] */
int32_t hct_ref_add_s16(const HctAddParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float32, input2:float32] -> outputs [output:float32] */
int32_t hct_ref_add_f32(const HctFloatActivationParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);
/* inputs [input1:float16, input2:float16] -> outputs [output:float16] */
int32_t hct_ref_add_f16(const HctFloatActivationParams *params,
    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif /* HCT_REF_ABI_H */
