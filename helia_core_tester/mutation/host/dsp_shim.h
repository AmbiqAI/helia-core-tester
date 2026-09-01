/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host emulation of the Armv7E-M DSP intrinsics used by the ns-cmsis-nn
 * int kernels, so the ARM_MATH_DSP code paths can be compiled and executed
 * on the host (x86) for differential and mutation testing.
 *
 * Vendored from the helia-core-tester PR #88 review harness (the
 * differential sweep that reproduced ns-cmsis-nn#343), extended with the
 * multiply-accumulate and saturating intrinsics needed to compile the
 * convolution / fully-connected support kernels.
 *
 * Pre-included via `-include` before any ns-cmsis-nn header; the macros
 * defined here pre-empt the (unavailable) ACLE definitions in
 * Include/Internal/arm_nn_compiler.h. GE flags are not modelled; the
 * kernels do not read them.
 */
#ifndef HOST_DSP_SHIM_H
#define HOST_DSP_SHIM_H

#include <stdint.h>

static inline uint32_t host_ror(uint32_t v, uint32_t n)
{
    n &= 31U;
    return n ? ((v >> n) | (v << (32U - n))) : v;
}

/* Sign-extend bytes 0 and 2 to the two halfwords. */
static inline int32_t host_sxtb16(uint32_t v)
{
    uint32_t lo = (uint32_t)(int32_t)(int8_t)(v & 0xFFU) & 0xFFFFU;
    uint32_t hi = (uint32_t)(int32_t)(int8_t)((v >> 16) & 0xFFU) & 0xFFFFU;
    return (int32_t)(lo | (hi << 16));
}

static inline int32_t host_sxtab16(uint32_t a, uint32_t v)
{
    uint32_t lo = (uint32_t)((int32_t)(int16_t)(a & 0xFFFFU) + (int32_t)(int8_t)(v & 0xFFU)) & 0xFFFFU;
    uint32_t hi = (uint32_t)((int32_t)(int16_t)((a >> 16) & 0xFFFFU) + (int32_t)(int8_t)((v >> 16) & 0xFFU)) & 0xFFFFU;
    return (int32_t)(lo | (hi << 16));
}

/* Per-halfword modulo add (GE flags not modelled; kernels do not read them). */
static inline int32_t host_sadd16(int32_t a, int32_t b)
{
    uint32_t lo = (uint32_t)((int32_t)(int16_t)((uint32_t)a & 0xFFFFU) + (int32_t)(int16_t)((uint32_t)b & 0xFFFFU)) & 0xFFFFU;
    uint32_t hi = (uint32_t)((int32_t)(int16_t)(((uint32_t)a >> 16) & 0xFFFFU) + (int32_t)(int16_t)(((uint32_t)b >> 16) & 0xFFFFU)) & 0xFFFFU;
    return (int32_t)(lo | (hi << 16));
}

static inline int16_t host_lo16(int32_t v) { return (int16_t)((uint32_t)v & 0xFFFFU); }
static inline int16_t host_hi16(int32_t v) { return (int16_t)(((uint32_t)v >> 16) & 0xFFFFU); }

static inline int32_t host_smulbb(int32_t a, int32_t b) { return (int32_t)host_lo16(a) * host_lo16(b); }
static inline int32_t host_smultt(int32_t a, int32_t b) { return (int32_t)host_hi16(a) * host_hi16(b); }

/* Multiply-accumulate: wrap-around (modulo) semantics, as on hardware. */
static inline int32_t host_smlabb(int32_t a, int32_t b, int32_t c)
{
    return (int32_t)((uint32_t)host_smulbb(a, b) + (uint32_t)c);
}

static inline int32_t host_smlatt(int32_t a, int32_t b, int32_t c)
{
    return (int32_t)((uint32_t)host_smultt(a, b) + (uint32_t)c);
}

/* Dual 16x16 multiply-add: c + lo(a)*lo(b) + hi(a)*hi(b), modulo 2^32. */
static inline int32_t host_smlad(int32_t a, int32_t b, int32_t c)
{
    return (int32_t)((uint32_t)c + (uint32_t)host_smulbb(a, b) + (uint32_t)host_smultt(a, b));
}

/* Dual 16x16 multiply with 64-bit accumulate. */
static inline int64_t host_smlald(int32_t a, int32_t b, int64_t c)
{
    return c + host_smulbb(a, b) + host_smultt(a, b);
}

/* Saturating 32-bit add. */
static inline int32_t host_qadd(int32_t a, int32_t b)
{
    int64_t r = (int64_t)a + b;
    if (r > INT32_MAX) { return INT32_MAX; }
    if (r < INT32_MIN) { return INT32_MIN; }
    return (int32_t)r;
}

static inline int8_t host_qsub8_lane(int8_t a, int8_t b)
{
    int32_t r = (int32_t)a - b;
    if (r > 127) { r = 127; }
    if (r < -128) { r = -128; }
    return (int8_t)r;
}

static inline int32_t host_qsub8(int32_t a, int32_t b)
{
    uint32_t out = 0;
    for (int lane = 0; lane < 4; lane++)
    {
        int8_t la = (int8_t)(((uint32_t)a >> (8 * lane)) & 0xFFU);
        int8_t lb = (int8_t)(((uint32_t)b >> (8 * lane)) & 0xFFU);
        out |= ((uint32_t)(uint8_t)host_qsub8_lane(la, lb)) << (8 * lane);
    }
    return (int32_t)out;
}

static inline int16_t host_qsub16_lane(int16_t a, int16_t b)
{
    int32_t r = (int32_t)a - b;
    if (r > 32767) { r = 32767; }
    if (r < -32768) { r = -32768; }
    return (int16_t)r;
}

static inline int32_t host_qsub16(int32_t a, int32_t b)
{
    uint32_t lo = (uint16_t)host_qsub16_lane(host_lo16(a), host_lo16(b));
    uint32_t hi = (uint16_t)host_qsub16_lane(host_hi16(a), host_hi16(b));
    return (int32_t)(lo | (hi << 16));
}

#define ROR(v, n) host_ror((uint32_t)(v), (uint32_t)(n))
#define SXTB16(v) host_sxtb16((uint32_t)(v))
#define SXTAB16(a, v) host_sxtab16((uint32_t)(a), (uint32_t)(v))
#define SXTB16_RORn(v, n) SXTB16(ROR((uint32_t)(v), (n)))
#define SXTAB16_RORn(a, v, n) SXTAB16((a), ROR((uint32_t)(v), (n)))
#define SADD16(a, b) host_sadd16((int32_t)(a), (int32_t)(b))
#define SMULBB(a, b) host_smulbb((int32_t)(a), (int32_t)(b))
#define SMULTT(a, b) host_smultt((int32_t)(a), (int32_t)(b))
#define SMLABB(a, b, c) host_smlabb((int32_t)(a), (int32_t)(b), (int32_t)(c))
#define SMLATT(a, b, c) host_smlatt((int32_t)(a), (int32_t)(b), (int32_t)(c))
#define SMLAD(a, b, c) host_smlad((int32_t)(a), (int32_t)(b), (int32_t)(c))
#define SMLALD(a, b, c) host_smlald((int32_t)(a), (int32_t)(b), (int64_t)(c))
#define QADD(a, b) host_qadd((int32_t)(a), (int32_t)(b))
#define QSUB8(a, b) host_qsub8((int32_t)(a), (int32_t)(b))
#define QSUB16(a, b) host_qsub16((int32_t)(a), (int32_t)(b))
#define PKHBT(a, b, n) (((((uint32_t)(a))) & 0x0000FFFFUL) | ((((uint32_t)(b)) << (n)) & 0xFFFF0000UL))
#define PKHTB(a, b, n) (((((uint32_t)(a))) & 0xFFFF0000UL) | ((((uint32_t)(b)) >> (n)) & 0x0000FFFFUL))

/* A few support-function bodies call the raw ACLE spellings directly. */
#define __sxtb16(v) host_sxtb16((uint32_t)(v))
#define __ror(v, n) host_ror((uint32_t)(v), (uint32_t)(n))

#endif /* HOST_DSP_SHIM_H */
