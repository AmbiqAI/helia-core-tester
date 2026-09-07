/*
 * Lane table for the per-lane leg of the issue #75 fold regression: every
 * golden-class x kernel-output-class pairing, 14 of which must fail. Shared by
 * the sweep driver and its kernel translation unit.
 */
#ifndef HELIA_FOLD_SWEEP_H
#define HELIA_FOLD_SWEEP_H

#include <stdint.h>

#include "arm_nnfunctions.h"

#define HELIA_FOLD_SWEEP_N 19
#define HELIA_FOLD_SWEEP_EXPECTED_FAILURES 14
#define HELIA_FOLD_SWEEP_ATOL 5e-05f
#define HELIA_FOLD_SWEEP_RTOL 2e-05f

/* One character per lane, indexed alongside the names below: NaN, +Inf, -Inf,
 * a finite value that matches its golden, a finite value that does not. */
#define HELIA_FOLD_SWEEP_KINDS "npmfnfpmnfmpnfnpmfF"
#define HELIA_FOLD_SWEEP_NAMES \
    "gNAN__aNaN", \
    "gNAN__aPInf", \
    "gNAN__aMInf", \
    "gNAN__aFin", \
    "gMNAN_aNaN", \
    "gMNAN_aFin", \
    "gPInf_aPInf", \
    "gPInf_aMInf", \
    "gPInf_aNaN", \
    "gPInf_aFin", \
    "gMInf_aMInf", \
    "gMInf_aPInf", \
    "gMInf_aNaN", \
    "gMInf_aFin", \
    "gFin__aNaN", \
    "gFin__aPInf", \
    "gFin__aMInf", \
    "gFin__aFin", \
    "gFin__aFinBad"
#define HELIA_FOLD_SWEEP_SHOULD_FAIL 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1

typedef float helia_fold_sweep_elem;

int32_t helia_fold_sweep_kernel(helia_fold_sweep_elem *output);

#endif
