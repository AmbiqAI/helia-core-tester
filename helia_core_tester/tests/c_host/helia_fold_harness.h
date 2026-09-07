/*
 * Shape shared by the fold-regression driver and its kernel translation unit
 * (issue #75).
 *
 * Lane layout is fixed so the expected failure count is the same on every
 * compiler:
 *
 *   [0] NAN       vs NaN        matched NaN                -> pass
 *   [1] -NAN      vs 3.5        expected NaN, got finite   -> FAIL 1
 *   [2] INFINITY  vs +Inf       matched +Inf               -> pass
 *   [3] INFINITY  vs -Inf       -Inf where +Inf expected   -> FAIL 2
 *   [4] -INFINITY vs -Inf       matched -Inf               -> pass
 *   [5] 0.5       vs NaN        NaN into a finite lane     -> FAIL 3
 *   [6] 0.25      vs 0.75       plain tolerance overrun    -> FAIL 4
 *   [7..31]       identical finite values                  -> pass
 */
#ifndef HELIA_FOLD_HARNESS_H
#define HELIA_FOLD_HARNESS_H

#include <stdint.h>

#include "arm_nnfunctions.h"

#define HELIA_FOLD_N 32
#define HELIA_FOLD_EXPECTED_FAILURES 4
#define HELIA_FOLD_ATOL 5e-05f
#define HELIA_FOLD_RTOL 2e-05f

typedef float helia_fold_elem;

int32_t helia_fold_kernel(helia_fold_elem *output);

#endif
