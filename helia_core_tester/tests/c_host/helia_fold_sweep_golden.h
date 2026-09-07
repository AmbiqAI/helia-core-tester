/*
 * Per-lane golden in the generated-file shape: file-scope `static const` array
 * of the element type with bare NAN / INFINITY literals.
 */
#ifndef HELIA_FOLD_SWEEP_GOLDEN_H
#define HELIA_FOLD_SWEEP_GOLDEN_H

#include <math.h>

#include "helia_fold_sweep.h"

static const helia_fold_sweep_elem helia_fold_sweep_expected[] = {
    NAN, NAN, NAN, NAN, -NAN, -NAN, INFINITY, INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, 3.5f, 3.5f, 3.5f, 3.5f, 0.25f
};

#endif
