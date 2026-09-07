/*
 * Golden tensor in the exact form the generator emits: a file-scope
 * `static const` array of the element type carrying bare NAN / -NAN /
 * INFINITY / -INFINITY literals, one include away from the validator call.
 * Compare the includes/ directory of a generated add_float_nonfinite_f32 test.
 */
#ifndef HELIA_FOLD_HARNESS_GOLDEN_H
#define HELIA_FOLD_HARNESS_GOLDEN_H

#include <math.h>

#include "helia_fold_harness.h"

static const helia_fold_elem helia_fold_expected_output[] = {
    NAN, -NAN, INFINITY, INFINITY, -INFINITY, 0.5f, 0.25f, 0.133486286f, 0.707955956f, 0.290714413f, -0.177686378f, -0.063936107f, 0.642722428f, -0.374953657f, 0.572441578f, -0.128899693f,
    -0.504739583f, -0.115708925f, 0.793903172f, 0.114048406f, 0.057288449f, -0.448035926f, 0.749724388f, -0.481601506f, 0.611840725f, -0.758276582f, -0.137358963f, -0.300535619f, 0.032033987f, -0.26191476f, -0.055692434f, -0.612801015f
};

#endif
