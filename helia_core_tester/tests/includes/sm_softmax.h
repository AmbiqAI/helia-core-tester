#ifndef SM_HARNESS_H
#define SM_HARNESS_H

#include <stdint.h>
// Input arrays may carry NAN/INFINITY tokens, and this header is included ahead of
// any other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"

static const cmsis_nn_dims sm_input_dims = {
    .n = 1,
    .h = 1,
    .w = 4,
    .c = 3
};

static const cmsis_nn_dims sm_output_dims = {
    .n = 1,
    .h = 1,
    .w = 4,
    .c = 3
};

// Input data (for testing)
static const int8_t sm_input[] = {
    1
};

// Expected output (golden)
static const int8_t sm_expected_output[] = {
    2
};

#endif