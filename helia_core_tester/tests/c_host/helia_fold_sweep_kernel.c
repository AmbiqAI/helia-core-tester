/*
 * Kernel translation unit for the per-lane sweep. Non-finite outputs are
 * assembled from bit patterns because this TU is built with the same
 * -ffinite-math-only as the driver and the values have to still be non-finite
 * when the validator reads them.
 */
#include "helia_fold_sweep.h"

#include <stdint.h>
#include <string.h>

static helia_fold_sweep_elem helia_fold_sweep_from_bits(uint32_t bits)
{
    helia_fold_sweep_elem value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

int32_t helia_fold_sweep_kernel(helia_fold_sweep_elem *output)
{
    static const char kinds[] = HELIA_FOLD_SWEEP_KINDS;
    for (int i = 0; i < HELIA_FOLD_SWEEP_N; ++i) {
        switch (kinds[i]) {
            case 'n': output[i] = helia_fold_sweep_from_bits(0x7FC00000u); break;
            case 'p': output[i] = helia_fold_sweep_from_bits(0x7F800000u); break;
            case 'm': output[i] = helia_fold_sweep_from_bits(0xFF800000u); break;
            case 'F': output[i] = 0.75f; break;
            default:  output[i] = 3.5f; break;
        }
    }
    return (int32_t)ARM_CMSIS_NN_SUCCESS;
}
