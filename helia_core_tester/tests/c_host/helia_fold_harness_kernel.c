/*
 * Stands in for the ns-cmsis-nn kernel a generated harness calls, and lives in
 * its own translation unit for the same reason the real one does: the driver's
 * compiler cannot see which lanes come back non-finite.
 */
#include "helia_fold_harness.h"

#include <stdint.h>
#include <string.h>

/*
 * Non-finite lanes come from IEEE bit patterns rather than NAN/INFINITY
 * literals because this TU is built with the same -ffinite-math-only as the
 * driver, and the values have to still be non-finite when the validator reads
 * them.
 */
static helia_fold_elem helia_fold_from_bits(uint32_t bits)
{
    helia_fold_elem value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

int32_t helia_fold_kernel(helia_fold_elem *output)
{
    static const float helia_fold_tail[] = {
        0.133486286f, 0.707955956f, 0.290714413f, -0.177686378f, -0.063936107f,
        0.642722428f, -0.374953657f, 0.572441578f, -0.128899693f, -0.504739583f,
        -0.115708925f, 0.793903172f, 0.114048406f, 0.057288449f, -0.448035926f,
        0.749724388f, -0.481601506f, 0.611840725f, -0.758276582f, -0.137358963f,
        -0.300535619f, 0.032033987f, -0.26191476f, -0.055692434f, -0.612801015f
    };

    output[0] = helia_fold_from_bits(0x7FC00000u);
    output[1] = 3.5f;
    output[2] = helia_fold_from_bits(0x7F800000u);
    output[3] = helia_fold_from_bits(0xFF800000u);
    output[4] = helia_fold_from_bits(0xFF800000u);
    output[5] = helia_fold_from_bits(0x7FC00000u);
    output[6] = 0.75f;

    for (int i = 7; i < HELIA_FOLD_N; ++i) {
        output[i] = helia_fold_tail[i - 7];
    }

    return (int32_t)ARM_CMSIS_NN_SUCCESS;
}
