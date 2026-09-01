/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host replacement for the FVP finish/EOT behaviour of
 * src/test_runtime/helia_test_runtime.c: instead of signalling EOT and
 * spinning, exit the process with 0 (all samples passed) or 1 (failures).
 * The real helia_test_finish is renamed away in the runtime's translation
 * unit via -Dhelia_test_finish=helia_test_finish_fvp_unused.
 */
#include <stdint.h>
#include <stdlib.h>

void helia_test_finish(int32_t failures)
{
    exit(failures == 0 ? 0 : 1);
}
