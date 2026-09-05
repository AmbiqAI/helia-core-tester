/*
 * Regression driver for issue #75: a validator that classifies a float element
 * after widening it to double loses non-finite lanes under -ffinite-math-only,
 * because the widening conversion carries the nnan/ninf flags that make the
 * binary64 exponent test provably false.
 *
 * This file reproduces a generated float harness rather than probing the
 * macro: golden as a file-scope `static const` array of the element type with
 * bare NAN/INFINITY literals, kernel output produced in a second translation
 * unit, and HELIA_VALIDATE_OUTPUTS(FLOAT, ...) expanded in a
 * *_test_case_run() that main calls. Every part of that shape is load-bearing
 * and none of it may be reduced: measured against the classify-after-widening
 * runtime, this shape drops a planted non-finite failure on Apple clang and on
 * Homebrew clang at -Ofast, -O3 -ffast-math and -O3 -ffinite-math-only, while
 * one-lane and two-classifier probes stay correct at every optimization level
 * on the same compilers. The fold needs enough surrounding validator code for
 * the optimizer to act on the flags, so a smaller driver silently stops being
 * a test. Expanding the macro directly in main() is also not equivalent: that
 * is the one variant of this shape that did not fold.
 *
 * helia_test_finish() is deliberately not called: it spins waiting for the FVP
 * to shut down, and nothing it does is part of the validation under test.
 */
#include <stdint.h>
#include <stdio.h>

#include "arm_nnfunctions.h"
#include "helia_fold_harness.h"
#include "helia_fold_harness_golden.h"

#include "test_runtime/helia_test_runtime.h"

static helia_fold_elem helia_fold_output[HELIA_FOLD_N];

static int32_t helia_fold_test_case_run(void)
{
    int32_t status = helia_fold_kernel(helia_fold_output);
    HELIA_VALIDATE_STATUS("Add", status);

    int failures = 0;
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        helia_fold_output,
        helia_fold_expected_output,
        HELIA_FOLD_N,
        1,
        HELIA_FOLD_ATOL,
        HELIA_FOLD_RTOL,
        20,
        failures
    );
    HELIA_VALIDATE_RETURN_FAILURES(failures);
}

int main(void)
{
    helia_test_platform_init();
    int32_t failures = helia_fold_test_case_run();
    printf(
        "HELIA_FOLD_RESULT failures=%d expected=%d\r\n",
        (int)failures,
        HELIA_FOLD_EXPECTED_FAILURES
    );
    return (int)failures == HELIA_FOLD_EXPECTED_FAILURES ? 0 : 1;
}
