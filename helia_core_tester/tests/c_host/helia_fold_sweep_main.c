/*
 * Per-lane leg of the issue #75 fold regression: validates one lane at a time
 * so each golden-class x kernel-class pairing gets an unambiguous verdict,
 * then once over the whole tensor so the aggregate matches a real harness.
 *
 * It keeps the generated-harness shape of helia_fold_harness_main.c for the
 * same reason, and must not be reduced either: against the
 * classify-after-widening runtime this shape loses 11 of the 14 must-fail
 * pairings under fast-math, where cut-down probes over one lane stay correct
 * at every optimization level.
 */
#include <stdint.h>
#include <stdio.h>

#include "arm_nnfunctions.h"
#include "helia_fold_sweep.h"
#include "helia_fold_sweep_golden.h"

#include "test_runtime/helia_test_runtime.h"

static helia_fold_sweep_elem helia_fold_sweep_output[HELIA_FOLD_SWEEP_N];
static const char *const helia_fold_sweep_names[] = { HELIA_FOLD_SWEEP_NAMES };
static const int helia_fold_sweep_should_fail[] = { HELIA_FOLD_SWEEP_SHOULD_FAIL };

static int helia_fold_sweep_lane(int lane)
{
    int failures = 0;
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        (helia_fold_sweep_output + lane),
        (helia_fold_sweep_expected + lane),
        1,
        1,
        HELIA_FOLD_SWEEP_ATOL,
        HELIA_FOLD_SWEEP_RTOL,
        20,
        failures
    );
    return failures;
}

static int32_t helia_fold_sweep_test_case_run(int *wrong)
{
    int32_t status = helia_fold_sweep_kernel(helia_fold_sweep_output);
    HELIA_VALIDATE_STATUS("Add", status);

    for (int i = 0; i < HELIA_FOLD_SWEEP_N; ++i) {
        printf("LANE %s\r\n", helia_fold_sweep_names[i]);
        int got = helia_fold_sweep_lane(i);
        int want = helia_fold_sweep_should_fail[i];
        if (got != want) {
            ++(*wrong);
        }
        printf(
            "LANEVERDICT %s want=%d got=%d %s\r\n",
            helia_fold_sweep_names[i],
            want,
            got,
            got == want ? "OK" : "WRONG"
        );
    }

    int failures = 0;
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        helia_fold_sweep_output,
        helia_fold_sweep_expected,
        HELIA_FOLD_SWEEP_N,
        1,
        HELIA_FOLD_SWEEP_ATOL,
        HELIA_FOLD_SWEEP_RTOL,
        40,
        failures
    );
    HELIA_VALIDATE_RETURN_FAILURES(failures);
}

int main(void)
{
    helia_test_platform_init();
    int wrong = 0;
    int32_t failures = helia_fold_sweep_test_case_run(&wrong);
    printf(
        "HELIA_FOLD_SWEEP_RESULT tensor_failures=%d expected=%d lanes_wrong=%d\r\n",
        (int)failures,
        HELIA_FOLD_SWEEP_EXPECTED_FAILURES,
        wrong
    );
    return wrong != 0 || (int)failures != HELIA_FOLD_SWEEP_EXPECTED_FAILURES ? 1 : 0;
}
