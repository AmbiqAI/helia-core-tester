/*
 * Drives HELIA_VALIDATE_FLOATS over every finite/non-finite operand pairing
 * (issue #75), for float and for float16, and prints one machine-readable line
 * per case. Built on the host at -Ofast and at -O3 -ffast-math by
 * test_float_nonfinite_compare.py: the target harnesses build with one or the
 * other, and it is the -ffinite-math-only both imply that the classification
 * has to survive.
 *
 * Operands come from helia_float_compare_host_producer.c so that the compiler
 * building this file cannot see a non-finite value being created.
 */
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "helia_float_compare_host_producer.h"
#include "test_runtime/helia_test_runtime.h"

struct helia_case
{
    const char *name;
    char actual_kind;
    float actual_finite;
    char expected_kind;
    float expected_finite;
};

static const struct helia_case helia_cases[] = {
    {"nan_vs_finite", 'n', 0.0f, 'f', 1.0f},
    {"finite_vs_nan", 'f', 1.0f, 'n', 0.0f},
    {"finite_vs_posinf", 'f', 1.0f, 'p', 0.0f},
    {"neginf_vs_posinf", 'm', 0.0f, 'p', 0.0f},
    {"nan_vs_nan", 'n', 0.0f, 'n', 0.0f},
    {"posinf_vs_posinf", 'p', 0.0f, 'p', 0.0f},
    {"neginf_vs_neginf", 'm', 0.0f, 'm', 0.0f},
    {"posinf_vs_finite", 'p', 0.0f, 'f', 1.0f},
    {"nan_vs_posinf", 'n', 0.0f, 'p', 0.0f},
    {"posinf_vs_fltmax", 'p', 0.0f, 'f', FLT_MAX},
    {"finite_match", 'f', 1.0f, 'f', 1.0f},
    {"finite_mismatch", 'f', 2.0f, 'f', 1.0f},
};

static const size_t helia_case_count = sizeof(helia_cases) / sizeof(helia_cases[0]);

static void helia_run_case_f32(const struct helia_case *test_case, const char *tag, double rtol)
{
    float actual[1];
    float expected[1];
    int failures = 0;

    actual[0] = helia_host_produce_f32(test_case->actual_kind, test_case->actual_finite);
    expected[0] = helia_host_produce_f32(test_case->expected_kind, test_case->expected_finite);

    printf("CASE %s.%s\r\n", test_case->name, tag);
    HELIA_VALIDATE_FLOATS(actual, expected, 1, 1e-5, rtol, 8, failures);
    printf("RESULT %s.%s failures=%d\r\n", test_case->name, tag, failures);
}

#ifdef HELIA_HOST_HAVE_F16

/* FLT_MAX is +Inf in binary16, which would turn the largest-finite pairing
 * into a matched-infinity pairing. */
static float helia_f16_finite(float value)
{
    return (value > 65504.0f) ? 65504.0f : value;
}

static void helia_run_case_f16(const struct helia_case *test_case, const char *tag, double rtol)
{
    helia_host_f16 actual[1];
    helia_host_f16 expected[1];
    int failures = 0;

    actual[0] = helia_host_produce_f16(
        test_case->actual_kind,
        helia_f16_finite(test_case->actual_finite)
    );
    expected[0] = helia_host_produce_f16(
        test_case->expected_kind,
        helia_f16_finite(test_case->expected_finite)
    );

    printf("CASE f16_%s.%s\r\n", test_case->name, tag);
    HELIA_VALIDATE_FLOATS(actual, expected, 1, 1e-5, rtol, 8, failures);
    printf("RESULT f16_%s.%s failures=%d\r\n", test_case->name, tag, failures);
}

static void helia_run_f16_tensor_cases(void)
{
    helia_host_f16 actual[3];
    helia_host_f16 expected[3];
    int failures = 0;

    actual[0] = helia_host_produce_f16('n', 0.0f);
    actual[1] = helia_host_produce_f16('f', 1.0f);
    actual[2] = helia_host_produce_f16('f', 2.0f);
    expected[0] = helia_host_produce_f16('f', 1.0f);
    expected[1] = helia_host_produce_f16('f', 1.0f);
    expected[2] = helia_host_produce_f16('f', 2.0f);

    printf("CASE f16_tensor_nan_lane\r\n");
    HELIA_VALIDATE_FLOATS(actual, expected, 3, 1e-5, 1e-5, 8, failures);
    printf("RESULT f16_tensor_nan_lane failures=%d\r\n", failures);
}

#endif /* HELIA_HOST_HAVE_F16 */

/*
 * The producer hides where a non-finite value came from, which matches the
 * shipped generator (separate kernel TU, no LTO) but is also the one shape no
 * compiler folds. These rows do the opposite: the bit pattern is a literal in
 * this translation unit, a few lines from the comparison, so the optimizer
 * knows the operand's class before HELIA_VALIDATE_FLOATS looks at it. That is
 * the shape an inlined or header-only kernel would produce. No compiler tried
 * here folds it even against the pre-fix classifier, so these rows are
 * coverage of the exposure rather than a detector for it: the discriminating
 * test is the generated-shape harness driven by
 * test_float_nonfinite_fold_harness.py, which does fold.
 */
static float helia_visible_f32(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void helia_run_visible_case_f32(
    const char *name,
    uint32_t actual_bits,
    uint32_t expected_bits
)
{
    float actual[1];
    float expected[1];
    int failures = 0;

    actual[0] = helia_visible_f32(actual_bits);
    expected[0] = helia_visible_f32(expected_bits);

    printf("CASE %s\r\n", name);
    HELIA_VALIDATE_FLOATS(actual, expected, 1, 1e-5, 1e-5, 8, failures);
    printf("RESULT %s failures=%d\r\n", name, failures);
}

static void helia_run_visible_cases_f32(void)
{
    helia_run_visible_case_f32("visible_nan_vs_finite", 0x7FC00000u, 0x3F800000u);
    helia_run_visible_case_f32("visible_finite_vs_nan", 0x3F800000u, 0x7FC00000u);
    helia_run_visible_case_f32("visible_neginf_vs_finite", 0xFF800000u, 0x3F800000u);
    helia_run_visible_case_f32("visible_posinf_vs_neginf", 0x7F800000u, 0xFF800000u);
}

#ifdef HELIA_HOST_HAVE_F16

static helia_host_f16 helia_visible_f16(uint16_t bits)
{
    helia_host_f16 value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void helia_run_visible_case_f16(
    const char *name,
    uint16_t actual_bits,
    uint16_t expected_bits
)
{
    helia_host_f16 actual[1];
    helia_host_f16 expected[1];
    int failures = 0;

    actual[0] = helia_visible_f16(actual_bits);
    expected[0] = helia_visible_f16(expected_bits);

    printf("CASE %s\r\n", name);
    HELIA_VALIDATE_FLOATS(actual, expected, 1, 1e-5, 1e-5, 8, failures);
    printf("RESULT %s failures=%d\r\n", name, failures);
}

static void helia_run_visible_cases_f16(void)
{
    helia_run_visible_case_f16("f16_visible_nan_vs_finite", 0x7E00u, 0x3C00u);
    helia_run_visible_case_f16("f16_visible_finite_vs_nan", 0x3C00u, 0x7E00u);
    helia_run_visible_case_f16("f16_visible_neginf_vs_finite", 0xFC00u, 0x3C00u);
    helia_run_visible_case_f16("f16_visible_posinf_vs_neginf", 0x7C00u, 0xFC00u);
}

#endif /* HELIA_HOST_HAVE_F16 */

/*
 * Tensor-level rows: a single element decides only the per-element verdict,
 * while the headroom sentinel is a property of the whole tensor.
 */
static void helia_run_tensor_case(
    const char *name,
    const float *actual,
    const float *expected,
    int size
)
{
    int failures = 0;

    printf("CASE %s\r\n", name);
    HELIA_VALIDATE_FLOATS(actual, expected, size, 1e-5, 1e-5, 8, failures);
    printf("RESULT %s failures=%d\r\n", name, failures);
}

static void helia_run_tensor_cases(void)
{
    const float nan_value = helia_host_produce_f32('n', 0.0f);
    float actual[3];
    float expected[3];

    /* One matched NaN lane must not cost the finite lanes their headroom. */
    actual[0] = nan_value;
    actual[1] = 1.0f;
    actual[2] = 2.00001f;
    expected[0] = nan_value;
    expected[1] = 1.0f;
    expected[2] = 2.0f;
    helia_run_tensor_case("tensor_matched_nan_with_finite", actual, expected, 3);

    actual[0] = 1.0f;
    actual[1] = 2.0f;
    actual[2] = 3.0f;
    expected[0] = 1.0f;
    expected[1] = nan_value;
    expected[2] = 3.0f;
    helia_run_tensor_case("tensor_nonfinite_mismatch", actual, expected, 3);

    actual[0] = nan_value;
    actual[1] = nan_value;
    actual[2] = nan_value;
    expected[0] = nan_value;
    expected[1] = nan_value;
    expected[2] = nan_value;
    helia_run_tensor_case("tensor_all_nan_matched", actual, expected, 3);

    /* A zero-length validation compares nothing, so it passes and records the
     * sentinel rather than a headroom number it never measured. */
    helia_run_tensor_case("tensor_empty", actual, expected, 0);
}

/*
 * Masked rows (issue #74): a don't-care lane must cost nothing -- no failure,
 * no headroom contribution -- while the lanes beside it stay fully asserted.
 */
static void helia_run_masked_case(
    const char *name,
    const float *actual,
    const float *expected,
    const uint8_t *mask,
    int size
)
{
    int failures = 0;

    printf("CASE %s\r\n", name);
    HELIA_VALIDATE_FLOATS_MASKED(actual, expected, mask, size, 1e-5, 1e-5, 8, failures);
    printf("RESULT %s failures=%d\r\n", name, failures);
}

static void helia_run_masked_cases(void)
{
    const float nan_value = helia_host_produce_f32('n', 0.0f);
    float actual[3];
    float expected[3];
    uint8_t mask[3];

    /* The generator zeroes a masked expected lane so the header stays finite. */
    actual[0] = nan_value;
    actual[1] = 1.0f;
    actual[2] = 2.0f;
    expected[0] = 0.0f;
    expected[1] = 1.0f;
    expected[2] = 2.0f;
    mask[0] = 1;
    mask[1] = 0;
    mask[2] = 0;
    helia_run_masked_case("masked_nonfinite_actual", actual, expected, mask, 3);

    actual[1] = 2.0f;
    helia_run_masked_case("masked_lane_with_unmasked_mismatch", actual, expected, mask, 3);

    actual[0] = nan_value;
    actual[1] = nan_value;
    actual[2] = nan_value;
    expected[0] = 0.0f;
    expected[1] = 0.0f;
    expected[2] = 0.0f;
    mask[1] = 1;
    mask[2] = 1;
    helia_run_masked_case("masked_all_lanes", actual, expected, mask, 3);

    /* A NULL mask is the plain validator: the summary line must stay absent. */
    actual[0] = 1.0f;
    actual[1] = 1.0f;
    actual[2] = 1.0f;
    expected[0] = 1.0f;
    expected[1] = 1.0f;
    expected[2] = 1.0f;
    helia_run_masked_case("masked_null_is_plain", actual, expected, NULL, 3);
}

int main(void)
{
    for (size_t i = 0; i < helia_case_count; ++i) {
        helia_run_case_f32(&helia_cases[i], "rtol", 1e-5);
    }
    /* rtol == 0 is the second failure mode in issue #75: 0 * Inf is NaN, so the
     * tolerance itself was NaN rather than Inf. */
    for (size_t i = 0; i < helia_case_count; ++i) {
        helia_run_case_f32(&helia_cases[i], "zerortol", 0.0);
    }

    helia_run_tensor_cases();
    helia_run_masked_cases();
    helia_run_visible_cases_f32();

#ifdef HELIA_HOST_HAVE_F16
    printf("F16_SUPPORTED 1\r\n");
    for (size_t i = 0; i < helia_case_count; ++i) {
        helia_run_case_f16(&helia_cases[i], "rtol", 1e-5);
    }
    for (size_t i = 0; i < helia_case_count; ++i) {
        helia_run_case_f16(&helia_cases[i], "zerortol", 0.0);
    }
    helia_run_f16_tensor_cases();
    helia_run_visible_cases_f16();
#else
    printf("F16_SUPPORTED 0\r\n");
#endif

    printf("HOST_SANITY_DONE\r\n");
    return 0;
}
