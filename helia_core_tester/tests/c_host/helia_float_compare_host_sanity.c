/*
 * Drives HELIA_VALIDATE_FLOATS over every finite/non-finite operand pairing
 * (issue #75) and prints one machine-readable line per case. Built on the host
 * at -Ofast by test_float_nonfinite_compare.py, so the same optimization level
 * the FVP/hardware harnesses use is what classifies the operands here.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "test_runtime/helia_test_runtime.h"

/*
 * Operands are assembled from bit patterns behind a volatile read: a literal
 * NAN/INFINITY would be a compile-time constant that -ffinite-math-only lets
 * the optimizer reason away before the comparison ever runs.
 */
static volatile uint32_t helia_nan_bits = 0x7FC00000u;
static volatile uint32_t helia_pos_inf_bits = 0x7F800000u;
static volatile uint32_t helia_neg_inf_bits = 0xFF800000u;

static float helia_float_from_bits(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float helia_operand(char kind, float finite_value)
{
    switch (kind) {
        case 'n':
            return helia_float_from_bits(helia_nan_bits);
        case 'p':
            return helia_float_from_bits(helia_pos_inf_bits);
        case 'm':
            return helia_float_from_bits(helia_neg_inf_bits);
        default:
            return finite_value;
    }
}

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
    {"finite_match", 'f', 1.0f, 'f', 1.0f},
    {"finite_mismatch", 'f', 2.0f, 'f', 1.0f},
};

static void helia_run_case(const struct helia_case *test_case, const char *tag, double rtol)
{
    float actual[1];
    float expected[1];
    int failures = 0;

    actual[0] = helia_operand(test_case->actual_kind, test_case->actual_finite);
    expected[0] = helia_operand(test_case->expected_kind, test_case->expected_finite);

    printf("CASE %s.%s\r\n", test_case->name, tag);
    HELIA_VALIDATE_FLOATS(actual, expected, 1, 1e-5, rtol, 8, failures);
    printf("RESULT %s.%s failures=%d\r\n", test_case->name, tag, failures);
}

int main(void)
{
    const size_t case_count = sizeof(helia_cases) / sizeof(helia_cases[0]);

    for (size_t i = 0; i < case_count; ++i) {
        helia_run_case(&helia_cases[i], "rtol", 1e-5);
    }
    /* rtol == 0 is the second failure mode in issue #75: 0 * Inf is NaN, so the
     * tolerance itself was NaN rather than Inf. */
    for (size_t i = 0; i < case_count; ++i) {
        helia_run_case(&helia_cases[i], "zerortol", 0.0);
    }

    printf("HOST_SANITY_DONE\r\n");
    return 0;
}
