/*
 * Shared standalone-firmware runtime for generated helia-core-tester tests.
 *
 * Every generated test .c file is its own standalone firmware image (its own
 * main()/HardFault_Handler/etc.), but the runtime and validation logic itself
 * is identical across all ~2,000+ generated files. Rather than re-emitting
 * this logic via Jinja into every generated file, it is compiled exactly once
 * into the `helia_test_runtime` static library and linked into every test
 * executable (see CMakeLists.txt). Generated files only need:
 *
 *   #include "test_runtime/helia_test_runtime.h"
 *
 * Validation macros (HELIA_VALIDATE_*) intentionally remain macros: they
 * must expand at the call site so the compiler can see the concrete element
 * type of the caller's arrays (for the compile-time float/int guard and for
 * correct promotion in the per-type loops below). Everything that does NOT
 * need call-site type information (platform init, fault handling, failure
 * reporting) is a plain compiled function instead.
 */
#ifndef HELIA_TEST_RUNTIME_H
#define HELIA_TEST_RUNTIME_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------- */
/* Platform / lifecycle (defined once in helia_test_runtime.c)            */
/* ---------------------------------------------------------------------- */

void helia_test_platform_init(void);
void helia_test_finish(int32_t failures);

/* ---------------------------------------------------------------------- */
/* Failure reporting (defined once in helia_test_runtime.c)               */
/* ---------------------------------------------------------------------- */

void helia_test_print_failure_count(int failures);
int helia_test_status_failure(const char *label, int status);
int helia_test_expected_status_failure(const char *label, int status, int expected_status);
int helia_test_scalar_int_mismatch(const char *label, const char *subject, int expected, int actual);
int helia_test_finish_validation(int failures);
double helia_test_float_tolerance(double expected, double atol, double rtol);

#ifdef __cplusplus
}
#endif

#define HELIA_VALIDATE_EXPECTED_STATUS(label, status, expected_status) \
    do { \
        int helia_status = (int)(status); \
        int helia_expected_status = (int)(expected_status); \
        if (helia_status != helia_expected_status) { \
            if (helia_expected_status == (int)ARM_CMSIS_NN_SUCCESS) { \
                return helia_test_status_failure((label), helia_status); \
            } \
            return helia_test_expected_status_failure((label), helia_status, helia_expected_status); \
        } \
    } while (0)

#define HELIA_VALIDATE_STATUS(label, status) \
    HELIA_VALIDATE_EXPECTED_STATUS((label), (status), ARM_CMSIS_NN_SUCCESS)

#define HELIA_VALIDATE_SCALAR_EQ_INT(label, subject, expected, actual) \
    do { \
        int helia_expected = (int)(expected); \
        int helia_actual = (int)(actual); \
        if (helia_actual != helia_expected) { \
            return helia_test_scalar_int_mismatch((label), (subject), helia_expected, helia_actual); \
        } \
    } while (0)

#define HELIA_VALIDATE_RETURN_FAILURES(failures) \
    return helia_test_finish_validation((failures))

/*
 * Compile-time guard (issue #54): the integer validators cast elements to
 * long long, which silently truncates float outputs (|v| < 1 becomes 0 on
 * both sides and always "matches"). If a generator regression ever routes a
 * float-typed output into an integer validator again, fail the BUILD instead
 * of silently passing forever. __fp16 is included only where the target
 * defines an IEEE half type, which is every configuration that can compile
 * float16 test data in the first place.
 */
#if (defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)) && defined(__FLT16_MAX__)
/* __fp16 and _Float16 are distinct types on GCC/Clang Arm targets; block both
 * (float16_t may typedef either depending on toolchain/host configuration). */
#define HELIA_ELEM_IS_FLOAT(expr) _Generic((expr), float: 1, double: 1, __fp16: 1, _Float16: 1, default: 0)
#elif defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
#define HELIA_ELEM_IS_FLOAT(expr) _Generic((expr), float: 1, double: 1, __fp16: 1, default: 0)
#elif defined(__FLT16_MAX__)
#define HELIA_ELEM_IS_FLOAT(expr) _Generic((expr), float: 1, double: 1, _Float16: 1, default: 0)
#else
#define HELIA_ELEM_IS_FLOAT(expr) _Generic((expr), float: 1, double: 1, default: 0)
#endif
#define HELIA_ASSERT_INT_VALIDATOR_INPUT(arr) \
    _Static_assert(!HELIA_ELEM_IS_FLOAT((arr)[0]), \
                   "integer validator applied to float outputs: validation-mode coercion " \
                   "(helia-core-tester issue #54); float outputs need the FLOAT validator")

#define HELIA_VALIDATE_EXACT_INTS(actual, expected, size, max_reports, failures) \
    do { \
        HELIA_ASSERT_INT_VALIDATOR_INPUT(actual); \
        HELIA_ASSERT_INT_VALIDATOR_INPUT(expected); \
        for (int helia_i = 0; helia_i < (size); ++helia_i) { \
            long long helia_act_val = (long long)((actual)[helia_i]); \
            long long helia_exp_val = (long long)((expected)[helia_i]); \
            if (helia_act_val != helia_exp_val) { \
                ++(failures); \
                if ((failures) <= (max_reports)) { \
                    printf("Mismatch[%d]: exp=%lld got=%lld\r\n", helia_i, helia_exp_val, helia_act_val); \
                } \
            } \
        } \
    } while (0)

#define HELIA_VALIDATE_TOLERANT_INTS(actual, expected, size, tolerance, max_reports, failures) \
    do { \
        HELIA_ASSERT_INT_VALIDATOR_INPUT(actual); \
        HELIA_ASSERT_INT_VALIDATOR_INPUT(expected); \
        for (int helia_i = 0; helia_i < (size); ++helia_i) { \
            long long helia_act_val = (long long)((actual)[helia_i]); \
            long long helia_exp_val = (long long)((expected)[helia_i]); \
            long long helia_diff = helia_act_val - helia_exp_val; \
            if (helia_diff < 0) { \
                helia_diff = -helia_diff; \
            } \
            if (helia_diff > (long long)(tolerance)) { \
                ++(failures); \
                if ((failures) <= (max_reports)) { \
                    printf( \
                        "Mismatch[%d]: exp=%lld got=%lld (diff=%lld)\r\n", \
                        helia_i, \
                        helia_exp_val, \
                        helia_act_val, \
                        helia_diff \
                    ); \
                } \
            } \
        } \
    } while (0)

/*
 * Headroom instrumentation (issue #53): every float-validated case prints a
 * single "HELIA_FLOAT_MAXDIFF" summary line, pass or fail, recording the
 * largest raw diff observed and the largest fraction of the tolerance budget
 * it consumed (diff / tol). This is what makes "re-measure headroom" a
 * mechanical FVP run instead of a by-hand exercise: the Python side just
 * greps this one line out of every float test's stdout. helia_max_frac is
 * left at -1.0 (sentinel, printed as "-1.000000") when a per-element
 * tolerance budget is exactly zero and the diff is nonzero -- an
 * unrepresentable "infinite" fraction rather than a real headroom number.
 */
#define HELIA_VALIDATE_FLOATS(actual, expected, size, atol, rtol, max_reports, failures) \
    do { \
        double helia_max_diff = 0.0; \
        double helia_max_frac = 0.0; \
        int helia_zero_tol_violation = 0; \
        for (int helia_i = 0; helia_i < (size); ++helia_i) { \
            double helia_act_val = (double)((actual)[helia_i]); \
            double helia_exp_val = (double)((expected)[helia_i]); \
            double helia_diff = fabs(helia_act_val - helia_exp_val); \
            double helia_tol = helia_test_float_tolerance( \
                helia_exp_val, \
                (double)(atol), \
                (double)(rtol) \
            ); \
            if (helia_diff > helia_max_diff) { \
                helia_max_diff = helia_diff; \
            } \
            if (helia_tol > 0.0) { \
                double helia_frac = helia_diff / helia_tol; \
                if (helia_frac > helia_max_frac) { \
                    helia_max_frac = helia_frac; \
                } \
            } else if (helia_diff > 0.0) { \
                helia_zero_tol_violation = 1; \
            } \
            if (helia_diff > helia_tol) { \
                ++(failures); \
                if ((failures) <= (max_reports)) { \
                    printf( \
                        "Mismatch[%d]: exp=%.6f got=%.6f (diff=%.6f, tol=%.6f)\r\n", \
                        helia_i, \
                        helia_exp_val, \
                        helia_act_val, \
                        helia_diff, \
                        helia_tol \
                    ); \
                } \
            } \
        } \
        printf( \
            "HELIA_FLOAT_MAXDIFF maxdiff=%.8e maxfrac=%.6f n=%d\r\n", \
            helia_max_diff, \
            helia_zero_tol_violation ? -1.0 : helia_max_frac, \
            (int)(size) \
        ); \
    } while (0)

#define HELIA_VALIDATE_BOOLEANS(actual, expected, size, max_reports, failures) \
    do { \
        for (int helia_i = 0; helia_i < (size); ++helia_i) { \
            int helia_act_val = ((actual)[helia_i]) ? 1 : 0; \
            int helia_exp_val = ((expected)[helia_i]) ? 1 : 0; \
            if (helia_act_val != helia_exp_val) { \
                ++(failures); \
                if ((failures) <= (max_reports)) { \
                    printf("Mismatch[%d]: exp=%d got=%d\r\n", helia_i, helia_exp_val, helia_act_val); \
                } \
            } \
        } \
    } while (0)

#define HELIA_VALIDATE_OUTPUTS_EXACT_INT(actual, expected, size, tolerance, atol, rtol, max_reports, failures) \
    HELIA_VALIDATE_EXACT_INTS((actual), (expected), (size), (max_reports), (failures))

#define HELIA_VALIDATE_OUTPUTS_TOLERANT_INT(actual, expected, size, tolerance, atol, rtol, max_reports, failures) \
    HELIA_VALIDATE_TOLERANT_INTS((actual), (expected), (size), (tolerance), (max_reports), (failures))

#define HELIA_VALIDATE_OUTPUTS_FLOAT(actual, expected, size, tolerance, atol, rtol, max_reports, failures) \
    HELIA_VALIDATE_FLOATS((actual), (expected), (size), (atol), (rtol), (max_reports), (failures))

#define HELIA_VALIDATE_OUTPUTS_BOOL(actual, expected, size, tolerance, atol, rtol, max_reports, failures) \
    HELIA_VALIDATE_BOOLEANS((actual), (expected), (size), (max_reports), (failures))

#define HELIA_VALIDATE_OUTPUTS_NONE(actual, expected, size, tolerance, atol, rtol, max_reports, failures) \
    do { \
        (void)(actual); \
        (void)(expected); \
        (void)(size); \
        (void)(tolerance); \
        (void)(atol); \
        (void)(rtol); \
        (void)(max_reports); \
        (void)(failures); \
    } while (0)

#define HELIA_VALIDATE_OUTPUTS(mode, actual, expected, size, tolerance, atol, rtol, max_reports, failures) \
    HELIA_VALIDATE_OUTPUTS_##mode((actual), (expected), (size), (tolerance), (atol), (rtol), (max_reports), (failures))

#endif /* HELIA_TEST_RUNTIME_H */
