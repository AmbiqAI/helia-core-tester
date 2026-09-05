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
#include <string.h>

#include "arm_nnfunctions.h"

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
void helia_test_nonfinite_mismatch(
    int index,
    int expected_class,
    double expected,
    int actual_class,
    double actual
);
void helia_test_nonfinite_mismatch_summary(int count);

/*
 * Non-finite classification (issue #75).
 *
 * The class of an element is decoded from that element's own storage bytes,
 * at its own width, before any floating-point operation touches the value.
 * The ordering is the point. Generated tests build at -Ofast or at
 * -O3 -ffast-math (both imply -ffinite-math-only), under which every FP
 * instruction is emitted with nnan/ninf -- including the widening conversion
 * that handing a float or float16_t element to a double-typed classifier would
 * require. The optimizer is then entitled to treat the widened result as
 * finite and to delete an exponent test applied to it, which would report zero
 * failures for a kernel that returned NaN. Decoding the element's own bytes
 * asks nothing of the optimizer: the test is integer arithmetic on a value no
 * FP instruction of ours produced. isnan()/isinf() are unusable for the same
 * reason -- they are licensed to fold to a constant false, the mechanism
 * behind AmbiqAI/ns-cmsis-nn#314.
 *
 * These macros expand in the generated harness translation unit, so the check
 * has to survive whatever flags that TU is compiled with, not just the ones
 * this runtime is built with. README.md records how the classification is
 * exercised.
 *
 * IEEE-754 binary16/32/64 share one rule: a maximal exponent field means Inf
 * when the significand is zero and NaN otherwise. Arm's alternative half
 * format does not follow it -- it has no Inf or NaN encodings and spends the
 * maximal exponent on finite values up to 131008 -- so a target selecting it
 * is rejected below rather than decoded by a rule that does not apply.
 */
enum {
    HELIA_FLOAT_CLASS_FINITE = 0,
    HELIA_FLOAT_CLASS_NAN = 1,
    HELIA_FLOAT_CLASS_POS_INF = 2,
    HELIA_FLOAT_CLASS_NEG_INF = 3
};

/*
 * float16_t is a typedef of __fp16 or of _Float16 depending on the toolchain
 * (ns-cmsis-nn Include/arm_nn_math_types_flt.h), and the two are distinct
 * types where both exist, so every _Generic over element types has to carry
 * whichever rows the target actually has.
 */
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FP16_FORMAT_ALTERNATIVE) \
    || defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
#define HELIA_HAVE_FP16_TYPE 1
#endif
#if defined(__FLT16_MAX__)
#define HELIA_HAVE_FLOAT16_TYPE 1
#endif

#if defined(HELIA_HAVE_FP16_TYPE) && defined(HELIA_HAVE_FLOAT16_TYPE)
#define HELIA_FLOAT16_GENERIC_ROWS(handler) __fp16: handler, _Float16: handler,
#elif defined(HELIA_HAVE_FP16_TYPE)
#define HELIA_FLOAT16_GENERIC_ROWS(handler) __fp16: handler,
#elif defined(HELIA_HAVE_FLOAT16_TYPE)
#define HELIA_FLOAT16_GENERIC_ROWS(handler) _Float16: handler,
#else
#define HELIA_FLOAT16_GENERIC_ROWS(handler)
#endif

/*
 * The alternative format reaches HELIA_HAVE_FP16_TYPE because the issue #54
 * guard has to cover every half-typed element the target can declare, but
 * helia_test_float_class_binary16() below would read 0x7C00 as
 * +Inf where that format means 65536.0. Rejecting the build is the only honest
 * answer: a validator that silently reclassifies finite kernel output is worse
 * than no build at all.
 */
#if defined(__ARM_FP16_FORMAT_ALTERNATIVE)
#error "__ARM_FP16_FORMAT_ALTERNATIVE has no Inf/NaN encodings; the binary16 class decoder is IEEE-only (helia-core-tester issue #75)"
#endif

_Static_assert(sizeof(float) == 4, "float is not IEEE-754 binary32 on this target");
_Static_assert(sizeof(double) == 8, "double is not IEEE-754 binary64 on this target");
#ifdef HELIA_HAVE_FP16_TYPE
_Static_assert(sizeof(__fp16) == 2, "__fp16 is not IEEE-754 binary16 on this target");
#endif
#ifdef HELIA_HAVE_FLOAT16_TYPE
_Static_assert(sizeof(_Float16) == 2, "_Float16 is not IEEE-754 binary16 on this target");
#endif

static inline int helia_test_float_class_binary16(const void *storage)
{
    uint16_t helia_bits = 0;
    memcpy(&helia_bits, storage, sizeof(helia_bits));
    if ((helia_bits & 0x7C00u) != 0x7C00u) {
        return HELIA_FLOAT_CLASS_FINITE;
    }
    if ((helia_bits & 0x03FFu) != 0u) {
        return HELIA_FLOAT_CLASS_NAN;
    }
    return (helia_bits & 0x8000u) ? HELIA_FLOAT_CLASS_NEG_INF : HELIA_FLOAT_CLASS_POS_INF;
}

static inline int helia_test_float_class_binary32(const void *storage)
{
    uint32_t helia_bits = 0;
    memcpy(&helia_bits, storage, sizeof(helia_bits));
    if ((helia_bits & 0x7F800000u) != 0x7F800000u) {
        return HELIA_FLOAT_CLASS_FINITE;
    }
    if ((helia_bits & 0x007FFFFFu) != 0u) {
        return HELIA_FLOAT_CLASS_NAN;
    }
    return (helia_bits & 0x80000000u) ? HELIA_FLOAT_CLASS_NEG_INF : HELIA_FLOAT_CLASS_POS_INF;
}

static inline int helia_test_float_class_binary64(const void *storage)
{
    uint64_t helia_bits = 0;
    memcpy(&helia_bits, storage, sizeof(helia_bits));
    if ((helia_bits & 0x7FF0000000000000ull) != 0x7FF0000000000000ull) {
        return HELIA_FLOAT_CLASS_FINITE;
    }
    if ((helia_bits & 0x000FFFFFFFFFFFFFull) != 0ull) {
        return HELIA_FLOAT_CLASS_NAN;
    }
    return (helia_bits & 0x8000000000000000ull) ? HELIA_FLOAT_CLASS_NEG_INF
                                                : HELIA_FLOAT_CLASS_POS_INF;
}

#ifdef __cplusplus
}
#endif

/*
 * Dispatches on the element's declared type so the decode happens at the
 * element's own storage width, and passes the element by address so the bytes
 * are read from where the kernel wrote them rather than from a copy the
 * compiler has watched go through an FP register.
 */
#define HELIA_FLOAT_CLASS_OF(element) \
    _Generic((element), \
        HELIA_FLOAT16_GENERIC_ROWS(helia_test_float_class_binary16) \
        float: helia_test_float_class_binary32, \
        double: helia_test_float_class_binary64 \
    )(&(element))

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
 * of silently passing forever. The half types are covered wherever the target
 * has one, which is every configuration that can compile float16 test data in
 * the first place.
 */
#define HELIA_ELEM_IS_FLOAT(expr) \
    _Generic((expr), HELIA_FLOAT16_GENERIC_ROWS(1) float: 1, double: 1, default: 0)
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
 * greps this one line out of every float test's stdout.
 *
 * Sentinels on the printed line (a real headroom number is never negative):
 *   maxfrac = -1.0  a per-element tolerance budget was exactly zero and the
 *                   diff was nonzero -- an unrepresentable "infinite" fraction
 *                   rather than a real headroom number. helia_max_frac itself
 *                   stays at 0.0; helia_zero_tol_violation gates the -1.0 at
 *                   print time.
 *   maxdiff = -1.0, maxfrac = -2.0  headroom is unmeasurable for this tensor:
 *                   a non-finite element mismatched, or no finite element was
 *                   compared at all. Matched non-finite elements are skipped
 *                   rather than voiding the measurement, so a tensor carrying a
 *                   few NaN lanes and finite values elsewhere still reports real
 *                   headroom for the finite ones. Either way NaN/Inf stays out
 *                   of the printf.
 *
 * Non-finite elements (issue #75) are classified straight from the element's
 * storage bytes, before the widening conversion and before the tolerance is
 * computed. Before the tolerance, because the tolerance is what goes bad:
 * rtol * |Inf| is Inf and 0 * |Inf| is NaN, and `diff > tol` is false against
 * either, so a tolerance derived from a non-finite expected value can never be
 * exceeded. Before the conversion, because under -ffinite-math-only the
 * conversion itself is what erases the evidence (see HELIA_FLOAT_CLASS_OF).
 * The count of mismatched non-finite elements is reported on its own
 * HELIA_NONFINITE_MISMATCHES line, so the reporting parser can tell this
 * defect from a tolerance overrun without inferring it from the headroom
 * sentinel, which has two causes. A matched pair (both NaN, or infinities of
 * the same sign) passes: ns-cmsis-nn's
 * Include/arm_nnfunctions_flt.h guarantees NaN-ness, not payload, so a matched
 * NaN passes regardless of sign or payload (see AmbiqAI/ns-cmsis-nn#333). Any
 * other pairing, including infinities of opposite sign, is a failure reported
 * through helia_test_nonfinite_mismatch() rather than the %f mismatch line.
 */
#define HELIA_VALIDATE_FLOATS(actual, expected, size, atol, rtol, max_reports, failures) \
    do { \
        double helia_max_diff = 0.0; \
        double helia_max_frac = 0.0; \
        int helia_zero_tol_violation = 0; \
        int helia_nonfinite_mismatches = 0; \
        int helia_finite_compared = 0; \
        for (int helia_i = 0; helia_i < (size); ++helia_i) { \
            int helia_act_class = HELIA_FLOAT_CLASS_OF((actual)[helia_i]); \
            int helia_exp_class = HELIA_FLOAT_CLASS_OF((expected)[helia_i]); \
            if (helia_act_class != HELIA_FLOAT_CLASS_FINITE \
                || helia_exp_class != HELIA_FLOAT_CLASS_FINITE) { \
                if (helia_act_class != helia_exp_class) { \
                    ++helia_nonfinite_mismatches; \
                    ++(failures); \
                    if ((failures) <= (max_reports)) { \
                        helia_test_nonfinite_mismatch( \
                            helia_i, \
                            helia_exp_class, \
                            (double)((expected)[helia_i]), \
                            helia_act_class, \
                            (double)((actual)[helia_i]) \
                        ); \
                    } \
                } \
                continue; \
            } \
            helia_finite_compared = 1; \
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
        helia_test_nonfinite_mismatch_summary(helia_nonfinite_mismatches); \
        int helia_unmeasurable = (helia_nonfinite_mismatches > 0 || !helia_finite_compared); \
        printf( \
            "HELIA_FLOAT_MAXDIFF maxdiff=%.8e maxfrac=%.6f n=%d\r\n", \
            helia_unmeasurable ? -1.0 : helia_max_diff, \
            helia_unmeasurable ? -2.0 : (helia_zero_tol_violation ? -1.0 : helia_max_frac), \
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
