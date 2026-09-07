#include "test_runtime/helia_test_runtime.h"

#include <stdio.h>
#include <string.h>

#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif

void helia_test_platform_init(void)
{
#ifdef USING_FVP_CORSTONE_300
    uart_init();
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
}

static void helia_test_signal_eot(void)
{
    // 0x04 (EOT) triggers FVP shutdown when:
    //   -C mps3_board.uart0.shutdown_on_eot=1
    putchar(4);
    fflush(stdout);
}

void helia_test_finish(int32_t failures)
{
    (void)failures;
    helia_test_signal_eot();
    while (1) { }
}

void HardFault_Handler(void)
{
    printf("HardFault\r\n");
    helia_test_signal_eot();
    while (1) { }
}

void helia_test_print_failure_count(int failures)
{
    printf("%d Failures\r\n", failures);
}

int helia_test_status_failure(const char *label, int status)
{
    printf("%s failed with status %d\r\n", label, status);
    helia_test_print_failure_count(1);
    return 1;
}

int helia_test_expected_status_failure(const char *label, int status, int expected_status)
{
    printf("%s failed with status %d (expected %d)\r\n", label, status, expected_status);
    helia_test_print_failure_count(1);
    return 1;
}

int helia_test_scalar_int_mismatch(const char *label, const char *subject, int expected, int actual)
{
    printf("%s %s mismatch: expected %d got %d\r\n", label, subject, expected, actual);
    helia_test_print_failure_count(1);
    return 1;
}

int helia_test_finish_validation(int failures)
{
    helia_test_print_failure_count(failures);
    return failures;
}

/*
 * Kept out of the header, and out of every harness translation unit with it:
 * this classifies a value that has already been widened to double, which is
 * exactly the shape issue #75 is about. It is safe only here, where `expected`
 * arrives as a function argument no fast-math-attributed instruction of this
 * TU produced. Harness code classifies through HELIA_FLOAT_CLASS_OF instead,
 * which decodes the element at its own storage width.
 */
static int helia_test_float_class(double value)
{
    return helia_test_float_class_binary64(&value);
}

double helia_test_float_tolerance(double expected, double atol, double rtol)
{
    // A tolerance derived from a non-finite expected value is itself Inf or
    // NaN, and `diff > tol` is false against either -- a budget that can never
    // be exceeded (issue #75). Callers classify non-finite elements before
    // reaching here; this keeps the function safe for any other caller.
    if (helia_test_float_class(expected) != HELIA_FLOAT_CLASS_FINITE) {
        return atol;
    }
    return (atol + (rtol * fabs(expected)));
}

/*
 * %.6g rather than %.6f: the finite half of a mixed pairing is often at the
 * extremes of the range (FLT_MAX against +Inf), where a fixed-point rendering
 * runs to 39 integer digits and truncates into a mangled number.
 */
static const char *helia_test_float_token(
    int value_class,
    double value,
    char *buffer,
    size_t buffer_size
)
{
    switch (value_class) {
        case HELIA_FLOAT_CLASS_NAN:
            return "nan";
        case HELIA_FLOAT_CLASS_POS_INF:
            return "+inf";
        case HELIA_FLOAT_CLASS_NEG_INF:
            return "-inf";
        default:
            break;
    }
    snprintf(buffer, buffer_size, "%.6g", value);
    return buffer;
}

/*
 * The caller passes the class it decoded from the element's storage rather
 * than letting this function re-derive it: reaching a double here costs a
 * widening conversion, which under -ffinite-math-only is licensed to produce
 * anything at all for a NaN or Inf operand. Only the finite half of a mixed
 * pairing is ever rendered from the value.
 */
void helia_test_nonfinite_mismatch(
    int index,
    int expected_class,
    double expected,
    int actual_class,
    double actual
)
{
    char expected_text[64];
    char actual_text[64];
    printf(
        "HELIA_NONFINITE_MISMATCH[%d]: exp=%s got=%s\r\n",
        index,
        helia_test_float_token(expected_class, expected, expected_text, sizeof(expected_text)),
        helia_test_float_token(actual_class, actual, actual_text, sizeof(actual_text))
    );
}

/*
 * Per-tensor count, so the reporting parser can distinguish a non-finite
 * mismatch from a tolerance overrun even when the per-element reports were
 * exhausted by earlier failures. The headroom sentinel cannot carry that
 * distinction: it also fires for a tensor with no finite element to measure.
 */
void helia_test_nonfinite_mismatch_summary(int count)
{
    if (count > 0) {
        printf("HELIA_NONFINITE_MISMATCHES n=%d\r\n", count);
    }
}

void helia_guard_arm(uint8_t *head, uint8_t *tail, void *body, size_t body_bytes, bool poison_body)
{
    memset(head, HELIA_GUARD_CANARY_BYTE, HELIA_GUARD_BYTES);
    memset(tail, HELIA_GUARD_CANARY_BYTE, HELIA_GUARD_BYTES);
    if (poison_body && body != NULL && body_bytes != 0) {
        memset(body, HELIA_GUARD_POISON_BYTE, body_bytes);
    }
}

void helia_guard_check(const char *label, const uint8_t *head, const uint8_t *tail, int *failures)
{
    unsigned int i;
    int head_breach = 0;
    int tail_breach = 0;
    for (i = 0; i < HELIA_GUARD_BYTES; ++i) {
        if (head[i] != HELIA_GUARD_CANARY_BYTE) {
            head_breach = 1;
            break;
        }
    }
    for (i = 0; i < HELIA_GUARD_BYTES; ++i) {
        if (tail[i] != HELIA_GUARD_CANARY_BYTE) {
            tail_breach = 1;
            break;
        }
    }
    if (head_breach || tail_breach) {
        ++(*failures);
        /* A corrupted head canary is a write before the buffer (underrun); a
         * corrupted tail canary is a write past it (overrun). Report which
         * so the direction of the boundary defect isn't misidentified. */
        printf("GuardBreach[%s]: %s%s%s detected (canary corrupted)\r\n",
               label,
               head_breach ? "underrun" : "",
               (head_breach && tail_breach) ? " and " : "",
               tail_breach ? "overrun" : "");
    }
}
