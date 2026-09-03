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

double helia_test_float_tolerance(double expected, double atol, double rtol)
{
    return (atol + (rtol * fabs(expected)));
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
    int breach = 0;
    for (i = 0; i < HELIA_GUARD_BYTES; ++i) {
        if (head[i] != HELIA_GUARD_CANARY_BYTE) {
            breach = 1;
            break;
        }
    }
    for (i = 0; i < HELIA_GUARD_BYTES; ++i) {
        if (tail[i] != HELIA_GUARD_CANARY_BYTE) {
            breach = 1;
            break;
        }
    }
    if (breach) {
        ++(*failures);
        printf("GuardBreach[%s]: buffer overrun detected (canary corrupted)\r\n", label);
    }
}
