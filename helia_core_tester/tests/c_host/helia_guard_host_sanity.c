/*
 * Host driver for the issue #68 buffer-overrun guards: declares buffers with
 * the same helia_guard_declare expansion the generated tests use (rendered
 * by test_guard_mechanism.py into helia_guard_host_decls.h), lets a stub
 * kernel write in and around them, and prints the GuardBreach lines plus a
 * failures count per case so the pytest can assert the detector's contract.
 */
#include "test_runtime/helia_test_runtime.h"
#include "helia_guard_host_decls.h"

#define BODY_COUNT 8
#define SCRATCH_BYTES 64
#define SCRATCH_USED 24

typedef enum {
    WRITE_IN_BOUNDS,
    WRITE_PAST_END,
    WRITE_BEFORE_START,
    WRITE_BOTH_ENDS,
} kernel_fault;

static void stub_kernel(int32_t *out, int n, kernel_fault fault)
{
    int i;
    for (i = 0; i < n; ++i) {
        out[i] = i;
    }
    if (fault == WRITE_PAST_END || fault == WRITE_BOTH_ENDS) {
        out[n] = 0x7F;
    }
    if (fault == WRITE_BEFORE_START || fault == WRITE_BOTH_ENDS) {
        out[-1] = 0x7F;
    }
}

static void stub_scratch_kernel(uint8_t *scratch, size_t used, int overrun_bytes)
{
    memset(scratch, 0x11, used + (size_t)overrun_bytes);
}

static void run_output_case(const char *case_id, kernel_fault fault)
{
    int failures = 0;
    printf("CASE %s\r\n", case_id);
    HELIA_GUARD_ARM(sanity_output, false);
    stub_kernel(sanity_output, BODY_COUNT, fault);
    HELIA_GUARD_CHECK(sanity_output, "sanity output", failures);
    printf("RESULT %s failures=%d\r\n", case_id, failures);
}

static void run_scratch_case(const char *case_id, int overrun_bytes)
{
    int failures = 0;
    size_t i;
    printf("CASE %s\r\n", case_id);
    HELIA_GUARD_ARM(sanity_scratch, true);
    for (i = 0; i < SCRATCH_BYTES; ++i) {
        if (sanity_scratch[i] != HELIA_GUARD_POISON_BYTE) {
            printf("POISON_MISSING at %u\r\n", (unsigned)i);
        }
    }
    HELIA_GUARD_STAMP_SLACK(sanity_scratch, SCRATCH_USED);
    stub_scratch_kernel(sanity_scratch, SCRATCH_USED, overrun_bytes);
    HELIA_GUARD_CHECK(sanity_scratch, "sanity scratch", failures);
    HELIA_GUARD_CHECK_SLACK(sanity_scratch, "sanity scratch slack", SCRATCH_USED, failures);
    printf("RESULT %s failures=%d\r\n", case_id, failures);
}

static void run_slack_edge_cases(void)
{
    int failures = 0;
    printf("CASE slack_no_room\r\n");
    HELIA_GUARD_ARM(sanity_scratch, true);
    HELIA_GUARD_STAMP_SLACK(sanity_scratch, SCRATCH_BYTES);
    memset(sanity_scratch, 0x22, SCRATCH_BYTES);
    HELIA_GUARD_CHECK(sanity_scratch, "sanity scratch", failures);
    HELIA_GUARD_CHECK_SLACK(sanity_scratch, "sanity scratch slack", SCRATCH_BYTES, failures);
    printf("RESULT slack_no_room failures=%d\r\n", failures);

    failures = 0;
    printf("CASE slack_short\r\n");
    HELIA_GUARD_ARM(sanity_scratch, true);
    HELIA_GUARD_STAMP_SLACK(sanity_scratch, SCRATCH_BYTES - 4);
    memset(sanity_scratch, 0x22, SCRATCH_BYTES - 4);
    HELIA_GUARD_CHECK_SLACK(sanity_scratch, "sanity scratch slack", SCRATCH_BYTES - 4, failures);
    printf("RESULT slack_short failures=%d\r\n", failures);

    failures = 0;
    printf("CASE slack_short_overrun\r\n");
    HELIA_GUARD_ARM(sanity_scratch, true);
    HELIA_GUARD_STAMP_SLACK(sanity_scratch, SCRATCH_BYTES - 4);
    memset(sanity_scratch, 0x22, SCRATCH_BYTES - 3);
    HELIA_GUARD_CHECK_SLACK(sanity_scratch, "sanity scratch slack", SCRATCH_BYTES - 4, failures);
    printf("RESULT slack_short_overrun failures=%d\r\n", failures);
}

int main(void)
{
    run_output_case("output_clean", WRITE_IN_BOUNDS);
    run_output_case("output_overrun", WRITE_PAST_END);
    run_output_case("output_underrun", WRITE_BEFORE_START);
    run_output_case("output_both", WRITE_BOTH_ENDS);
    run_scratch_case("scratch_clean", 0);
    run_scratch_case("scratch_slack_overrun", 1);
    run_scratch_case("scratch_slack_deep_overrun", SCRATCH_BYTES - SCRATCH_USED + 1);
    run_slack_edge_cases();
    printf("HOST_SANITY_DONE\r\n");
    return 0;
}
