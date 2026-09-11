#include "convolve_float_default_f16_convolve.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdint.h>
#include "test_runtime/helia_test_runtime.h"


#ifdef HELIA_BENCHMARK_MODE

#ifndef HELIA_BENCHMARK_WARMUP_RUNS
#define HELIA_BENCHMARK_WARMUP_RUNS 3
#endif
#ifndef HELIA_BENCHMARK_MEASURED_RUNS
#define HELIA_BENCHMARK_MEASURED_RUNS 10
#endif

#if !defined(__CORTEX_M) || (__CORTEX_M < 4)
#error "HELIA_BENCHMARK_MODE requires a Cortex-M4/M55 target with a DWT cycle counter; cortex-m0 has none (see Config._validate_hardware_benchmark)."
#endif

typedef int32_t (*helia_bench_op_fn)(void);

static inline void helia_dwt_enable(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline uint32_t helia_dwt_cycles(void)
{
    return DWT->CYCCNT;
}

/*
 * Runs `op` HELIA_BENCHMARK_WARMUP_RUNS times untimed (cache/branch-predictor
 * warmup, no cycle counting), then HELIA_BENCHMARK_MEASURED_RUNS times timed
 * via the DWT cycle counter. Only the op call itself is timed -- never
 * `_bench_init()` (buffer setup / weight-sum precompute) and never output
 * validation/compare, which happen separately in the non-benchmark path.
 */
static inline void helia_benchmark_run(const char *name, helia_bench_op_fn op)
{
    helia_dwt_enable();

    for (int i = 0; i < HELIA_BENCHMARK_WARMUP_RUNS; i++) {
        (void)op();
    }

    printf("[BENCH] %s warmup_runs=%d measured_runs=%d\r\n",
           name, HELIA_BENCHMARK_WARMUP_RUNS, HELIA_BENCHMARK_MEASURED_RUNS);

    for (int i = 0; i < HELIA_BENCHMARK_MEASURED_RUNS; i++) {
        uint32_t start = helia_dwt_cycles();
        (void)op();
        uint32_t end = helia_dwt_cycles();
        printf("[PERF] %s: %lu cycles\r\n", name, (unsigned long)(end - start));
    }
}

#endif // HELIA_BENCHMARK_MODE

// Context for buffer allocation
static cmsis_nn_context convolve_float_default_f16_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
// Buffer size calculated conservatively to handle MVE and DSP implementations
#define CONVOLVE_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX 1024
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    uint8_t body[CONVOLVE_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX];
    uint8_t tail[HELIA_GUARD_BYTES];
} convolve_float_default_f16_buffer_guard;
#define convolve_float_default_f16_buffer (convolve_float_default_f16_buffer_guard.body)


#define CONVOLVE_FLOAT_DEFAULT_F16_OUTPUT_SIZE (1 * 6 * 6 * 5)
static struct {
    uint8_t head[HELIA_GUARD_BYTES];
    float16_t body[CONVOLVE_FLOAT_DEFAULT_F16_OUTPUT_SIZE];
    uint8_t tail[HELIA_GUARD_BYTES];
} convolve_float_default_f16_output_guard;
#define convolve_float_default_f16_output (convolve_float_default_f16_output_guard.body)

// Bias dimensions
static const cmsis_nn_dims convolve_float_default_f16_bias_dims = {
    .n = 0, .h = 0, .w = 0, .c = 5
};

int32_t convolve_float_default_f16_run(
    const float16_t* __restrict input,
    float16_t* __restrict output
) {
        // Calculate required buffer size
    int32_t required_buffer_size = arm_convolve_f16_get_buffer_size(
        &convolve_float_default_f16_conv_params,
        &convolve_float_default_f16_input_dims,
        &convolve_float_default_f16_filter_dims,
        &convolve_float_default_f16_output_dims,
        ARM_NN_LAYOUT_NHWC
    );
    // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).
    HELIA_GUARD_ARM(convolve_float_default_f16_buffer, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_STAMP_SLACK(convolve_float_default_f16_buffer, 0u);
    // The slack is stamped as wholly unused here so that an early return from the
    // capacity check below leaves every canary in a checked state; it is re-stamped
    // with the real size once the context is populated (#68).


    // The sizer's answer is checked before it becomes a context size (#133). A negative
    // answer is the documented out-of-range sentinel and never a usable size; an answer
    // above this case's static means our generation-time bound and the shipped kernel
    // disagree. They are separate failures because they have separate owners.
    HELIA_VALIDATE_SIZER("arm_convolve_f16_get_buffer_size", required_buffer_size);
    HELIA_VALIDATE_SIZER_FITS("arm_convolve_f16_get_buffer_size", required_buffer_size, CONVOLVE_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX);

    // Initialize context buffer
    convolve_float_default_f16_ctx.buf = convolve_float_default_f16_buffer;
    convolve_float_default_f16_ctx.size = required_buffer_size;
    HELIA_GUARD_STAMP_SLACK(convolve_float_default_f16_buffer, convolve_float_default_f16_ctx.buf == convolve_float_default_f16_buffer ? (size_t)convolve_float_default_f16_ctx.size : 0u);


        // Run convolution - different signatures for s8 vs s16
    return arm_convolve_f16(
        &convolve_float_default_f16_ctx,
        &convolve_float_default_f16_conv_params,
        &convolve_float_default_f16_input_dims,
        input,
        &convolve_float_default_f16_filter_dims,
        convolve_float_default_f16_weights,
        &convolve_float_default_f16_bias_dims,
        convolve_float_default_f16_biases,
        &convolve_float_default_f16_output_dims,
        output,
        ARM_NN_LAYOUT_NHWC
    );

}

#ifdef HELIA_BENCHMARK_MODE
// --benchmark support: convolve_float_default_f16_bench_init() is the one-time buffer/context/
// weight-sum setup (untimed); convolve_float_default_f16_bench_op() is *only* the kernel call
// (timed, 3 warmup + 10 measured runs by default -- see common/standalone/benchmark.j2).
static int32_t convolve_float_default_f16_bench_init(void)
{
        // Calculate required buffer size
    int32_t required_buffer_size = arm_convolve_f16_get_buffer_size(
        &convolve_float_default_f16_conv_params,
        &convolve_float_default_f16_input_dims,
        &convolve_float_default_f16_filter_dims,
        &convolve_float_default_f16_output_dims,
        ARM_NN_LAYOUT_NHWC
    );
    // Armed before the capacity check below: an early return there would otherwise leave
    // these canaries unstamped, and the unconditional check in _test_case_run would
    // report a fabricated breach instead of the real sizer error (#68).
    HELIA_GUARD_ARM(convolve_float_default_f16_buffer, true /* pure scratch: poison to catch read-before-write */);
    HELIA_GUARD_STAMP_SLACK(convolve_float_default_f16_buffer, 0u);
    // The slack is stamped as wholly unused here so that an early return from the
    // capacity check below leaves every canary in a checked state; it is re-stamped
    // with the real size once the context is populated (#68).


    // The sizer's answer is checked before it becomes a context size (#133). A negative
    // answer is the documented out-of-range sentinel and never a usable size; an answer
    // above this case's static means our generation-time bound and the shipped kernel
    // disagree. They are separate failures because they have separate owners.
    HELIA_VALIDATE_SIZER("arm_convolve_f16_get_buffer_size", required_buffer_size);
    HELIA_VALIDATE_SIZER_FITS("arm_convolve_f16_get_buffer_size", required_buffer_size, CONVOLVE_FLOAT_DEFAULT_F16_BUFFER_SIZE_MAX);

    // Initialize context buffer
    convolve_float_default_f16_ctx.buf = convolve_float_default_f16_buffer;
    convolve_float_default_f16_ctx.size = required_buffer_size;
    HELIA_GUARD_STAMP_SLACK(convolve_float_default_f16_buffer, convolve_float_default_f16_ctx.buf == convolve_float_default_f16_buffer ? (size_t)convolve_float_default_f16_ctx.size : 0u);


    return ARM_CMSIS_NN_SUCCESS;
}

static int32_t convolve_float_default_f16_bench_op(void)
{
        // Run convolution - different signatures for s8 vs s16
    return arm_convolve_f16(
        &convolve_float_default_f16_ctx,
        &convolve_float_default_f16_conv_params,
        &convolve_float_default_f16_input_dims,
        convolve_float_default_f16_input,
        &convolve_float_default_f16_filter_dims,
        convolve_float_default_f16_weights,
        &convolve_float_default_f16_bias_dims,
        convolve_float_default_f16_biases,
        &convolve_float_default_f16_output_dims,
        convolve_float_default_f16_output,
        ARM_NN_LAYOUT_NHWC
    );

}

static void convolve_float_default_f16_benchmark_run(void)
{
    // A sizer check inside _bench_init() returns before the context is populated, so the
    // benchmark must not proceed on that path: _bench_op() would call the kernel with an
    // uninitialised context and time whatever happened, recording cycle counts that mean
    // nothing, or crash. The failing check has already printed its marker naming the
    // sizer; this reports the skip and runs nothing (#133).
    //
    // What this still does not do is fail the case. A benchmark reports cycles, not a
    // verdict, and helia_benchmark_run() has no failure channel to carry one, so the
    // run ends with zero failures either way. The non-benchmark run of the same case is
    // what turns a bad sizer answer into a verdict.
    if (convolve_float_default_f16_bench_init() != ARM_CMSIS_NN_SUCCESS) {
        printf("[BENCH] convolve_float_default_f16 skipped: scratch sizer rejected before the context was populated\r\n");
        return;
    }
    helia_benchmark_run("convolve_float_default_f16", convolve_float_default_f16_bench_op);
}
#endif // HELIA_BENCHMARK_MODE

int32_t convolve_float_default_f16_test_case_run(void)
{
    HELIA_GUARD_ARM(convolve_float_default_f16_output, false /* real output, not scratch: don't poison */);
    int32_t status = convolve_float_default_f16_run(convolve_float_default_f16_input, convolve_float_default_f16_output);
    int failures = 0;
    HELIA_GUARD_CHECK(convolve_float_default_f16_buffer, "Convolve scratch", failures);
    HELIA_GUARD_CHECK_SLACK(convolve_float_default_f16_buffer, "Convolve scratch slack", convolve_float_default_f16_ctx.buf == convolve_float_default_f16_buffer ? (size_t)convolve_float_default_f16_ctx.size : 0u, failures);
    HELIA_GUARD_CHECK(convolve_float_default_f16_output, "Convolve output", failures);
    HELIA_VALIDATE_STATUS("Convolve", status);

    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        convolve_float_default_f16_output,
        convolve_float_default_f16_expected_output,
        CONVOLVE_FLOAT_DEFAULT_F16_OUTPUT_SIZE,
        1,
        0.001f,
        0.001f,
        20,
        failures
    );
    HELIA_VALIDATE_RETURN_FAILURES(failures);
}

int main(void)
{
    helia_test_platform_init();
#ifdef HELIA_BENCHMARK_MODE
    convolve_float_default_f16_benchmark_run();
    helia_test_finish(0);
#else
    int32_t failures = convolve_float_default_f16_test_case_run();
    helia_test_finish(failures);
#endif
}