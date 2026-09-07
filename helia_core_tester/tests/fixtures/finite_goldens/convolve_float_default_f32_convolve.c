#include "convolve_float_default_f32_convolve.h"
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
static cmsis_nn_context convolve_float_default_f32_ctx;

// Runtime scratch buffer (max upper bound; actual size queried at runtime)
// Buffer size calculated conservatively to handle MVE and DSP implementations
#define CONVOLVE_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX 1692
static uint8_t convolve_float_default_f32_buffer[CONVOLVE_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX];


#define CONVOLVE_FLOAT_DEFAULT_F32_OUTPUT_SIZE (1 * 6 * 6 * 5)
static float convolve_float_default_f32_output[CONVOLVE_FLOAT_DEFAULT_F32_OUTPUT_SIZE];

// Bias dimensions
static const cmsis_nn_dims convolve_float_default_f32_bias_dims = {
    .n = 0, .h = 0, .w = 0, .c = 5
};

int32_t convolve_float_default_f32_run(
    const float* __restrict input,
    float* __restrict output
) {
        // Calculate required buffer size
    int32_t required_buffer_size = arm_convolve_f32_get_buffer_size(
        &convolve_float_default_f32_conv_params,
        &convolve_float_default_f32_input_dims,
        &convolve_float_default_f32_filter_dims,
        &convolve_float_default_f32_output_dims,
        ARM_NN_LAYOUT_NHWC
    );

    if (required_buffer_size > CONVOLVE_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Initialize context buffer
    convolve_float_default_f32_ctx.buf = convolve_float_default_f32_buffer;
    convolve_float_default_f32_ctx.size = required_buffer_size;


        // Run convolution - different signatures for s8 vs s16
    return arm_convolve_f32(
        &convolve_float_default_f32_ctx,
        &convolve_float_default_f32_conv_params,
        &convolve_float_default_f32_input_dims,
        input,
        &convolve_float_default_f32_filter_dims,
        convolve_float_default_f32_weights,
        &convolve_float_default_f32_bias_dims,
        convolve_float_default_f32_biases,
        &convolve_float_default_f32_output_dims,
        output,
        ARM_NN_LAYOUT_NHWC
    );

}

#ifdef HELIA_BENCHMARK_MODE
// --benchmark support: convolve_float_default_f32_bench_init() is the one-time buffer/context/
// weight-sum setup (untimed); convolve_float_default_f32_bench_op() is *only* the kernel call
// (timed, 3 warmup + 10 measured runs by default -- see common/standalone/benchmark.j2).
static int32_t convolve_float_default_f32_bench_init(void)
{
        // Calculate required buffer size
    int32_t required_buffer_size = arm_convolve_f32_get_buffer_size(
        &convolve_float_default_f32_conv_params,
        &convolve_float_default_f32_input_dims,
        &convolve_float_default_f32_filter_dims,
        &convolve_float_default_f32_output_dims,
        ARM_NN_LAYOUT_NHWC
    );

    if (required_buffer_size > CONVOLVE_FLOAT_DEFAULT_F32_BUFFER_SIZE_MAX) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Initialize context buffer
    convolve_float_default_f32_ctx.buf = convolve_float_default_f32_buffer;
    convolve_float_default_f32_ctx.size = required_buffer_size;


    return ARM_CMSIS_NN_SUCCESS;
}

static int32_t convolve_float_default_f32_bench_op(void)
{
        // Run convolution - different signatures for s8 vs s16
    return arm_convolve_f32(
        &convolve_float_default_f32_ctx,
        &convolve_float_default_f32_conv_params,
        &convolve_float_default_f32_input_dims,
        convolve_float_default_f32_input,
        &convolve_float_default_f32_filter_dims,
        convolve_float_default_f32_weights,
        &convolve_float_default_f32_bias_dims,
        convolve_float_default_f32_biases,
        &convolve_float_default_f32_output_dims,
        convolve_float_default_f32_output,
        ARM_NN_LAYOUT_NHWC
    );

}

static void convolve_float_default_f32_benchmark_run(void)
{
    convolve_float_default_f32_bench_init();
    helia_benchmark_run("convolve_float_default_f32", convolve_float_default_f32_bench_op);
}
#endif // HELIA_BENCHMARK_MODE

int32_t convolve_float_default_f32_test_case_run(void)
{
    int32_t status = convolve_float_default_f32_run(convolve_float_default_f32_input, convolve_float_default_f32_output);
    HELIA_VALIDATE_STATUS("Convolve", status);

    int failures = 0;
    HELIA_VALIDATE_OUTPUTS(
        FLOAT,
        convolve_float_default_f32_output,
        convolve_float_default_f32_expected_output,
        CONVOLVE_FLOAT_DEFAULT_F32_OUTPUT_SIZE,
        1,
        5e-05f,
        2e-05f,
        20,
        failures
    );
    HELIA_VALIDATE_RETURN_FAILURES(failures);
}

int main(void)
{
    helia_test_platform_init();
#ifdef HELIA_BENCHMARK_MODE
    convolve_float_default_f32_benchmark_run();
    helia_test_finish(0);
#else
    int32_t failures = convolve_float_default_f32_test_case_run();
    helia_test_finish(failures);
#endif
}