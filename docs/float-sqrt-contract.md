# Float sqrt and reciprocal sqrt descriptors

Refs AmbiqAI/ns-cmsis-nn#295 and AmbiqAI/ns-cmsis-nn#477.

The 52 standalone cases cover `arm_nn_sqrt_f32/f16` and `arm_rsqrt_f32/f16`.
Each precision/operator has normal positive inputs at lengths 1, 7, 8, 9, 17,
257; mixed special values both out of place and in place; exact powers of four;
and null-input, null-output, zero-size and negative-size calls. Input and output
buffers have guards, and out-of-place calls must preserve their inputs.
`required_kernel_symbols` keeps these cases out of builds against older kernels.

Finite positive goldens come from NumPy float64 evaluation rounded to the output
precision. Both FP16 operations and FP32 sqrt compare exactly. FP32 rsqrt permits
one ULP on finite positive results, except powers of four, which compare exactly.
`HELIA_VALIDATE_FLOAT_BITS` performs the comparison on integer representations;
zero, infinity and NaN always require identical bits. These cases use this bit
validator, rather than the generic absolute/relative comparator. The generated
sidecar's `max_ulp` scalar records its limit; `comparison.atol/rtol` are zero and
do not describe the ULP exception.

| Input | sqrt | rsqrt |
| --- | --- | --- |
| +0 / -0 | +0 / -0 | +infinity / -infinity |
| +infinity | +infinity | +0 |
| Negative nonzero, including -infinity | Positive quiet NaN | Positive quiet NaN |
| NaN | Input bits with quiet bit set | Input bits with quiet bit set |

The negative-domain quiet NaN is literally `0x7fc00000` (FP32) or `0x7e00`
(FP16). A host libm negative-domain NaN is not a valid bitwise oracle.

Positive FP32 subnormals follow the processor's input flush control. The generated
harness reads FPSCR.FZ on Arm or MXCSR.DAZ on x86: flushed inputs expect +0 for
sqrt and +infinity for rsqrt. Other targets use gradual underflow. FP16
subnormals widen to normal FP32 and retain their mathematical result.

`test_sqrt_float_generation.py` checks all 65,536 FP16 encodings, using exact
rational squared-midpoint bounds for positive finite results and explicit bits
for special values. It also queries a real LiteRT reference interpreter for
finite FP32 inputs and compiles generated validators with deliberate numerical,
special-value and memory faults. The kernel Unity suites separately exhaust the
FP16 kernels themselves; the 52 generated descriptors are a compact coverage
suite, not an exhaustive kernel run.
