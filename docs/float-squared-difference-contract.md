# Float squared difference: what the tester asserts

Kernel: `arm_elementwise_squared_difference_f16` (AmbiqAI/ns-cmsis-nn#490),
declared in `Include/arm_nnfunctions_flt.h`. There is no f32 kernel; an FP32
descriptor is rejected at kernel selection. Descriptors:
`assets/descriptors/BasicMathFunctions/squared_difference_float.yaml`.

## The kernel

Flat: two equal-length operands, one output, `block_size`; no dims, no
broadcast, no activation clamp. Two legs behind one argument guard:

- MVE (`ARM_MATH_MVE_FLOAT16` without `ARM_MATH_AUTOVECTORIZE`): a
  tail-predicated loop of 8 halves per iteration, `vsubq` then `vmulq`.
- Scalar: a `_Float16` loop, `diff = a - b; out = diff * diff`.
- Guard: any NULL pointer or `block_size < 1` returns `ARM_CMSIS_NN_ARG_ERROR`
  before either loop.

Per element the kernel performs two IEEE-754 binary16 operations, each rounded
once. That is the whole contract the header states; it says nothing about
non-finite inputs.

## Golden model

`OpSquaredDifference._float_reference()` computes the difference in
float64 (exact for two halves, which need at most 40 significand bits),
narrows it once to binary16, squares in float64 (exact, 22 bits) and narrows
once more. That is bit for bit the result of the two rounded operations both
legs perform. NumPy's own half arithmetic widens to float32 per operation and
can double-round, which is why the model does not use it.

Random-draw cases compare at the float suite's FP16 default (`atol` and
`rtol` of 1e-3). The pinned cases below set `hint.extras.bit_exact: true` and
compare storage bits through `HELIA_VALIDATE_FLOAT_BITS`: their golden is
exact by construction, so any difference is a real defect, and the bit
comparison also asserts the sign of zero, which a `fabs(actual - expected)`
tolerance cannot see.

## Cases

Block sizes are chosen against the 8-lane MVE loop: 1 and 7 run only a
predicated partial vector; 8, 16, 128 and 512 run only full vectors; 9, 15
and 45 run full vectors plus a tail (of 1, 7 and 5 lanes). The two non-finite sweeps put the tokens in
the first vector and in the final partial vector respectively.

Pinned operands (`hint.extras.input_1_values` / `input_2_values`, flat NHWC,
exact binary16 values, validated at generation time):

| case | what it pins |
| --- | --- |
| `overflow` | 65536 overflows to +Inf; 255.875 squares to 65472, the largest finite square of a half; the two ties of the difference rounding (255.9375 rounds up to 256 and overflows, 255.8125 rounds down to 255.75); a difference that itself overflows; 32761 rounding up across the 2^15 binade. |
| `underflow` | subnormal squares (2^-16, 2^-20, 2^-22), the smallest normal (2^-14) and smallest subnormal (2^-24) results, 2^-26 rounding to zero, the smallest subnormal squared, and `-0 - 0` squaring to +0. A leg that flushes subnormal inputs or results is caught here. |
| `equal_operands` | identical operands across the finite range, so every square is +0. |

Non-finite: both sweep cases use `nonfinite_policy: mask` because the header
carries no NaN/Inf contract (README, "Non-finite inputs"). They assert
`SUCCESS`, that every finite lane matches, and that the token lanes did not
poison the rest of their vector. Flip them to `strict` once ns-cmsis-nn
documents the behaviour.

Faults: one case per operand of the guard's short-circuit chain
(`null_input_1`, `null_input_2`, `null_output`) and both sides of the
`block_size` bound (`zero_block`, `negative_block`). Each asserts
`ARM_CMSIS_NN_ARG_ERROR`; the cases that pass an output buffer also poison it
first and require every byte to survive (`HELIA_GUARD_CHECK_UNTOUCHED`).

## Coverage

With the cases above, the kernel's `Source/BasicMathFunctions/
arm_elementwise_squared_difference_f16.c` is fully covered per leg: the MVE
loop in the `--coverage --coverage-mve-float` cortex-m55 f16 leg, the scalar
loop in the plain `--coverage` leg, and the guard's return in both. Every
branch of the guard is taken in both directions across the case set.

Every descriptor carries `required_kernel_symbols`, so a checkout that
predates ns-cmsis-nn#490 skips the file rather than failing the build.

Reproduce locally (the FVP is Linux-only; the CI image works on macOS Docker
with `--platform linux/amd64`):

```bash
uv run helia_core_tester full --cpu cortex-m55 --suite float --float-precision f16 \
  --op SquaredDifference --coverage --coverage-mve-float \
  --cmsis-nn-root <ns-cmsis-nn checkout with #490>
```
