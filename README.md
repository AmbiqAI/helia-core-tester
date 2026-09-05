# Helia-Core Tester

Toolkit for CMSIS-NN testing: generate test assets, build for FVP, run tests, and publish coverage.

## Quick Start

```bash
uv sync
uv run helia_core_tester --help
```

## Commands

- `uv run helia_core_tester generate`
- `uv run helia_core_tester build`
- `uv run helia_core_tester run`
- `uv run helia_core_tester full`
- `uv run helia_core_tester clean`
- `uv run helia_core_tester clean-all`
- `uv run helia_core_tester doctor`
- `uv run helia_core_tester coverage-merge`

Removed interfaces:
- `gap-check` subcommand
- `--skip-conversion`
- `--skip-runners`
- `--regen-generated-tests-after-cleanup`
- report-dir override flags
- `--include-float` (replaced by `--suite float` or `--suite both`)

## Suite-Based Runs

Run integer-only (default):

```bash
uv run helia_core_tester full --cpu cortex-m0,cortex-m4,cortex-m55 --suite int
```

Run float-only:

```bash
uv run helia_core_tester full --cpu cortex-m4,cortex-m55 --suite float --float-precision both
```

Run both suites as separate flows:

```bash
uv run helia_core_tester full --cpu cortex-m0,cortex-m4,cortex-m55 --suite both
```

## Canonical Artifacts

Generated tests:
- `artifacts/generated_tests/<suite>/<cpu>/manifest.json`
- `artifacts/generated_tests/<suite>/<cpu>/tests.cmake`
- `artifacts/generated_tests/<suite>/<cpu>/<Family>/<descriptor_name>/...`

Build outputs:
- `artifacts/build-<suite>-<cpu>-<compiler>/tests/<Family>/<descriptor_name>.elf`

Reports:
- generation: `artifacts/reports/generation/<suite>/<cpu>/`
- test execution: `artifacts/reports/tests/<suite>/<cpu>/`
- per-suite CPU coverage: `artifacts/reports/coverage/<suite>/<cpu>/`
- merged coverage: `artifacts/reports/coverage/merged/`

Generation report files (always emitted):
- `generation_summary.json`
- `generation_failures.json`
- `conversion_failures.json`
- `manifest_pointer.json`
- `capability_skips.json`

## Float Descriptor Foundation

Use `tensor_dtypes` for new float-aware descriptors. Legacy `activation_dtype` and `weight_dtype`
still work, but the loader now normalizes both styles into `resolved_tensor_dtypes` and
`resolved_comparison`.

Example:

```yaml
name: quantize_fp32_to_s8_basic
operator: Quantize
tensor_dtypes:
  input: FP32
  output: S8
input_shape: [1, 4]
```

Reference ops for float infrastructure:
- `Quantize` is the source of truth for `FP32 -> S8/S16`
- `Dequantize` is the source of truth for `S8/S16 -> FP32`

Future ops should consume resolved tensor roles rather than raw legacy dtype fields:
- `self.tensor_dtype("input")`
- `self.tensor_c_type("output")`
- `self.tensor_litert_dtype("input")`
- `self.comparison_config()`

### Non-finite inputs

A float-suite descriptor can set `input_mode: nonfinite_sweep` to overwrite the leading flat
elements of its input with non-finite tokens, leaving the remaining elements as ordinary uniform
draws. `nonfinite_tokens` names the tokens and their order; it defaults to `[nan, inf, -inf]`.
Naming a subset is how a descriptor stays inside what ns-cmsis-nn guarantees: sigmoid declares NaN
unsupported and the MVE tanh legs destroy it by design, so those descriptors sweep `[inf, -inf]`
only. Expected outputs come from the same reference path the descriptor already uses, so
propagation cases (NaN in, NaN out) and clamping cases (`tanh(+Inf)` is 1, `relu6(+Inf)` is 6) are
both expressed as normal goldens; the clamping ones are finite and compare under ordinary
tolerance. Serialized arrays carry the C99 `NAN` and `INFINITY` macros. In static initializers
these are observed to keep their bit pattern at `-Ofast` on the toolchains this project gates
(arm-none-eabi-gcc 15 and Apple clang 21), but they are not guaranteed to: clang documents the
macros as undefined behaviour under `-ffinite-math-only` and diagnoses them with
`-Wnan-infinity-disabled`. `tests/test_nonfinite_input_mode.py` compiles the emitted literals and
checks the bit patterns, so the observation is re-established on whatever compiler is in use rather
than assumed. The mode applies to the input tensor only, so weights, alpha and second operands stay
finite and each swept element isolates a single token. Requesting the mode on an op that samples
outside the shared helpers is a generation error rather than a silently finite case.
`cortex-m0` is the only soft-float leg in the target matrix: it runs the f32 suite through
`-mfloat-abi=soft`, so float-to-integer conversion goes through `__aeabi_*` rather than a VFP
instruction, and the two differ on non-finite operands. A descriptor whose contract holds only
there declares `required_capabilities: [soft_float]` and is capability-skipped, with a manifest
entry, on every hard-float target. Generating and building that leg is supported here, but nothing
runs it: no workflow in this repo does, and ns-cmsis-nn's `helia-core-tester.yml` runs `cortex-m0`
under `--suite int` only, with its float legs on `cortex-m4` and `cortex-m55`. The
ns-cmsis-nn#314 guard case therefore has no runner until that consumer workflow adds a
`--cpu cortex-m0 --suite float --float-precision f32` leg.

Spreading operators with a `strict` golden take one token per case. A reduction, a pooling
window, a softmax row and a convolution accumulator all fold many input elements into one output
element, so a case carrying `[nan, inf, -inf]` would put `+Inf` and `-Inf` in the same group and
the golden would then be a statement about `(+Inf) + (-Inf)` rather than about the kernel. Those
descriptors set `nonfinite_tokens` to a single token and there is one case per token, which also
leaves the groups the token does not reach finite and fully asserted -- that is what catches a
vector leg that poisons a whole register instead of one lane. Elementwise and pure-data-movement
operators keep all three tokens in one case. A `mask` case may carry several tokens in one group,
because the group it lands in is don't-care anyway; `mean_float_nonfinite_two_token_*` and
`reduce_sum_float_nonfinite_two_token_*` do exactly that to build the `+Inf` with `-Inf`
reduction group AmbiqAI/ns-cmsis-nn#429 reports hanging at `-Ofast`.

`nonfinite_policy` decides how the golden is compared. It is required whenever `input_mode` is
`nonfinite_sweep` and rejected otherwise; there is no default, because whether an uncontracted
non-finite output may be pinned is a per-kernel question.

- `strict` asserts the reference value on every lane. It is only legitimate where ns-cmsis-nn
  documents the behaviour -- the elementwise family, the standalone hard swish, the
  RELU/RELU6/LEAKY_RELU activations, `arm_reduce_sum_*` and `arm_nn_mean_*` -- or where the
  operator is pure data movement (pad, reshape, transpose, strided slice, concatenation, split),
  since a copy has no freedom to specify.
- `mask` marks as don't-care every lane whose *reference* output is non-finite **and** every lane
  a swept token can reach. The second set is the larger one: a pooling window that sees `+Inf`
  reduces to a finite number at the other lanes of its row, and which of them the token moves is
  the kernel's own fold order, not a contract. Reachability is measured rather than declared --
  the generator re-runs the op's reference with the token positions replaced by finite probes and
  marks every output lane that moves. Every remaining lane still has to match, and the case still
  has to return `ARM_CMSIS_NN_SUCCESS` without faulting or timing out. This is for kernels that
  document nothing (abs, softmax, batch norm, pooling) or that declare the result unspecified
  outright (`arm_minimum_*`/`arm_maximum_*`, `Include/arm_nnfunctions_flt.h:658-703`). It asserts
  robustness and non-corruption of the neighbouring lanes without encoding an uncontracted value
  as a golden. A case that ends up masking every lane fails generation, since it would assert
  nothing beyond `SUCCESS`.

A masked case emits the mask alongside the golden, whose masked entries are written as `0.0f` so
the generated header stays finite-only, and the harness prints `HELIA_MASKED_LANES: k of n`. The
reporting parser records both `k` and `n`, because "passed with one lane masked" and "passed
with every lane but one masked" are different claims.

To scaffold a new tester op, start from `helia_core_tester/scripts/scaffold_operator.py`.

Generated LiteRT-only ops should route through `build_<op>_op()` and resolve tensor roles from
`tensor_dtypes` or the normalized descriptor metadata instead of hand-parsing legacy dtype fields.

### Non-finite float comparison

Float outputs are compared element by element against `atol + rtol * |expected|`, but each
element is classified before that tolerance is computed. A NaN or infinite operand is never
run through the tolerance, because `rtol * |Inf|` is `Inf` and `0 * |Inf|` is `NaN`, and
`diff > tol` is false against either. Matched non-finite operands pass: NaN against NaN, or
two infinities of the same sign. For the families whose ns-cmsis-nn header notes state it
(elementwise add/sub/mul, `arm_nn_activation` RELU/RELU6/LEAKY_RELU, hard_swish, and
mean/reduce_sum), `Include/arm_nnfunctions_flt.h` guarantees the NaN-ness of an element and not
its payload, so a matched NaN passes regardless of sign or payload (see AmbiqAI/ns-cmsis-nn#333).
Minimum and maximum are documented as unspecified for non-finite inputs, so a matched NaN there
is a property of the implementation rather than a guarantee. Every other pairing fails, including
infinities of
opposite sign and a non-finite value against a finite one, and is reported on a
`HELIA_NONFINITE_MISMATCH[i]` line that prints `nan`/`+inf`/`-inf` symbolically; the reporting
parser classifies those results as `nonfinite_mismatch`.

A mismatched non-finite element is also counted on a `HELIA_NONFINITE_MISMATCHES n=<k>` line,
emitted once per tensor when `k > 0`. That line, not the headroom sentinel, is what the parser
classifies on, because the sentinel has a second cause.

Matched non-finite elements are excluded from the headroom measurement rather than voiding it,
so a tensor with a few NaN lanes and finite values elsewhere still reports real `maxdiff` and
`maxfrac` for the finite elements. The `maxdiff=-1.0 maxfrac=-2.0` headroom sentinel is
reported only when a non-finite element mismatched or when no finite element was compared at
all, which includes a zero-length validation: it compares nothing, so it passes and records the
sentinel rather than a headroom number it never measured.

The classification decodes the IEEE-754 exponent and significand fields out of the element's
own storage bytes, at the element's own width (binary16, binary32 or binary64, selected by
`_Generic` on the element type), before the element is converted to `double` for the tolerance
arithmetic. Doing it in that order is what makes it immune to `-ffinite-math-only`, which both
`-Ofast` and `-O3 -ffast-math` imply: under that flag every floating-point instruction is
emitted with `nnan`/`ninf`, so a class test applied to a value produced by a widening
conversion may be deleted as provably false, and `isnan`/`isinf` may fold to a constant false.
An integer test on bytes read from the array asks nothing of the optimizer.

The host driver in `helia_core_tester/tests/c_host/` is built and executed at both `-Ofast` and
`-O3 -ffast-math`, for `float` and `_Float16`, by
`helia_core_tester/tests/test_float_nonfinite_compare.py`. Alongside it,
`helia_core_tester/tests/test_float_nonfinite_fold_harness.py` builds and runs the full generated
harness shape at the same two flag sets: a file-scope `static const` golden carrying
`NAN`/`INFINITY` literals, the kernel output produced in a second translation unit, and the
validation macro expanded in a `*_test_case_run()`. That shape is the one where a
classify-after-widening validator actually loses lanes, so it is what holds the fix in place;
smaller probes stay correct on the same compiler either way. The Arm targets are compiled and
their classification call sites counted, not executed, with the toolchains and results recorded
in the pull request.

## Coverage Merge

```bash
uv run helia_core_tester coverage-merge --cpu cortex-m0,cortex-m4,cortex-m55 --suite both
```

Outputs:
- `artifacts/reports/coverage/merged/coverage_merged.info`
- `artifacts/reports/coverage/merged/coverage_merged_summary.json`
- `artifacts/reports/coverage/merged/coverage_merged_summary.md`
- `artifacts/reports/coverage/merged/index.html`

Behavior:
- for a single suite (`--suite int` or `--suite float`), merge is strict and fails if any requested CPU input is missing.
- for `--suite both`, merge requires at least one suite input per requested CPU and reports suite-specific missing inputs.

## Clean Contract

- `clean`: removes selected CPU artifacts for generated tests, reports (generation/tests/coverage), and matching build dirs.
- `clean-all`: removes all `artifacts/generated_tests`, all `artifacts/reports`, and all `artifacts/build-*` directories.

## Release Process

- Pull request titles should use conventional commit prefixes such as `feat:`, `fix:`, `perf:`, `refactor:`, `chore:`, `docs:`, `test:`, `ci:`, or `build:`
- Pushes to `main` update a release PR through release-please; release-please updates the version files and changelog, but does not create a GitHub Release.
- Merging the release PR creates the `vX.Y.Z` git tag automatically through the tag workflow.
- The release workflow manages `CHANGELOG.md`, `pyproject.toml`, and `helia_core_tester/__init__.py`.
- To force a specific version, add a `Release-As: 1.2.3` footer to the merged commit body.

## Config Precedence

Resolved config order:
- code defaults
- `helia_core_tester.toml`
- environment (`HELIA_CORE_TESTER_*`)
- CLI options

After validation, resolved config is immutable.
