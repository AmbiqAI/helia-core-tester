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

To scaffold a new tester op, start from `helia_core_tester/scripts/scaffold_operator.py`.

Generated LiteRT-only ops should route through `build_<op>_op()` and resolve tensor roles from
`tensor_dtypes` or the normalized descriptor metadata instead of hand-parsing legacy dtype fields.

### Non-finite float comparison

Float outputs are compared element by element against `atol + rtol * |expected|`, but each
element is classified before that tolerance is computed. A NaN or infinite operand is never
run through the tolerance, because `rtol * |Inf|` is `Inf` and `0 * |Inf|` is `NaN`, and
`diff > tol` is false against either. Matched non-finite operands pass: NaN against NaN, or
two infinities of the same sign. ns-cmsis-nn's `Include/arm_nnfunctions_flt.h` guarantees the
NaN-ness of an element and not its payload, so a matched NaN passes regardless of sign or
payload (see AmbiqAI/ns-cmsis-nn#333). Every other pairing fails, including infinities of
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

Exercised on: `arm-none-eabi-gcc` 15.2.rel1 (cortex-m55 hard-float, cortex-m0 soft-float, `-O3`
and `-Ofast`); `clang --target=thumbv8.1m.main-none-unknown-eabihf -mcpu=cortex-m55` (`-O3`,
`-O3 -ffast-math`, `-Ofast`); and the host driver in
`helia_core_tester/tests/c_host/`, run at both `-Ofast` and `-O3 -ffast-math` for `float` and
`_Float16` by `helia_core_tester/tests/test_float_nonfinite_compare.py`.

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
