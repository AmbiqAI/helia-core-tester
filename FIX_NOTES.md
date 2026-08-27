# FIX_NOTES — issue #66: filtered runs don't prune stale generated artifacts

<https://github.com/AmbiqAI/helia-core-tester/issues/66>

## Symptom

A filtered run reports more cases than the filter admits:

```
uv run helia_core_tester full --suite float --float-precision f16
# reported: 34 total cases   expected: 17
```

The extra 17 are `f32` cases from an earlier run in the same working tree. They
also surface as spurious `BUILD_FAILED` ("ELF file not found in build
directory"), and during PR #63 verification a stale case produced a misleading
`implicit declaration of arm_nn_activation_f16` build error. `--name` filtering
has the same failure.

## Reproduction

Environment: `.devcontainer/` (or `uv sync --frozen`), Python 3.11.

Unit-level, hardware-free — `helia_core_tester/tests/test_filtered_run_artifact_pruning.py`:

1. Call `generation.test_ops.test_generation()` with `suite=float,
   float_precision=f32` into a tmp `generated_tests/float/cortex-m55/` →
   one `nn_activation_float_tanh_f32/` case dir.
2. Call it again with `float_precision=f16` into the **same** dir.
3. Before the fix: `manifest.json` lists only the f16 case, but the
   `nn_activation_float_tanh_f32/` dir (with its `.tflite` + `.c`) is still on
   disk. The descriptor-aware reporting pass counts it → inflated total.

The ELF half is reproduced by placing a stale `*.elf` under
`build-*/tests/` that is absent from `manifest.json` and asserting discovery
picks it up.

End-to-end (`full` on real FVP) was not run here — no Arm toolchain / FVP / the
`ns-cmsis-nn` source tree in this environment. The mechanism is covered at the
unit level for each of the three spots below.

## Root cause

Nothing keys the **artifact tree** to the active filter. `manifest.json` /
`tests.cmake` are rewritten fresh each run, but three spots trust "whatever is
on disk":

1. **Generation** — `generation/test_ops.py::test_generation` does
   `top_generated.mkdir(parents=True, exist_ok=True)` plus per-case
   `mkdir(exist_ok=True)` and never deletes case dirs a prior, wider filter
   wrote. It relies on `generation/conftest.py::pytest_configure` to `rmtree`
   the dir — which does **not** run for `--skip-generation`, a direct
   `helia_core_tester build`, or a programmatic call.
2. **Build tree** — a filtered rebuild in a tree a wider run already built
   leaves that run's binaries in `build-<suite>-<cpu>-*/tests/`; CMake never
   deletes the output of a target it no longer defines. This is what makes the
   reporting `full` run count 34.
3. **ELF discovery / reporting** — `fvp/cmake.py::find_elves` globs *all*
   `build-*/tests/**/*.elf`, and `fvp/reporting.py::run_tests_with_reporting`
   rebuilds `active_descriptors` from every descriptor whose generated dir *or*
   build-tree `.elf` still exists, with
   `TestReport.total_tests = len(descriptor_results)`.

## Fixes (prune to the active filter — not filter-fingerprinted paths)

Prune keeps the canonical `artifacts/generated_tests/<suite>/<cpu>/...` layout
documented in `README.md`; a fingerprinted path would ripple through
`core/path_layout.py`, that contract, and `coverage-merge`.

| File | Change |
|---|---|
| `helia_core_tester/generation/test_ops.py` | New `_prune_stale_test_dirs()`, called at the top of `test_generation` after `top_generated` is resolved: removes case dirs (identified by their `descriptor.yaml`) not in the post-`--limit` `filtered_descriptors` set, then drops emptied family dirs. Self-contained — no longer depends on the conftest side effect. |
| `helia_core_tester/fvp/cmake.py` | New `manifest_test_names(generated_tests_dir)` → the set of `manifest.json` test names, or `None` when there is no manifest (preserves old behaviour). New `_prune_stale_test_elves(build_dir, allowed, verbosity)`, called from `cmake_configure`: deletes `build_dir/tests/**/*.elf` whose stem is not in that set. `find_elves(build_dir, allowed_names=None)` gains the optional allow-set so a stale binary is never *run* even on the `--no-build` path. |
| `helia_core_tester/fvp/orchestrator.py`, `helia_core_tester/fvp/reporting.py` | Pass `manifest_test_names(cpu_generated_tests_dir)` into `find_elves`. |

`shutil.rmtree` / `unlink` are left to raise (no `ignore_errors`): a failed
prune must surface, not leave stale state a later step consumes — matching
`core/steps/clean.py`'s existing reasoning.

## Verification

```
uv run pytest -q helia_core_tester/tests/test_filtered_run_artifact_pruning.py   # 6 passed
uv run pytest -q helia_core_tester/tests                                         # 416 passed, 12 failed
```

The 12 failures are **pre-existing** at HEAD `080db5f` (the PR #63 merge) —
unrelated descriptor/template/reference drift, byte-identical to the baseline
before this change. The fix adds 6 passing tests and no regressions.

`black` / `flake8` / `mypy` are dev deps but have no config and no CI wiring,
and the tree is not `black`-clean; the change matches surrounding style rather
than reformatting.

## Not covered / follow-ups

- `full --skip-generation` still reuses whatever `generated_tests/` and
  `build-*/` hold — by design; `find_elves`' allow-set keeps a stale binary
  from *running*, but the reporting `active_descriptors` scan can still count a
  stale generated dir. Document `--skip-generation` as "assumes a matching
  prior generate", or extend the prune to the build step.
- `full` never invokes the existing `clean` subcommand; unchanged here.
