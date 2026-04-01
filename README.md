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

## Canonical Artifacts

Generated tests:
- `artifacts/generated_tests/<cpu>/manifest.json`
- `artifacts/generated_tests/<cpu>/tests.cmake`
- `artifacts/generated_tests/<cpu>/<Family>/<descriptor_name>/...`

Build outputs:
- `artifacts/build-<cpu>-<compiler>/tests/<Family>/<descriptor_name>.elf`

Reports:
- generation: `artifacts/reports/generation/<cpu>/`
- test execution: `artifacts/reports/tests/<cpu>/`
- per-CPU coverage: `artifacts/reports/coverage/<cpu>/`
- merged coverage: `artifacts/reports/coverage/merged/`

Generation report files (always emitted):
- `generation_summary.json`
- `generation_failures.json`
- `conversion_failures.json`
- `manifest_pointer.json`

## Coverage Merge

```bash
uv run helia_core_tester coverage-merge --cpu cortex-m0,cortex-m4,cortex-m55
```

Outputs:
- `artifacts/reports/coverage/merged/coverage_merged.info`
- `artifacts/reports/coverage/merged/coverage_merged_summary.json`
- `artifacts/reports/coverage/merged/coverage_merged_summary.md`
- `artifacts/reports/coverage/merged/index.html`

Behavior:
- merge is strict and fails if any requested CPU `coverage.info` input is missing.

## Clean Contract

- `clean`: removes selected CPU artifacts for generated tests, reports (generation/tests/coverage), and matching build dirs.
- `clean-all`: removes all `artifacts/generated_tests`, all `artifacts/reports`, and all `artifacts/build-*` directories.

## Release Process

- Pull request titles should use conventional commit prefixes such as `feat:`, `fix:`, `perf:`, `refactor:`, `chore:`, `docs:`, `test:`, `ci:`, or `build:`.
- Pushes to `main` update a release PR through release-please; merging that release PR creates the `vX.Y.Z` tag, GitHub Release, and version bumps.
- The release workflow manages `CHANGELOG.md`, `pyproject.toml`, and `helia_core_tester/__init__.py`.
- To force a specific version, add a `Release-As: 1.2.3` footer to the merged commit body.

## Config Precedence

Resolved config order:
- code defaults
- `helia_core_tester.toml`
- environment (`HELIA_CORE_TESTER_*`)
- CLI options

After validation, resolved config is immutable.
