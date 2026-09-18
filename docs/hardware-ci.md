# Hardware CI: the nightly kernel suite on the bench runners

`helia_core_tester hardware run` is the local-first entry point for running the
generated CMSIS-NN kernel suite on a real board. The **Hardware Nightly**
GitHub Actions workflow (`.github/workflows/hardware-nightly.yml`) runs that
same command once per board, every night, on the lab's self-hosted per-board
runners, and uploads each board's result bundles as an artifact.

It is the second workflow on those benches: heliaPROFILER's Hardware Validation
runs there too, and both share the runner contract below.

## The runner contract

Each self-hosted runner owns exactly one board, is confined to that board's
probe, and advertises three labels:

```text
self-hosted
hpx-hardware
<board>            # e.g. apollo510_evb
```

`runs-on: [self-hosted, hpx-hardware, "<board>"]` therefore routes a board's job
to the runner that owns it, on whichever bench it is attached to. Boards run in
parallel; a runner takes one job at a time, and that exclusivity is the only
serialisation the workflow relies on — there is no concurrency group, because
one keyed by board would throttle several runners of one board type back to a
single job.

### Environment

| Variable | Meaning | Required |
|---|---|---|
| `HPX_BOARD` | the one board this runner owns; must equal the matrix board | yes |
| `HPX_JLINK_SERIAL` | that board's J-Link probe serial | yes |
| `HPX_JLINK_DLL` | the `libjlinkarm.so` pylink should load | recommended |
| `JLINK_PATH` | the `JLinkExe` binary (flash, reset, probe inspection) | recommended |

The first step of every board job checks `HPX_BOARD` and `HPX_JLINK_SERIAL`,
refuses a runner whose board disagrees with the job's matrix entry, and derives
the session id from them. Serials are never workflow inputs: with several probes
on one bench an implicit serial is ambiguous, and a job must not be able to name
another board's probe.

The same two variables are what the CLI itself reads — `--board` defaults to
`$HPX_BOARD` and `--serial-no` to `$HPX_JLINK_SERIAL` — so a bench session by
hand and the nightly resolve identically.

### Packages on the runner's `PATH`

The runner services expose only their declared package set, not the host's
general tools (`lab.embedded.packages` in the lab's NixOS configuration). The
nightly needs:

- `git` (checkout) and `uv` (dependency install; the job then uses
  `uv python install 3.11`)
- `cmake` and `ninja` — NSX configures and builds the benchmark firmware
- `gcc-arm-embedded` — the cross compiler NSX's toolchain file resolves off
  `PATH`, and the `arm-none-eabi-{size,nm,objdump}` tools the memory report and
  the provenance block read the linked image with
- SEGGER J-Link (`JLinkExe` plus the Ambiq device pack) — flash, reset, RTT
- `jq` — the workflow's summary and empty-selection steps read the run's JSON
  document with it. **The guard step fails with a named error when `jq` is
  missing**, rather than letting the job discover it an hour later.

`flatc` is **not** required, and its absence must never fail a job. Generation
reaches for it on one path only: the LSTM operator, to rewrite a converted
`.tflite` through the FlatBuffers schema. Every step of that path is wrapped,
and any failure — `flatc` missing included — falls back to the validated
`ns-cmsis-nn` `UnitTest/TestCases/TestData` reference vectors, which
`hardware run` points generation at through `CMSIS_NN_ROOT` (the synced
`ns-cmsis-nn` module of the firmware's own NSX app). LSTM cases are therefore
generated and streamed either way; what changes is whether their tensors are
freshly converted or the checked-in reference set. The guard step records which
in the job summary and emits a notice when `flatc` is absent.

Adding `flatbuffers` to the runner package set is a small, self-contained change
to the lab configuration and would make the LSTM data path the same on the bench
as on a developer machine that has it. It is a nice-to-have, not a blocker.

### Caches

The runner service's `HOME` is a root-owned, NixOS-managed directory, so the
workflow points uv and the NSX app tree at directories beside the workspace
(`${{ github.workspace }}/../`), which survive `actions/checkout`'s clean:

- `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`
- `HCT_CACHE_DIR` — the job passes `--build-dir "$HCT_CACHE_DIR/hardware/<board>"`,
  so the rendered NSX app, `nsx.lock` and the synced module checkouts (the Ambiq
  SDK among them) are not re-cloned nightly. `hardware build` re-renders and
  rebuilds whenever the manifest, baseline or build options change, so a warm
  tree is an optimisation and never a stale-firmware risk. Deleting that
  directory on the bench is the escape hatch.

Generated tests are *not* cached: `actions/checkout` cleans the workspace, so
every nightly regenerates `artifacts/generated_tests/` from the descriptors
against the same kernel checkout the firmware links. That is the intended
behaviour — there are no committed generated tests in this repo.

## What the nightly runs

```bash
uv run helia_core_tester hardware run \
  --board "$HPX_BOARD" \
  --serial-no "$HPX_JLINK_SERIAL" \
  --suite both \
  --session-id "nightly-<run_id>-<board>" \
  --build-dir "$HCT_CACHE_DIR/hardware/<board>" \
  --json
```

That single command builds the firmware as an NSX app, flashes it unless the
board already confirms it runs this exact build, generates the tests for the
board's CPU from the same kernel checkout the firmware links, streams every
bridged case, writes the result bundle, and exits non-zero on any correctness
mismatch — which fails the board's job.

Before it, the job runs `helia_core_tester doctor` and
`helia_core_tester probes match --board <board>`, so a bench whose toolchain or
probe is wrong fails in seconds rather than after a firmware build.

There is no cheap `--list` preview of the case set the way hpx has one: bridging
reads the generated tests, which the job has not produced yet. Two checks stand
in for it — the `family` input is validated against
`generated_test_bridge.bridged_families()` *before* generate and build, and
`totals.ran == 0` after the run is a job failure, so a green job always means
cases ran.

## Boards

The board matrix comes from `assets/hardware_boards.yaml`. The `plan` job runs
`helia_core_tester/scripts/board_matrix.py` and feeds its JSON array to
`strategy.matrix.board`, so the workflow names no board of its own: adding a row
to the board table is all it takes for the nightly to cover a new board (once a
runner advertises that board's label). A `boards` input that names an id the
table does not carry is an error, not a job queued forever against a label no
runner advertises.

## Dispatching a run by hand

```bash
gh workflow run hardware-nightly.yml --ref <branch> \
  -f boards=apollo510_evb \
  -f suite=int \
  -f family=BasicMathFunctions \
  -f limit=8
```

| Input | Default | Meaning |
|---|---|---|
| `boards` | `""` | Comma-separated board ids; empty means every board in the table. The scheduled run passes no inputs, so an explicit default list here would silently keep a newly added table row out of the nightly. |
| `suite` | `both` | `int`, `float` or `both` — the CLI's `--suite`. `both` runs int and float in one session, one flash, one bundle. |
| `family` | `""` | One operator family (the CLI's `--family` is singular), e.g. `BasicMathFunctions`. Empty streams every bridged family. |
| `limit` | `""` | The CLI's `--limit`: bridge only the first N discovered tests per suite/family. |
| `session_suffix` | `""` | Appended to `nightly-<run_id>-<board>`. Must match `[A-Za-z0-9._-]+`; it becomes the bundle's directory name. |

Narrowing with `family` and `limit` narrows the **streamed** set, not
generation: `hardware run` always generates the whole suite for the board's CPU,
because the generated tests are what the run is evidence about. A smoke dispatch
is therefore quick on the board and not much quicker on the host.

There is no `cmsis_nn_ref` input. The CLI has `--cmsis-nn-root` (a local
checkout, declared to NSX as a `source: {path:}` module) but no ref override:
adding one means plumbing a per-project revision through the app renderer's
`module_registry` block *and* deciding what it does to the bundle's
qualification verdict, which compares the lock's outcome against the baseline's
pins. That is its own change, not a line in a CI workflow. Until then the
nightly always builds the baseline's qualified kernel commit, and a
kernel-candidate A/B is a local `--cmsis-nn-root` run.

## Artifacts

Each board job uploads one artifact:

```text
hardware-kernels-<run_id>-<board>
├── artifacts/reports/hardware/<session_id>/     # the result bundle
│   ├── session_manifest.json
│   ├── session_summary.json
│   ├── memory_report.json
│   ├── case_summary.csv
│   ├── correctness/ outputs/ logs/
│   ├── hct_provenance.json
│   └── nsx.lock
└── artifacts/reports/hardware-nightly-run.json  # the --json run summary
```

The upload runs whether or not the run passed, and overwrites an artifact of the
same name, so "re-run failed jobs" replaces that board's artifact with the new
attempt's bundle. Consumers group a run's artifacts by the GitHub run id, which
is in every session id.

### What the hpx dashboard would need

The dashboard (`AmbiqAI/hpx_dashboard`) does not ingest these bundles today, and
nothing in this repository changes that. Three things on its side would:

1. **Source repository.** `scripts/sync_github_artifacts.py` has
   `DEFAULT_REPOSITORY = "AmbiqAI/helia-profiler"` and downloads one repo's runs.
   It would need `AmbiqAI/helia-core-tester` as a second source.
2. **Artifact prefix.** The same script filters on
   `ARTIFACT_PREFIX = "hardware-validation-"`. These artifacts are
   `hardware-kernels-<run_id>-<board>`.
3. **A kernel engine in the whitelist.** `hpx_dashboard/bundle.py` maps engines
   to CMSIS-NN providers in `_ENGINE_CMSIS_NN_PROVIDERS`
   (`tflm → arm`, `helia-rt → ns`, `helia-aot → ns`) and rejects a case whose
   engine is not there. Kernel benchmark cases need an entry of their own
   (`ns`, since the firmware links `ns-cmsis-nn`).

Beyond those three, the bundle *shape* differs: `load_validation_bundle` reads
hpx's `validation_manifest.json` (one entry per model/engine case), while a
tester bundle is `session_manifest.json` plus `session_summary.json` (one entry
per kernel case). The dependency provenance block is already the shared part —
`session_summary.json`'s `dependencies` is hpx's
`DependencyProvenance.to_dict()` shape, so `dataset.py`'s per-project lookup
reads a tester bundle unchanged. The rest is a reader on the dashboard side, or
an hpx-shaped manifest on this one; that is a follow-up, not part of this
workflow.

## Schedule

05:00 UTC nightly (22:00 PDT / 21:00 PST the previous day). heliaPROFILER's
hardware validation nightly is 09:00 UTC on the same benches, and both ask for
the same physical boards — a runner takes one job at a time, so the two
schedules are deliberately four hours apart rather than queued behind each
other.

## Contract tests

`helia_core_tester/tests/test_hardware_nightly_workflow.py` pins the workflow
against this document and against the CLI: the runner labels, the guard's
environment checks, the artifact name and `overwrite`, the dispatch inputs, and
the bundle path (against `result_bundle.bundle_root_for`, not a string typed
twice). The shell steps are executed under the host's real bash with `uv`
stubbed out, so a quoting or argument-assembly mistake fails a unit test instead
of a 05:00 bench run. `helia_core_tester/tests/test_board_matrix.py` covers the
matrix script, including that it agrees with `load_board_table()` and imports
nothing the hosted plan job would not have.
