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
- `uv run helia_core_tester boards` / `probes list` / `probes match`
- `uv run helia_core_tester hardware run|build|flash|stream|memory-report`
- `uv run helia_core_tester explain <bundle> [--case ID] [--op OP] [--json]`
- `uv run helia_core_tester score <baseline...> --candidate <bundle> --check <check.json>` (`--no-check` instead, for humans scoring harness changes; `candidate eval` runs this for you)
- `uv run helia_core_tester candidate check <tree> --base <sha>`, `candidate baseline`, `candidate eval` (see [Agent loop](#agent-loop))
- `uv run helia_core_tester agent-loop init|selftest|launch|status|stop` (see [Running an agent campaign](#running-an-agent-campaign))

Removed interfaces:
- `gap-check` subcommand
- `--skip-conversion`
- `--skip-runners`
- `--regen-generated-tests-after-cleanup`
- report-dir override flags
- `--include-float` (replaced by `--suite float` or `--suite both`)
- the `perf-stream` command group and `scripts/run_hardware_perf_suite.sh` (replaced by `hardware run`, see below)

## Hardware CLI

The FVP commands above simulate; `hardware` runs the generated kernel tests on a
real Ambiq board over SEGGER RTT (one universal `hct_benchmark_server` firmware,
per-case data streamed from the host). The whole pipeline is one command:

```bash
uv run helia_core_tester hardware run --board apollo510_evb
```

That generates the tests for the board's CPU, builds the firmware, flashes it only
unless the board already confirms (via its TARGET_INFO build id) that it runs this exact
build, streams every bridged case,
writes the result bundle under `artifacts/reports/hardware/<session-id>/`,
and prints the pass/fail summary (`--json` prints one JSON document on stdout
instead, with the human output on stderr). Exit codes match `score`: 0 pass,
1 a case failed correctness, 2 bad flags, 3 refused before running (dirty
tester, a `--skip-flash` or `--golden-from` mismatch, no case matches the
selection), 5 error (cmake, J-Link, transport, probe, or a tester bug), 130 interrupted (Ctrl-C). Useful narrowing flags: `--suite int|float|both`,
`--family`, `--test-name`, `--limit`, `--precision fp16|fp32` (float-only shortcut,
not combinable with `--suite both` or `--test-name`), `--fvp-gate off|advisory|strict`,
`--op`/`--dtype` (repeatable, matched like `generate --op/--dtype`; a case's dtype is its
activation dtype, or `S4` for s4-weight cases),
`--case-id`/`--cases-from <file>` (exact case ids, e.g. a rerun list; not combinable with `--limit`;
with `--suite int` or `float`, `--op`/`--dtype`/`--case-id` also narrow generation; other cases stay in the tree),
`--skip-generate`, `--skip-flash`, `--force-flash`.

Held-out shapes: `generate --random-shapes N --shape-seed S` draws N s8
Convolve and N s8 DepthwiseConv cases (`rs<S>_conv_*`, `rs<S>_dw_*`) instead of
the `assets/` descriptors. `--op`/`--dtype` limit the draw to matching ops
(e.g. `--op Convolve` draws only `rs<S>_conv_*`, the same cases as an
unfiltered draw). Every `--op` token must name a registered op by operator,
descriptor stem or path; case-name prefixes such as `rs7_conv` are refused,
since drawn names (and hidden ids) cannot be filtered. Each op draws
from its own seeded stream, cycling through every wrapper route and sized to fit
the smallest board workspace. The descriptors, the drawn ops (`ops`) and a per-route count land in
`artifacts/random_shapes/s<S>/<cpu>/`; the cases join the generated tree beside
the fixed ones. Run them with
`hardware run --skip-generate --test-name rs<S>_`. A draw with a flat golden is
dropped and counted as `skipped_degenerate`. To add an op, register its route
list and layer sampler in `GENERATORS` (`generation/random_shapes.py`) with a
new stream id, and teach the op-specific helpers (`layer_route`, `footprint`,
`layer_macs`, `_relu6_gain`, `_descriptor`) its layout.

Hidden shapes: `generate --random-shapes N --hidden-dir DIR` draws the same
kind of cases from a secret seed instead (env `HCT_HIDDEN_SEED`, or
`--hidden-seed-file F`; 16+ characters, e.g. `python -c "import secrets; print(secrets.token_hex(16))"`). DIR and
F must sit outside the tester tree. DIR mirrors a tester root: descriptors in
`DIR/artifacts/random_shapes/<cpu>/`, cases in `DIR/artifacts/generated_tests/`,
reports in `DIR/artifacts/reports/`. Case ids are opaque keyed hashes
(`h<12 hex>`), and summaries record only `seed_commitment`, a SHA-256 of the secret.
`hardware run --hidden-set DIR` adds every DIR case for the board's CPU to
the run, whatever the other filters; it needs only DIR, not the secret. Pass the
same DIR to baseline and candidate runs. The bundle marks them with a `hidden`
column (`true`/`false`) in `case_summary.csv` and cases.json, and records
`selection.hidden_set` (`seed_commitment`, `cases`) in session_summary.json. A DIR
with no set for the board's CPU, or with a case that cannot run, is refused.

Correctness: int cases use the per-operator LSB tolerance in
`generation/io/dtypes.py`, and every case records `max_abs_diff` and `diff_count`
(elements differing at all) in `case_summary.csv` and `cases.json`; both are null
when unknown (unvalidated outputs, wrong output size).
`--strict-compare` drops the tolerance so int outputs must match the golden
exactly. `--golden-from <bundle dir>` judges each case against that past run's
`outputs/` instead (bit-exact for int, the usual tolerance for float), which pins
the current kernels' rounding where the TFLite golden differs by design.

`--golden-from` is the compare mode for kernel changes: run the baseline
kernels once, then judge the changed kernels against that bundle. Strict
compare against the generated goldens does not work there, because 40 of 131
s8 conv/depthwise cases differ from the TFLite goldens by 1 LSB on apollo510
(ns-cmsis-nn v7.39.3; issue #220).

Every bundle records a per-case `input_digest` in `cases.json` and
`correctness/<case>.json` (also in each `case_manifest.json`): the sha256 of
everything streamed to the board except the expected output (input, weight,
bias and quant blobs, scalar parameters, kernel id). Before flashing,
`--golden-from` refuses a case that the baseline lacks, that the baseline run
failed (unless `--golden-allow-failed`), whose baseline `input_digest` differs
or is absent (bundles from before this field must be rerun), or whose output is
missing or wrong-sized. The session manifest's `compare` block records which
mode ran, plus the baseline path (`golden_from`) and `golden_session_id`; each
case's `expected_output_sha256` is the digest of the output it was compared
against. The steps are also available individually as
`hardware build`, `hardware flash [--force]`, `hardware stream` and
`hardware memory-report`.

The firmware builds as a neuralspotx (NSX) app rendered into
`build/hardware/<board>/nsx_app`. Which kernels it builds:

- Nested layout (the tester at `ns-cmsis-nn/Tests/helia-core-tester`): the
  enclosing ns-cmsis-nn checkout, working-tree edits included.
- Standalone clone: the pinned ns-cmsis-nn release (`v7.40.0`).
- `--cmsis-nn-ref REF` builds another tag or commit; `--cmsis-nn-root PATH`
  builds another local checkout.

Edit kernels in the checkout. Every build recopies a local checkout's
`Include/`, `Source/` and `cmake/` trees, plus `nsx/nsx-module.yaml` and
`nsx/CMakeLists.txt`, into `nsx_app/modules/nsx-cmsis-nn`, so edits made
there are overwritten. Other files under `nsx/` are not used.

`hardware run` generates the tests from the same kernel tree: the local
checkout, or for a ref the clone NSX syncs into `nsx_app/modules/ns-cmsis-nn`.

`--no-inline-asm` builds requantize without inline assembly (`--inline-asm`
turns it back on); `--update-dependencies` re-resolves the NSX modules into
`nsx.lock`. Each build saves its kernel options in `nsx_app/.hct-options.json`.
`hardware build`, `flash` and `run` reuse the saved kernel source
(`--cmsis-nn-root` or `--cmsis-nn-ref`) when you leave it out. Switches
(`--inline-asm`, `--placement`) do not persist: an unpassed switch builds its
default, so pass the same switches to `build`, `flash` and `run`. An option that
differs from the saved build rebuilds and prints one line naming the change.
`--skip-flash` keeps the flashed build's options and refuses a different flag. A
saved ref you did not pass with `--cmsis-nn-ref` follows the pinned release, so
a pin bump rebuilds those build dirs the same way. Every build, flash, run and
stream prints the kernel source, inline asm setting, placement and toolchain.

`--toolchain atfe` builds with Arm Toolchain for Embedded clang (from
`$ATFE_ROOT/bin`) instead of `gcc`, the default. Like placement it is a switch.
Each toolchain has its own default build dir (`build/hardware/<board>-atfe` for
atfe), and a build dir whose CMake cache holds the other compiler drops that
cache and reconfigures. The bundle records the compiler in
`session_manifest.json` `build.toolchain`.

`--placement` picks where operands live. `tcm` (the default) keeps every operand
in one workspace: DTCM on Apollo510 and Apollo330P, and SRAM (`RWMEM`) on
Apollo3P, whose 64 KiB TCM can't hold it. Neither path touches a D-cache, so it
measures kernel-only cost. `mram` (Apollo510, Apollo330P)
programs each case's weights and bias into a reserved MRAM pool past the image
and evicts them from the D-cache before every timed call, as a large model's
layer sees them; activations, scratch, multipliers and shifts stay in DTCM, as
hpx places a TCM-sized arena. Neither programming nor eviction counts toward
the timed cycles or `prepare_cycles`. Give each placement its own `--build-dir`. The
bundle records it in `session_manifest.json` `target.placement`.

PMU counters are selected with `--pmu-counters GROUP:SELECTION` (repeatable, on
`hardware run` and `hardware stream`; hpx syntax). `GROUP` is `cpu`, `memory` or
`mve`; `SELECTION` is `all`, `default`, or a comma-separated list of `ARM_PMU_*`
names from `assets/pmu/armv8m_pmu_events.json`:

```bash
uv run helia_core_tester hardware run --board apollo510_evb --family ConvolutionFunctions \
  --pmu-counters mve:all --pmu-counters cpu:default
uv run helia_core_tester hardware stream --pmu-counters mve:ARM_PMU_MVE_STALL,ARM_PMU_MVE_PRED
```

The default is every group at its default selection. Each group runs in passes of
up to four chained 32-bit event counters (the Cortex-M55 PMU has eight 16-bit slots),
so `mve:all` costs nine passes per case. One run takes up to 32 passes (the firmware's
`HCT_SERVER_MAX_PASSES`), so the full catalog (`--pmu-counters all`, i.e.
`cpu:all memory:all mve:all`, 18 passes) fits one run and one bundle.
`ARM_PMU_CPU_CYCLES` is always reported from the PMU cycle counter alongside the DWT
cycles. `case_summary.csv` gets one column per counter (median per invocation) plus
`overflow_detected`, `valid_for_regression` and `timing_status`. `valid_for_regression`
is true only when the output matched and `timing_status` is `valid` or
`degenerate_output`; the other statuses are `error_path` (expects any non-success status, such as an argument or no-impl error), `overflow`,
`zero_cycles`, `below_floor` (median under 3x the board's empty-call floor, which
`session_manifest.json` records as `timing_floor`). `degenerate_output` is
informational only: the golden is constant, has at most two values, or is at
least 90 % saturated, which weakens the correctness check, not the cycles.
Goldens under 8 elements are never judged, BOOL goldens are degenerate only when
constant, and a descriptor's `degenerate_golden_reason` marks the shape as intended.
`session_summary.json` records the passes, counters and per-stage/per-case timing.
`--pmu-groups a,b` still works as a deprecated alias for `--pmu-counters a:default
--pmu-counters b:default`.

Identity resolution rules:

- `--board` is the only identity flag. The CPU, NSX board name, SEGGER device name,
  SWD speed, build dir (`build/hardware/<board>`), default session id
  (`<board>-<UTC timestamp>`) and the linker-script SoC and flash/RAM region names
  `hardware memory-report` measures against all come from the row in
  `assets/hardware_boards.yaml` (`helia_core_tester boards` lists it). Default:
  `$HPX_BOARD`, else `apollo510_evb`.
- DWT-only boards (`pmu_tier: dwt`, such as the Cortex-M4 `apollo3p_evb`) report DWT
  cycles only: the counter default becomes cycles only and event counters are refused.
  Cortex-M4 runs the int suite and FP32; FP16 cases are skipped and `--precision fp16`
  is refused.
- The apollo330mP_evb J-Link device `Apollo330P_510L` is an Ambiq-supplied entry,
  not in SEGGER's stock database; install it under `~/.config/SEGGER/JLinkDevices/`
  (NSX flashes through it too).
- Session sizing comes from the target: every RTT session starts with the firmware's
  `TARGET_INFO` (cases and PMU passes per plan, receive-buffer bytes, PMU width) and
  the host batches the bridged cases from it, so a board with different firmware
  limits needs no host change.
- `--serial-no` is optional: the flag wins, then `$HPX_JLINK_SERIAL`, then the
  connected J-Link probes enumerated through pylink. Exactly one connected probe is
  used as-is; zero or several is an error naming what was found.
  `helia_core_tester probes list` shows the probes, `probes match --board B` prints
  the serial the hardware commands would pick.
- The J-Link shared library pylink loads is resolved from `$HPX_JLINK_DLL` (the
  library file), then `$JLINK_PATH` (the `JLinkExe` binary or its directory), then
  the directory of `JLinkExe` on PATH, then pylink's own search (ldconfig,
  `/opt/SEGGER`). These are the same variables the lab runners export for hpx;
  `helia_core_tester doctor` prints which one resolved the library.

### Board matrix

Run one selection on several boards and get one summary:

```bash
uv run python -m helia_core_tester.scripts.board_matrix run \
  --board apollo510_evb:1160003180 --board apollo330mP_evb:1160003409 \
  --suite int --limit 2 --pmu-counters mve:default [--family F] [--precision fp16|fp32] \
  [-- <hardware run args>]
uv run python -m helia_core_tester.scripts.board_matrix summarize <bundle>... --out <dir>
```

Each board gets its own `hardware run` and its usual result bundle. Boards run in
parallel when every `--board` names a distinct probe serial, else one at a time
(`--sequential` forces that). A DWT-only board runs its default counters when
`--pmu-counters` asks for events, and its summary row says so. Both commands write
`board_matrix.json` (schema `hct.hardware.board_matrix` v1: per-board status,
golden passed/failed/rejected, boot status and clock, build id, kernel source,
bundle; per shared case, `median_cycles` and `ARM_PMU_MVE_INST_RETIRED` per board)
and `board_matrix.md` into `artifacts/reports/hardware/matrix-<UTC stamp>` (or
`--out`), and exit 1 unless every board passed (2 for bad input, a
matrix-owned option such as `--build-dir` after `--`, or an unreadable bundle). Concurrent runs from one checkout
stage their cases under `artifacts/stream_cases/<suite>/<board>` and take turns
generating into a shared CPU tree.

`helia_core_tester doctor` reports the hardware toolchain (arm-none-eabi-gcc,
cmake, ninja, the neuralspotx version, the J-Link library) as
informational checks; missing hardware tools do not fail doctor.

`.github/workflows/hardware-nightly.yml` runs `hardware run --suite both` over
the full catalog every night at 05:00 UTC on the lab runners, one job per board
(default `apollo510_evb`, `apollo330mP_evb`, `apollo3p_evb`; `--pmu-counters all`
on PMU boards), and uploads each board's result bundle with the `--json`
document as `hardware-nightly-run.json` (schemas and version policy:
`docs/performance-streaming-design.md`, "Result bundle").
Under the same board lock each job then runs an MRAM-placement leg (MRAM
boards; `hct-mram-<board>-...`) and a kernel leg built from ns-cmsis-nn `main`,
resolved to one SHA at plan time (`hct-main-<board>-...`, session
`nightly-main-<run>-<board>`). `-f cmsis_nn_ref=<branch, tag or full SHA>` swaps the
ref; any ref but `main` uploads as `hct-ref-<board>-...`. The kernel leg does not
gate the run: its failures show only in the summary table.
`gh workflow run hardware-nightly.yml -f boards=apollo510_evb -f limit=2` runs
it by hand.

### PMU feedback

`helia_core_tester explain <bundle>` reads a hardware bundle (or a directory of
bundles) and prints at most four lines per MAC case: cycles/MAC against a ceiling
from `assets/scoring/ceilings.yaml`, where the cycles go (IPC, MVE share, MVE MAC
instructions against the ideal count, stall shares, L1D refills, prepare share), a
diagnosis and up to three ranked hints. `--op` takes conv, depthwise or fc. `--json` emits schema `hct.pmu_explain`
v3: `bundles` (board, cpu, placement per bundle) and one flat `cases` list,
each case naming its `bundle` and its `pct_of_peak` (0-100); `pmu_explain.explain_case` gives the same result for one case row. Rules live
in `helia_core_tester/hardware/pmu_explain.py` (`_rule_*`); each ranks by the share
of cycles it explains. Cycles-only bundles (Cortex-M4) get % of peak only.

The nightly captures the full catalog (`all`, 18 passes). For a tuning loop, request
only what the rules read (4 passes, `pmu_explain.AGENT_PMU_SELECTION`):

```
--pmu-counters cpu:ARM_PMU_INST_RETIRED,ARM_PMU_STALL_FRONTEND,ARM_PMU_STALL_BACKEND
--pmu-counters memory:ARM_PMU_L1D_CACHE_REFILL
--pmu-counters mve:ARM_PMU_MVE_INST_RETIRED,ARM_PMU_MVE_INT_MAC_RETIRED,ARM_PMU_MVE_FP_MAC_RETIRED,ARM_PMU_MVE_PRED,ARM_PMU_MVE_STALL_RESOURCE_MEM,ARM_PMU_MVE_STALL_DEPENDENCY
```

## Agent loop

An optimization agent edits ns-cmsis-nn kernels. It submits each candidate
with one command and gets one JSON verdict back. Everything trusted (this
tester, the baseline bundles, hidden cases and their seed) stays outside
the agent's view.

### Trust boundary

- The agent runs in a sandbox where only its own ns-cmsis-nn worktree is
  writable. It cannot read this tester tree, the baseline dir, the hidden
  set, its seed file, result bundles or logs.
- The evaluator runs `candidate eval` as the human user, outside the
  sandbox, and hands the agent only stdout and the exit code.
  - On the bench host, wrap it in `bench-agent run`. The command runs in
    `$HOME`, so use absolute paths:

    ```bash
    bench-agent run apollo510_evb --reason <sha> -- timeout 30m \
      uv --directory /abs/helia-core-tester run helia_core_tester \
      candidate eval --kernels /abs/agent-tree --baseline /abs/baseline
    ```

  - From elsewhere, use the `hct-run` client from nixos-config (its
    `candidate eval` mode is a follow-up).
- `candidate eval` reads the agent's tree as plain files and never runs git
  in it. It copies `Source/`, `Include/`, `cmake/` and `nsx/` into a fresh
  checkout of the base commit, then checks and builds that copy. Edits made
  after the copy, and anything in the agent's `.git`, do not reach the
  build. The copy walks by directory handle and never follows a symlink:
  symlinks are copied as links, and the check rejects them. FIFOs, sockets
  and devices are skipped. The copy refuses past 4 MiB per file (sparse
  files count at their full size), 64 MiB copied in total, 5000 files and
  dirs, or 64 levels of dirs; the real trees are about 4.4 MiB in 448
  files. Bytes count as they are read, so a file that grows mid-copy is
  caught. `--max-file-bytes`, `--max-total-bytes` and `--max-files` change
  the limits.
- `hardware run` stderr names every case, so it goes to
  `<baseline>/logs/`, never to the agent.
- Hidden case ids never print. Hidden cases still count in the verdict and
  in the family totals.
- The tester must be committed: `candidate baseline` and `candidate eval`
  refuse a dirty or unknown tester (exit 3, stage `tester`) before any
  check or run, with no opt-out.
- Run one eval at a time per baseline dir: each eval rebuilds
  `<baseline>/snapshot`.

### Flow

1. Baseline, once per base commit, by the human. Use a clean checkout at
   that commit:

   ```bash
   uv run helia_core_tester candidate baseline --kernels ~/ns-cmsis-nn \
     --board apollo510_evb --out ~/hct-eval/dw-s8 --repeats 3 \
     --op DepthwiseConv --dtype S8
   ```

   - The first run generates, builds, flashes and streams. The other runs
     stream the same build again, so the scorer gets a noise band.
   - Every eval reuses the baseline's placement and inline asm (defaults
     `tcm` and on), the agent PMU counters (see [PMU feedback](#pmu-feedback);
     none on DWT boards) and `--fvp-gate off`.
   - `--out` receives the bundles, `baseline.json` (base commit and run
     options), `code_graph.json` (a digest and references per kernel
     function and data object, from the built objects), `kernels.git` (the
     base commit, fetched from the clean tree) and `logs/`.
   - Hidden cases (`--hidden-set DIR`, made by `generate --hidden-dir DIR
     --hidden-seed-file F`) need the hidden-set PRs; until they merge, the
     flag fails the first run. Keep DIR and F outside the sandbox.
2. Candidate, as often as needed, one at a time. The agent edits `Source/` and `Include/`
   in its worktree, and the evaluator runs:

   ```bash
   uv run helia_core_tester candidate eval --kernels <agent worktree> \
     --baseline ~/hct-eval/dw-s8
   ```

   It does five things:
   1. Snapshot the agent's trees.
   2. Run `candidate check` against the base commit.
   3. Run `hardware run` with the baseline's options, `--golden-from` its first
      run (bit-exact) and `--skip-generate`.
   4. Run `score` against every baseline repeat.
   5. Print the verdict.

   On apollo330mP, 50 DW s8 cases took about 1 minute per eval and the
   two-repeat baseline about 2 minutes (measured).

The verdict (schema `hct.candidate_eval` v3) has these fields:

- `verdict`, `exit_code`, and `stage`: `tester` (dirty tester),
  `baseline` (unusable baseline dir), `check`, `run`, `objects` (once
  `candidate check` scans built objects), `score`, or `eval` for an
  unexpected error.
- `findings`: the check's findings, on rejection.
- `score`, `families` and `failures`. Prepare cycles move with code layout,
  so `prepare_regression` fires only when they go missing, pass
  `prepare_max_ratio` times the baseline, or grow enough to pay for the
  case's timed gain (`prepare_share_pct` of the cycles saved; see
  `assets/scoring/noise_floors.yaml`). Under case gate scope `touched`,
  untouched cases fail only on missing prepare cycles.
- `cases`: one entry per public case, with cycles, speedup, delta, noise
  band and `touched`: whether code reachable from the case's inner symbol,
  or its timed symbol itself, changed in the built objects.
- `case_gate`: `{scope, reason}`. Code layout moves untouched kernels by
  a few percent, so with scope `touched` only touched cases face the
  case and family regression gates and count in the score and family
  geomeans. Untouched drift shows per family as `untouched_cases` and
  `untouched_geomean` and never fails; correctness gates still cover
  every case. Scope `all` (with a reason) when the baseline predates
  `code_graph.json` or the candidate objects are unreadable.
- `hints`: `pct_of_peak` (0-100), a diagnosis and ranked hints for each public MAC case
  (from `explain`).
- `hidden`: null without hidden cases; otherwise case, touched and failure counts,
  plus the scorer's hidden `subscores` when it reports them. `failed` lists
  each hidden failure as `{kind, symbol, via, touched, count}`: the inner
  kernel, the timed symbol when it differs, and how many cases share the
  entry. It never names a case, shape or cycle count.

| Exit | Verdict | Meaning |
|---|---|---|
| 0 | `pass` | Correct, no regression, score above `--min-score` |
| 1 | `fail` | Output mismatch, regression, lost timing or changed inputs |
| 2 | | Bad flags, such as a non-finite `--min-score` or a file as `--out` |
| 3 | `rejected` | The diff leaves `Source/`/`Include/` or uses a banned construct |
| 3 | `refused` / `not_comparable` | Bad baseline dir, oversized candidate, dirty tester, golden misfit, moved cases, other build |
| 4 | `no_gain` | Correct but not faster |
| 5 | `error` | Build, board, transport or tester error; see `<baseline>/logs/` |

### `--skip-generate`

- `candidate eval` always passes `--skip-generate`. The baseline generated
  the cases into this tester's `artifacts/generated_tests`, and
  `--golden-from` refuses any case whose inputs changed.
- Do not regenerate between the baseline and its evals, for example with a
  plain `hardware run` on the same tester. A baseline case that disappears
  makes eval refuse (`missing_case`, exit 3); rerun `candidate baseline`.

## Running an agent campaign

`agent-loop` wraps the [agent loop](#agent-loop) into one campaign: a
headless Claude agent edits one kernel family in its own ns-cmsis-nn tree,
and three wrapper scripts are its only way to build, disassemble and run
on a board. Every eval runs `candidate eval` once per leg (`tcm`, `mram`)
and appends a row to the campaign ledger.

Needs, on the bench host: `uv`, `git`, `patch`, `bench-agent`, the
`claude` CLI (logged in), the Arm toolchain (`arm-none-eabi-size`,
`arm-none-eabi-objdump`), and a committed tester checkout.

### 1. Write a campaign file

Start from `assets/campaigns/conv-s8.example.yaml` or `dw-s8.example.yaml`:

```yaml
name: conv-s8                  # lowercase, digits, dashes
board: apollo510_evb           # tester board id
bench_id: apollo510_evb        # bench-agent board id (default: board)
legs: [tcm, mram]              # default: both when the board has MRAM
target:
  op: Convolve                 # `hardware run --op`
  dtype: S8                    # `hardware run --dtype`
  case_ids: []                 # optional `--case-id` filters
kernels:
  repo: ~/ns-cmsis-nn          # any checkout that has the ref
  ref: v7.40.0                 # tag, branch or SHA
evals: 12                      # charged board evals
cost_usd: 25                   # claude --max-budget-usd
model: claude-opus-5-5
hidden_shapes: 12              # random target-op shapes; 0 for none
repeats: 3                     # baseline runs per leg
secrets_dir: ~/hct-secrets/conv-s8   # outside the workspace
# start_patch: ~/campaigns/conv-s8-1/ledger/007.diff
# start_notes: |
#   - 1x1 path vectorized; generic path untouched.
```

Optional: `min_score` (passed to `candidate eval`), `lock_timeout_s`
(board lock wait, default 600), `eval_timeout_s` (per leg, default 300),
`submit_deadline_s` (whole submit, default 540, at most 570, under the
agent's 10 minute Bash limit), `retries` (per leg, default 1) and `max_infra_errors` (busy or
failing board results in a row, default 5). The hidden set holds only the
target op and dtype. Hidden shapes exist for `Convolve` and `DepthwiseConv`
S8 only; set `hidden_shapes: 0` for other targets.
`agent-loop validate FILE` checks a file without side effects.

### 2. Initialize the workspace

```bash
uv run helia_core_tester agent-loop init conv-s8.yaml -w ~/campaigns/conv-s8
```

`init` refuses a dirty tester. It then:

1. Adds a detached worktree of this tester's HEAD at `W/tester`, and links
   `artifacts/downloads` to save a toolchain download. Every later command
   runs from that worktree, so the campaign stays on one tester commit
   while you keep working here.
2. Clones `W/base` (clean) and `W/agent` (tagged `base`) as standalone
   one-commit repos with no remote, so git commands in the agent's tree
   cannot reach other repos. `start_patch` (a ledger diff, or any unified
   diff of `Source/`/`Include/`) is applied to `W/agent`. It is parsed
   strictly: every header pair must name a path under `Source/` or
   `Include/`, and any line outside a counted hunk is refused.
3. Writes a fresh secret seed and hidden set in `secrets_dir` (mode 0700,
   seed 0600). The dir must be new or empty. The workspace records that
   it owns the dir, and only that workspace's rerun reuses its seed.
4. Records one `candidate baseline` per leg under `bench-agent run`.
5. Builds the base kernel library once and saves per-object code sizes
   (`W/size-ref.json`).
6. Renders `W/prompt.md` from a template filled with the routes seen in
   the first leg's baseline (timed and inner symbols, median and best
   cycles per MAC) and the ceiling from `assets/scoring/ceilings.yaml`.
   Writes `W/agent-settings.json` (absolute paths) and `W/bin/{submit,check,disasm}`.

Init saves the campaign first, skips any step whose output exists, and
marks the workspace ready at the end; the other commands refuse a
workspace that is not ready. After a failure rerun the same command (a rerun reuses the campaign's
seed). To change the campaign file, start a new workspace. On apollo330mP, a DepthwiseConv S8 campaign (50 public and 24
hidden cases, `repeats: 2`) took about 6 minutes to initialize, and each
`submit` about 3.5 minutes for both legs (measured).

The baselines are tied to the tester commit through the harness digest:
moving `W/tester` to another commit makes every eval `not_comparable`.

### 3. Check the permissions

```bash
uv run helia_core_tester agent-loop selftest -w ~/campaigns/conv-s8
```

A cheap model (`--model haiku`, no saved session, $1 cap) tries a fixed
list of tool calls. It should be allowed to read the agent tree, write
`Source/`, and run `bin/disasm`. It should be denied ledger, campaign,
baseline, tester and secrets reads, writes outside `Source/`/`Include/`,
and `cat`, `touch`, `curl` and git outside the agent tree in the shell.
Exit 0 when every call matches.

### 4. Launch, watch, stop

```bash
uv run helia_core_tester agent-loop launch -w ~/campaigns/conv-s8
uv run helia_core_tester agent-loop status -w ~/campaigns/conv-s8 [--tail 20] [--json]
uv run helia_core_tester agent-loop stop -w ~/campaigns/conv-s8
```

- `launch` starts `claude -p` in `W/agent` in its own process session with
  `--permission-mode dontAsk`, only the Read, Edit, Write, Glob, Grep and
  Bash tools, `--setting-sources project`, `--strict-mcp-config`,
  `--max-budget-usd cost_usd` and a fixed `--session-id`. Each run writes
  its own stream log under `W/logs/`; `W/agent-run.json` holds the real
  claude pid, the session id and the logs.
- `launch --resume` continues that session with the same flags and caps
  it at `cost_usd` minus the cost of the finished runs. A run killed
  before its `result` event counts as $0, so set the cap with margin.
- `status` prints the ledger, whether the pid is alive, recent tool calls
  from the newest log, its cost and turns once it has a `result` event,
  and the cost of all finished runs.
- `stop` sends SIGTERM to the agent's process group.

### What the agent sees

- `W/bin/check`: copies the agent's `Source/`, `Include/`, `cmake/` and
  `nsx/` into a fresh clone of the base, runs `candidate check`, builds
  the kernels with `hardware build` and reports code size against the
  base. No board, no eval.
- `W/bin/disasm FN`: one function from the last check build, up to 600
  lines.
- `W/bin/submit`: stages a copy of the agent's trees in `W/submit/tree`
  and runs check on it. A tree that fails, or a check that runs past the
  deadline, is rejected and costs no eval.
  Every leg then judges that frozen copy, so edits made during a submit
  wait for the next one. Each leg runs under `bench-agent run --timeout`
  and `timeout`, both cut to fit `submit_deadline_s`. A later leg runs
  only when the earlier legs reached stage `score`. The overall verdict is
  the worst leg, and `pass` needs every leg.
  - A submit that cannot take the submit lock in time (another submit is
    running) returns a free `error` and a ledger row without an id.
  - No verdict at all (board busy past the lock wait, bench-agent or
    tester failure, `refused` at stage `tester` or `baseline`) is retried,
    then recorded with `infra: true`. It costs no eval, and the view hides
    any earlier leg's scores. After `max_infra_errors` such submits in a
    row, submit tells the agent to stop.
  - Verdict `error` from `candidate eval` (often a kernel fault) is
    retried once, then charged, even when the retry finds the board busy. A leg killed by `timeout` (a hang) is
    charged without a retry.
  - The printed view keeps touched case rows only, a count and speedup
    range for untouched cases, and hints for the ten slowest touched cases
    (first leg only). The same text goes to `W/agent-results/NNN.json`,
    since Claude Code truncates long Bash output.
  - Exit codes follow `candidate eval`; 6 means the budget is spent.

### After the run

`W/ledger/` holds, per eval `NNN`: the diff of the judged copy (`NNN.diff`),
each leg attempt's full verdict (`NNN.<leg>.<attempt>.json`, may name hidden cases' kernels
but never their shapes) and stderr. `ledger.jsonl` has one row per submit
with verdict, charged and infra flags, per-family geomeans and code size
delta. Ids come from a counter under the submit lock. To continue from the
best eval, start a new campaign with `start_patch: W/ledger/NNN.diff` and
a few lines of `start_notes`.

### Containment limits

- The permission rules are not an OS sandbox. The agent runs as you, and
  `dontAsk` denies only what Claude Code recognizes. Read-only shell
  commands inside the agent tree (cwd) are auto-allowed, and so is git in
  that tree; the standalone clone keeps them from reaching other repos.
- Deny rules cover the workspace's trusted entries, `secrets_dir`, the
  source ns-cmsis-nn and tester checkouts, and `~/.claude`. Paths outside
  these are not listed. Do not run a campaign on a host with secrets you
  would not show the agent.
- The trusted side runs as you too. `candidate eval` copies the agent's
  trees into a fresh checkout and never runs git in them, but a compiled
  kernel still runs on the board.
- `bench-agent` serializes board use by flock; it does not reserve a board
  for the whole campaign.

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
- `Dequantize` with `entry: arm_dequantize_f16_bits_f32` widens binary16 bit patterns and checks
  every output bit against the NaN rule its build compiles. Tag the input `FP16` to run it on the f16
  legs or `U16` (binary16 storage, accepted for this entry only) to run it on the f32 legs. A case
  holds 1 to 65536 halves: the special classes, random patterns, then three NaNs so the vector and
  scalar tails meet the NaN rule. Keys its path would not use are refused.

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
`reduce_sum_float_nonfinite_two_token_*` do exactly that to build a `+Inf` with `-Inf`
reduction group, which is adjacent to AmbiqAI/ns-cmsis-nn#429 but not its input. #429 puts one
token per group, so the `*_nonfinite_issue429_flatten_*` and `*_nonfinite_issue429_generic_*`
cases carry that placement instead: `[inf, nan]` at flat positions 1 and 3 of four
three-element groups on the innermost axis, and `[nan, inf]` at flat positions 0 and 7 of a
`[1, 2, 2, 2]` input reduced over H, which is the non-innermost axis #429's generic case uses.

A recurrent operator takes one token per case at the first time step of the first batch row.
The recurrence carries whatever that produces into every later step of that row, so the token
reaches the whole row; the descriptors therefore use `batch_size: 2` (SVDF: `input_batches: 2`)
so the other row stays finite and fully asserted, which is what catches a vector leg that
poisons a whole register rather than one lane. Under `mask` (LSTM, SVDF) reachability masks
the token's row; under `strict` (GRU) the row is asserted, NaN lanes by class and the rest by
tolerance. The LSTM and GRU gates swallow an infinity by saturating -- `sigmoid` of an infinity
is 1 or 0 and `tanh` of one is ±1 -- so those cases have a finite reference; the LSTM ones are
masked purely by measured reachability, not by the finiteness of the golden. SVDF's default
±1e30 activation clamps do the same to the
value, and its `time_batches` exceeds its `sequence_steps` so the step-0 column is still in the
state ring when the last step produces the output.

`nonfinite_positions` is what places them. It defaults to the leading run `0..k-1` and pairs
element for element with `nonfinite_tokens`; a descriptor sets it where that run cannot express
the placement, either one token per reduction group or a pooling window that SAME padding only
partly covers (`avg_pool_float_nonfinite_nan_same_odd_f32` and its max-pool twin put the token
at flat index 72 of a `[1, 5, 5, 3]` input, the one real element of the bottom-right window).

`nonfinite_policy` decides how the golden is compared. It is required whenever `input_mode` is
`nonfinite_sweep` and rejected otherwise; there is no default, because whether an uncontracted
non-finite output may be pinned is a per-kernel question.

- `strict` asserts the reference value on every lane. It is only legitimate where ns-cmsis-nn
  documents the behaviour -- the elementwise family, the standalone hard swish, the
  RELU/RELU6/LEAKY_RELU activations, `arm_reduce_sum_*` and `arm_nn_mean_*`, and
  `arm_gru_unidirectional_f32`/`_f16`, whose public declarations carry a NaN contract for a
  token in the input, the previous state or the candidate gate's weight or bias, and state that
  Inf follows the arithmetic (a token confined to the update or reset gate's weight or bias is
  absorbed by the kernel while the reference still produces NaN, so such a case stays `mask`)
  -- or where the operator is pure data movement (pad, reshape, transpose, strided slice,
  concatenation, split),
  since a copy has no freedom to specify.
- `mask` marks as don't-care every lane whose *reference* output is non-finite **and** every lane
  a swept token can reach. The second set is the larger one: a pooling window that sees `+Inf`
  reduces to a finite number at the other lanes of its row, and which of them the token moves is
  the kernel's own fold order, not a contract. Reachability is measured rather than declared --
  the generator re-runs the op's reference with the token positions replaced by finite probes and
  marks every output lane that moves. Every remaining lane still has to match, and the case still
  has to return `ARM_CMSIS_NN_SUCCESS` without faulting or timing out. This is for kernels whose
  doxygen block says nothing about non-finite input (the `arm_softmax_f32` block specifies
  arguments and return status only, and abs, batch norm, `arm_lstm_unidirectional_f32`/`_f16`
  and `arm_svdf_f32` are the same), that declare the
  result unspecified outright (the `arm_minimum_f32` and `arm_maximum_f32` blocks: "The result
  ... for any non-finite input, is unspecified"), or that document legs which disagree: the
  `arm_max_pool_f16` note calls the scalar leg's NaN behaviour unspecified at the shipped
  `-Ofast` and declines to promise NaN propagation end to end, the `arm_avg_pool_f16` note
  has NaN propagating at every optimization level on non-MVE while the MVE clamp resolves it to
  a bound, and the `arm_svdf_f16` note has NaN propagating through the input-activation clamp on
  every build while the MVE output-activation clamp resolves it to a bound, or that are
  simply undocumented (the `arm_elementwise_squared_difference_f16` block of
  ns-cmsis-nn#490 says nothing about non-finite input). It asserts robustness and non-corruption of the neighbouring lanes without encoding an
  uncontracted value as a golden. Two generation-time guards keep the measurement honest: a case
  that ends up masking every lane fails, since it would assert nothing beyond `SUCCESS`, and so
  does a case where no lane moves between the probes, since a two-sided activation clamp that
  saturates all three probes to the same bound is not evidence that the token is confined.

A masked case emits the mask alongside the golden, whose masked entries are written as `0.0f` so
the golden stays finite -- the input arrays still carry the tokens -- and the harness prints
`HELIA_MASKED_LANES: k of n`. The reporting parser records both `k` and `n`, because "passed with
one lane masked" and "passed with every lane but one masked" are different claims; a capture
reporting `k > n` cannot have come from the harness, so it is recorded as a failed case with a
corrupted-capture reason rather than raising.

Hardware streaming carries the exact generated mask as host-only comparison metadata;
masked lanes are skipped before float classification and finite tolerance checks. Rebridge
previously saved masked case bundles from their generated test sources before replaying
them: older manifests lack the bitmap, and zeroed goldens cannot reconstruct it. Existing
strict bundles need no migration. Matching NaNs and same-sign infinities pass strict float
comparison; other non-finite pairings fail. This does not impose NaN-payload or signed-zero
bit equality.

`nonfinite_policy` is required by `OperationBase.nonfinite_policy()`, not by the schema: the
`if`/`then` gate in `schema.json` is documentation until the descriptor loader validates the whole
schema (#100).

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

## Operand sign span

Int cases for the operators wired to the rule (Abs, Add, Sub, Mul, SquaredDifference,
Minimum, Maximum, PReLU) must feed each operand data that spans negative, near-zero and
positive values **after** the input offset is applied, i.e. `value - zero_point`. A one-signed
operand cannot discriminate the sign-dependent kernel paths: the packed DSP loop of
ns-cmsis-nn#343 dropped the sign of `value + input_offset`, and abs, PReLU and min/max branch
on it directly. Uniform `[-1, 1]` float data plus a TFLite zero point does not guarantee the
span, so generation enforces it (`OperationBase._enforce_int_operand_sign_span`, issue #81
property 2).

"Near-zero" is absolute, not a fraction of the operand's own range: the operand must contain a
post-offset value within one count of zero. A relative rule let a large-magnitude s16 operand
whose closest approach was thousands of counts count as covered.

When a runtime input operand the generator owns does not span, generation steers it: one
post-offset value at half the operand's own magnitude is planted per **missing** region, into
the elements whose post-offset magnitude is smallest. Least-extreme rather than leading,
because a full-scale element carries saturation coverage a mid-range one does not, and on a
short operand the leading elements are the whole case. Missing-only rather than all three,
because most operands lack only the near-zero boundary and replacing the negative and positive
elements as well would discard data the case was written around. The span is re-checked after
planting; in the rare case where a planted element was the sole carrier of a region that was
present, the full negative / zero / positive triple is planted instead. Steering is
deterministic and independent of the RNG stream, and the golden is computed after it, so a
re-run reproduces the same data and the same expected output.

Operands with fewer than three elements cannot hold all three regions and are out of scope
entirely -- not steered, not refused, not requiring a waiver. That covers broadcast scalars and
one- and two-element rows; the operand they broadcast against still has to span. PReLUScalar's
cases are all one- or two-pixel and sit below this floor, which is why that operator is not
wired to the rule.

Two kinds of operand are check-only: the generator never steers them, so a failing one must be
waived. An operand baked into the TFLite model (a PReLU alpha) cannot move, because the
reference interpreter would keep using the model's copy and the golden would stop matching the
emitted array. An operand the descriptor pins explicitly (`hint.extras.input_values`, or
`input_1_values` / `input_2_values` for the float squared difference) must not
move, because the pinned values are the case.

An operand that is intentionally one-signed opts out in its descriptor under
`operand_sign_span_exempt`, naming the operand and the reason:

```yaml
operand_sign_span_exempt:
  input: pinned uniformly negative input to hold the alpha branch on every lane (hct#81)
  alpha: PReLU's alpha is the positive slope constant baked into the TFLite model; steering it
    would leave the reference interpreter using the model's copy, and the kernel branches on
    the sign of the input, not of alpha (hct#81)
```

The reason is required, and the key must name an operand the operator actually submits to the
rule: each wired operator declares those labels as `SIGN_SPAN_OPERANDS`, and a waiver on
anything else fails generation instead of silently waiving nothing. `operand_sign_span_exempt`
is also declared in `helia_core_tester/generation/descriptors/schema.json`, but that schema is
not enforced at load time (#100), so the rule lives in code.

## Mutation scoring

`python -m helia_core_tester.mutation run --cmsis-nn-root <checkout>` generates cases, applies
each catalogued mutant to a copy of the kernel source, rebuilds, and reports which cases kill
which mutant (issue #76; catalog in `helia_core_tester/mutation/catalog.py`).

`--cpu` defaults to `cortex-m55`, the widest capability set the int corpus uses, because some
killers are capability-gated descriptors: generating for a narrower CPU removes them from the
corpus. The corpus CPU's capabilities are passed to the scorer, and a mutant that declares
`requires_capabilities` the corpus does not have is reported `NOT_APPLICABLE` instead of
`SURVIVED`. That distinction matters: `SURVIVED` is a claim about the suite ("no case detects
this bug class"), while `NOT_APPLICABLE` says the run never sampled the question.
`--fail-on-survivor` fires only on a real survivor. `requantize_tail_drop` is the current
example -- its only killers are the MVE-gated chunked-equivalence requantize cases, so
`--cpu cortex-m4` reports it not applicable.

The host kernel library is the DSP build (`ARM_MATH_DSP`, no `ARM_MATH_MVEI`), so a cortex-m55
case's `*_get_buffer_size_mve` call is compiled as the plain sizer, which answers for the build it
runs in. Otherwise the DSP kernel gets MVE-sized scratch, and the 1xN convolve cases (whose MVE
route needs less) overrun it. Cases that assert an MVE-only guard, such as the
`null_weight_sum_ctx` faults, still fail the host baseline (the DSP kernel never reads that
context and returns success) and are excluded from scoring; the board and FVP runs cover them.

With `--cases-root`, the corpus CPU is read from the tree on disk (its `manifest.json`, or an
`artifacts/generated_tests/<suite>/<cpu>/` path) rather than from `--cpu`, so a `--cpu` that
does not match the cases cannot excuse a mutant whose killers are in that tree. A `--cpu` that
contradicts the tree is an error, and a tree that records no CPU requires `--cpu` explicitly.

## Pipeline efficiency

Defaults chosen so a repeat run is cheap and a hung kernel cannot wedge a leg.

Generation reuse:
- each generated case carries a `.stamp` over its descriptor document, the case name, target CPU, suite, seed, the identity of the ns-cmsis-nn checkout (commit when the checkout is a clean git tree, a content digest of its `Include/` and UnitTest TestData otherwise), and a generator-version hash (the generation sources, `core/cpu_targets.py`, `core/path_layout.py`, the templates under `assets/templates`, a SHA-256 of `uv.lock` for the resolved dependency set, and the Python version and machine architecture). Float precision is not a stamp input: it selects which descriptors a run generates, not what any one of them emits.
- a case whose stamp still matches is reused: no TFLite conversion, no inference, no file emission. Its manifest entry is rebuilt from the on-disk sidecar, so build and run see the same tree either way.
- a case whose stamp does not match has its directory removed before regeneration, so output a previous descriptor emitted under a different file name cannot survive into the new build.
- capability and kernel-symbol skips are re-evaluated every run, because a different ns-cmsis-nn checkout can add or remove a symbol.
- `generation_summary.json` and `manifest.json` record generated, reused and pruned counts.
- `--force-generate` (also `force_generate` in `helia_core_tester.toml`, `HELIA_CORE_TESTER_FORCE_GENERATE`) regenerates everything.
- cases outside the active filter are pruned from the tree at the end of a run.

Parallel FVP runs:
- `--run-jobs` defaults to `min(host cores, 4)`. FVP boot dominates per-case time, so parallelism is the lever that matters, but an unbounded default on a shared or metered runner is a cost risk.
- `--run-jobs 0` is the explicit opt-in for every host core; `HELIA_CORE_TESTER_RUN_JOBS` overrides either way.

Per-case timeout:
- `--timeout` defaults to 300 seconds per case. A timed-out case is reported as a `TIMEOUT` case and rendered as a JUnit failure with its own message, rather than blocking the run until the CI job cap with no per-case result.
- `--timeout 0` disables it: the value is always forwarded to the run step, so `0` really does hand the run back to the CI job cap as the only backstop. The masked non-finite policy's "returns SUCCESS and does not time out" only holds with a timeout in force.

Compiler cache (opt-in):
- when `ccache` or `sccache` is on `PATH`, the CMake configure adds `CMAKE_C_COMPILER_LAUNCHER` and, at verbosity 1 or higher, logs which launcher it picked.
- `HELIA_CORE_TESTER_COMPILER_LAUNCHER` names a specific tool; a name that is not on `PATH` fails the configure rather than building uncached. Set it to `none` (or empty) to build without a launcher even where one is installed, which is what a reproducibility build wants.
- no image ships either tool; a host without one builds exactly as before.
- With `ENABLE_COVERAGE=ON` (including `--coverage`), the instrumented `cmsis-nn` target bypasses `CMAKE_C_COMPILER_LAUNCHER`: cached objects can contain another build's profile-output paths. Uninstrumented harness targets and non-coverage builds retain their launcher. This does not bypass caches hidden inside compiler wrappers or custom compile rules.

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
- for `--suite both`, merge requires both int and float inputs for every requested CPU; missing pairs are named in the failure output and reports.
- `--include-mve-float` adds optional cortex-m55 float-MVE coverage; it cannot replace a missing required int/float input. Reports are still written when required inputs are missing.
- `--include-mve-int` adds optional cortex-m55 integer-MVE coverage from a `--coverage --coverage-mve-int` run (`artifacts/reports/coverage/int-mve`), under the same rules. The default coverage build defines `ARM_MATH_AUTOVECTORIZE`, which compiles out integer MVE paths guarded by `!ARM_MATH_AUTOVECTORIZE`; `--coverage-mve-int` builds integer sources without it, except `arm_nn_mat_mul_core_4x_s8.c`. Float sources get the same define unless the run adds `--coverage-mve-float`; such a build sets `HELIA_CMSIS_NN_FLOAT_AUTOVECTORIZE` for the harness, and a float case marked `autovectorize_declines` then expects `ARM_CMSIS_NN_NO_IMPL_ERROR`, as it does on a core without MVE float.

## Clean Contract

- `clean`: removes selected CPU artifacts for generated tests, reports (generation/tests/coverage), and matching build dirs.
- `clean-all`: removes all `artifacts/generated_tests`, all `artifacts/reports`, and all `artifacts/build-*` directories.

## Release Process

- Pull request titles should use conventional commit prefixes such as `feat:`, `fix:`, `perf:`, `refactor:`, `chore:`, `docs:`, `test:`, `ci:`, or `build:`
- Pushes to `main` update a release PR through release-please, which updates the version files and changelog.
- Merging the release PR makes release-please create the `vX.Y.Z` tag and GitHub Release on the merge commit, with that version's changelog section as notes, and mark the release PR `autorelease: tagged`.
- Merging any other PR creates no tag or release; no other workflow creates them (see `helia_core_tester/tests/test_release_workflow.py`).
- The release workflow manages `CHANGELOG.md`, `pyproject.toml`, `helia_core_tester/__init__.py` and the package version in `uv.lock`.
- To force a specific version, add a `Release-As: 1.2.3` footer to the merged commit body.

## Config Precedence

Resolved config order:
- code defaults
- `helia_core_tester.toml`
- environment (`HELIA_CORE_TESTER_*`)
- CLI options

After validation, resolved config is immutable.
