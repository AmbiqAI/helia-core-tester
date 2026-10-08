"""Orchestration behind `helia_core_tester hardware run` / `hardware stream`.

Everything here calls the Python entry points directly (the same GenerateStep the
top-level `generate` command uses, then the firmware build/flash helpers and the
RTT session runner) -- never a subprocess into the tester's own CLI.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from ..core.cpu_targets import get_cpu_profile, normalize_cpu
from .boards import BoardSpec, default_session_id
from .firmware_build import (
    FlashDecision,
    build_id_path,
    built_kernels,
    flash_firmware,
    nsx_app_dir,
    read_build_id,
    resolve_build_dir,
    stage_kernels,
)
from .errors import RunRefused
from .measurement import (
    TooManyPassesError,
    UnsupportedCounterError,
    check_outbox_fits,
    check_pass_count,
    counter_passes_for_selection,
    resolve_counter_selection,
)
from .pmu_catalog import CPU_CYCLES_NAME, GROUPS, default_selection
from .result_bundle import write_timing
from .run_summary import make_live_progress_printer

if TYPE_CHECKING:
    from .generated_test_bridge import CaseSelection
    from .nsx_app import AppOptions

PRECISION_SUFFIX = {"fp16": "_f16", "fp32": "_f32"}
# `--precision` value -> Config.float_precision value for the generate step.
PRECISION_FLOAT_PRECISION = {"fp16": "f16", "fp32": "f32"}


def apply_precision(precision: Optional[str], suite: str, test_name: Optional[str]) -> tuple[str, Optional[str]]:
    """Expand `--precision fp16|fp32` into (suite, test_name).

    It forces the float suite and narrows the test-name substring filter to the
    `_f16`/`_f32` suffix. Refuses `--suite both` (the shortcut selects float cases
    only) and an explicit `--test-name` (both are a single substring match, so
    combining them would silently narrow to whichever cases contain both).
    """
    if precision is None:
        return suite, test_name
    if str(suite).strip().lower() == "both":
        raise ValueError("--precision cannot be combined with --suite both (it selects float cases only).")
    suffix = PRECISION_SUFFIX.get(precision.lower())
    if suffix is None:
        raise ValueError(f"--precision must be 'fp16' or 'fp32' (got '{precision}').")
    if test_name:
        raise ValueError("--precision and --test-name cannot be combined (both filter via a single substring match).")
    return "float", suffix


def float_precision_for(precision: Optional[str]) -> Optional[str]:
    """Config.float_precision the generate step must use for `--precision`, or None to
    leave it to the TOML/env/default. Without this the shortcut only narrowed the
    *discovery* filter, so e.g. HELIA_CORE_TESTER_FLOAT_PRECISION=f32 with
    `--precision fp16` generated no _f16 cases and then found nothing to run."""
    if precision is None:
        return None
    return PRECISION_FLOAT_PRECISION[precision.lower()]


def validate_fvp_gate(fvp_gate: Optional[str]) -> None:
    if fvp_gate is None:
        return
    from .fvp_gate import GATE_POLICIES

    if fvp_gate not in GATE_POLICIES:
        raise ValueError(f"--fvp-gate must be one of {', '.join(GATE_POLICIES)} (got {fvp_gate!r})")


def parse_pmu_groups(pmu_groups: str) -> tuple[str, ...]:
    return tuple(g.strip() for g in pmu_groups.split(",") if g.strip())


PmuSelection = Dict[str, Union[str, List[str]]]


def parse_pmu_counters(values: Sequence[str]) -> PmuSelection:
    """Parse repeated `--pmu-counters GROUP:SELECTION` values (hpx syntax).

    SELECTION is `all`, `default`, or a comma-separated list of catalog counter names
    (`mve:all`, `cpu:default`, `mve:ARM_PMU_MVE_STALL,ARM_PMU_MVE_PRED`). A bare `all`
    selects every group at `all` (the full catalog, one run). Groups keep
    their command-line order, which is the order the PMU passes run in. Unknown groups,
    counter names and empty name lists (`mve:,`) are rejected here, naming the valid
    choices, and so is a selection that plans more passes than the firmware runs per
    SESSION_PLAN (measurement.MAX_PASSES_PER_PLAN) -- all before any probe I/O.
    """
    selection: PmuSelection = {}
    if any(raw.strip().lower() == "all" for raw in values):
        if len(values) > 1:
            raise ValueError("--pmu-counters: bare 'all' already selects every group")
        values = [f"{group}:all" for group in GROUPS]
    for raw in values:
        group, sep, spec = raw.partition(":")
        group = group.strip().lower()
        spec = spec.strip()
        if not sep or not group or not spec:
            raise ValueError(f"--pmu-counters expects GROUP:SELECTION (got {raw!r}); e.g. mve:all or cpu:ARM_PMU_INST_RETIRED")
        if group not in GROUPS:
            raise ValueError(f"--pmu-counters: unknown group {group!r} (expected one of: {', '.join(GROUPS)})")
        if group in selection:
            raise ValueError(f"--pmu-counters: group {group!r} given more than once")
        if spec.lower() in ("all", "default"):
            selection[group] = spec.lower()
        else:
            names = [name.strip() for name in spec.split(",")]
            if not any(names) or not all(names):
                raise ValueError(
                    f"--pmu-counters: {raw!r} names an empty counter for group {group!r}; "
                    "SELECTION must be all, default, or a comma-separated list of ARM_PMU_* names "
                    "with no blank entries."
                )
            selection[group] = names
        try:
            resolve_counter_selection({group: selection[group]})
        except UnsupportedCounterError as exc:
            raise ValueError(f"--pmu-counters: {exc}") from exc
    try:
        check_pass_count(counter_passes_for_selection(selection))
    except TooManyPassesError as exc:
        raise ValueError(f"--pmu-counters: {exc}") from exc
    return selection


def pmu_groups_to_selection(groups: Sequence[str]) -> PmuSelection:
    """The deprecated `--pmu-groups a,b` form: every named group at its default selection."""
    selection: PmuSelection = {}
    for group in groups:
        if group not in GROUPS:
            raise ValueError(f"--pmu-groups: unknown group {group!r} (expected one of: {', '.join(GROUPS)})")
        selection[group] = "default"
    return selection


def resolve_pmu_options(pmu_counters: Sequence[str], pmu_groups: Optional[str], *, warn: Callable[[str], None] = None) -> PmuSelection:
    """Combine the CLI's `--pmu-counters` (repeatable) and deprecated `--pmu-groups`
    into one selection; neither given means every group at its default."""
    if pmu_groups is not None:
        if pmu_counters:
            raise ValueError("--pmu-groups is deprecated and cannot be combined with --pmu-counters.")
        groups = parse_pmu_groups(pmu_groups)
        (warn or (lambda message: print(message, file=sys.stderr)))(
            f"[hardware] --pmu-groups is deprecated; use "
            f"{' '.join(f'--pmu-counters {g}:default' for g in groups)} instead."
        )
        return pmu_groups_to_selection(groups)
    if pmu_counters:
        return parse_pmu_counters(pmu_counters)
    return default_selection()


@contextlib.contextmanager
def generation_lock(repo_root: Path, cpu: str) -> Iterator[None]:
    """Serialize generation into one CPU's tree."""
    # Same-CPU boards share generated tests.
    try:
        import fcntl
    except ImportError:  # pragma: no cover - no flock on Windows
        yield
        return
    path = repo_root / "artifacts" / "generated_tests" / f".{cpu}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def generate_tests_for_board(
    repo_root: Path,
    board: BoardSpec,
    suite: str,
    float_precision: Optional[str] = None,
    cmsis_nn_root: Optional[Path] = None,
    select: Optional[CaseSelection] = None,
) -> None:
    """Run the generate step for the board's CPU and the requested suite, exactly as
    `helia_core_tester generate --cpu <board.cpu> --suite <suite>
    [--float-precision <float_precision>]` would. `float_precision` (f16/f32/both)
    is an explicit override when given; otherwise the TOML/env/default applies.
    `cmsis_nn_root` is the kernel tree the firmware compiles. `select` narrows
    generation to its ops, dtypes and case ids, like `generate --op/--dtype/--name`."""
    from ..core.logging import setup_logger
    from ..core.steps import GenerateStep

    config = _board_config(repo_root, board, suite, float_precision, cmsis_nn_root, select)
    setup_logger(verbosity=config.verbosity)
    with generation_lock(repo_root, board.cpu):
        result = GenerateStep(config).execute()
    if not (result.success or result.skipped):
        raise RuntimeError(f"Generation failed: {result.message}")


def _board_config(
    repo_root: Path, board: BoardSpec, suite: str, float_precision: Optional[str], cmsis_nn_root: Optional[Path] = None,
    select: Optional[CaseSelection] = None,
):
    """Generation config for the board's CPU."""
    from ..core.config import Config

    overrides = {"project_root", "cpu", "suite"}
    kwargs = {}
    # One suite may match nothing under both.
    if select is not None and suite != "both":
        for key, values in (("op_filter", select.ops), ("dtype_filter", select.dtypes), ("name_filter", select.case_ids)):
            if values:
                kwargs[key] = ",".join(values)
                overrides.add(key)
                # Other runs share this tree.
                kwargs["keep_unselected"] = True
                overrides.add("keep_unselected")
    if float_precision is not None:
        kwargs["float_precision"] = float_precision
        overrides.add("float_precision")
    if cmsis_nn_root is not None:
        kwargs["cmsis_nn_root"] = cmsis_nn_root
        overrides.add("cmsis_nn_root")
    return Config(
        project_root=repo_root,
        cpu=board.cpu,
        suite=suite,
        _explicit_overrides=overrides,
        **kwargs,
    )


def generation_precision(repo_root: Path, board: BoardSpec, options: "StreamOptions") -> Optional[str]:
    """Float precision generation resolves, or None."""
    from ..core.errors import CMSISNNToolsError

    try:
        config = _board_config(repo_root, board, options.suite, options.float_precision)
    except (CMSISNNToolsError, ValueError):
        return None
    # Null when the run has no float cases.
    if "float" not in config.effective_suites_for_cpu(board.cpu):
        return None
    return config.effective_float_precision_for_cpu(board.cpu)


def resolved_selection(repo_root: Path, board: BoardSpec, options: "StreamOptions") -> dict[str, Any]:
    """The case and counter selection the run used."""
    from .fvp_gate import DEFAULT_GATE

    selection = {
        "suite": options.suite,
        "limit": options.limit,
        "family": options.family,
        "test_name": options.test_name,
        "ops": list(options.ops),
        "dtypes": list(options.dtypes),
        "case_ids": list(options.case_ids),
        "precision": generation_precision(repo_root, board, options),
        "pmu_counters": options.pmu_counters,
        "fvp_gate": options.fvp_gate or DEFAULT_GATE,
        "compare": options.compare_record(),
    }
    if options.hidden_set is not None:
        selection["hidden_set"] = hidden_record(options.hidden_set, board.cpu)
    return selection


class HiddenSetError(RunRefused):
    """The hidden set cannot serve this run."""


def hidden_summary(hidden_set: Path, cpu: str) -> dict[str, Any]:
    """The hidden set's summary for cpu."""
    from ..generation.random_shapes import hidden_root

    path = hidden_root(hidden_set, cpu) / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {}
    if not isinstance(summary, dict) or not summary.get("seed_commitment"):
        raise HiddenSetError(f"No {cpu} hidden set in {hidden_set}")
    return summary


def hidden_record(hidden_set: Path, cpu: str) -> dict[str, Any]:
    """Seed commitment and case count."""
    from ..core.path_layout import generated_tests_dir

    root = generated_tests_dir(hidden_set, normalize_cpu(cpu))
    cases = sum(1 for case in root.glob("*/*") if (case / "descriptor.yaml").is_file())
    return {"seed_commitment": hidden_summary(hidden_set, cpu)["seed_commitment"], "cases": cases}


def hidden_bundles(repo_root: Path, board: BoardSpec, options: "StreamOptions") -> list:
    """Bridge the hidden set; mark each case."""
    from .session_runner import build_generated_test_case_bundles

    from .case_bundle import hidden_bundle

    hidden_summary(options.hidden_set, board.cpu)
    # No FVP report names hidden cases.
    bundles, skipped = build_generated_test_case_bundles(
        repo_root, cpu=normalize_cpu(board.cpu), family=None, suite="int", fvp_gate="off",
        board_id=board.id, tests_root=options.hidden_set,
    )
    # A partial set skews the score.
    if skipped or not bundles:
        raise HiddenSetError(f"{len(skipped)} hidden case(s) cannot run; {len(bundles)} can")
    return [hidden_bundle(bundle) for bundle in bundles]


@dataclass
class StreamOptions:
    suite: str = "int"
    family: Optional[str] = None
    test_name: Optional[str] = None
    limit: Optional[int] = None
    # Same matching as `generate --op/--dtype`.
    ops: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    # `{group: "all" | "default" | [names]}` -- see parse_pmu_counters().
    pmu_counters: PmuSelection = field(default_factory=default_selection)
    fvp_gate: Optional[str] = None
    session_id: Optional[str] = None
    float_precision: Optional[str] = None
    """Config.float_precision for the generate step when `--precision` was given (f16/f32)."""
    strict_compare: bool = False
    """Integer outputs must match exactly: no tolerance."""
    golden_from: Optional[Path] = None
    """Result bundle whose outputs replace the goldens; implies strict."""
    golden_allow_failed: bool = False
    """Accept golden cases the past run failed."""
    hidden_set: Optional[Path] = None
    """Root from `generate --hidden-dir`; its cases join the run."""

    def selection(self) -> CaseSelection:
        """The op, dtype and id filters."""
        from .generated_test_bridge import CaseSelection

        return CaseSelection(ops=self.ops, dtypes=self.dtypes, case_ids=self.case_ids)

    def compare_record(self) -> dict[str, Any]:
        """How outputs get judged, for the bundle."""
        return {
            "strict": self.strict_compare or self.golden_from is not None,
            "golden_from": str(self.golden_from) if self.golden_from else None,
            "golden_session_id": golden_session_id(self.golden_from),
        }


def golden_session_id(golden_dir: Optional[Path]) -> Optional[str]:
    """The baseline bundle's session id."""
    if golden_dir is None:
        return None
    try:
        manifest = json.loads((golden_dir / "session_manifest.json").read_text(encoding="utf-8"))
        return str(manifest["session_id"])
    except (OSError, ValueError, KeyError, TypeError):
        # Bundles are named by session.
        return golden_dir.name


def fit_to_board(board: BoardSpec, options: StreamOptions, *, explicit_pmu: bool) -> StreamOptions:
    """Narrow or refuse what the board cannot run."""
    if not get_cpu_profile(board.cpu).supports_execution_dtype("FP16"):
        if options.float_precision == "f16":
            raise ValueError(f"{board.id} ({board.cpu}) runs no FP16 cases.")
        # Config narrows "both" itself.
        if options.suite == "float":
            options.float_precision = options.float_precision or "f32"
    if board.pmu_tier == "dwt":
        events = any(p.counters for p in counter_passes_for_selection(options.pmu_counters))
        if explicit_pmu and events:
            raise ValueError(f"{board.id} has no PMU; it counts DWT cycles only.")
        # One empty pass: DWT cycles.
        options.pmu_counters = {"cpu": [CPU_CYCLES_NAME]}
    return options


@dataclass
class HardwareRunOutcome:
    session_id: str
    result: object
    bundle: Path
    skipped: list
    flash: Optional[FlashDecision] = None
    # Wall-clock seconds per stage (generate/build/flash/stream/total) and per case;
    # also written into the bundle's session_summary.json. See stage_timing().
    timing: Dict[str, Any] = field(default_factory=dict)
    # CMSIS-NN entry points timed; see entry_coverage.
    coverage: Optional[Dict[str, Any]] = None

    @property
    def failed_case_ids(self) -> list[str]:
        return [c.case_bundle.case_id for c in self.result.cases if not c.comparison.passed]


def prepare_bundles(
    repo_root: Path, board: BoardSpec, options: StreamOptions, hidden: Optional[list] = None,
) -> tuple[list, list]:
    """Bridge the cases; apply the compare mode."""
    from .case_bundle import strict_bundle
    from .generated_test_bridge import HW_CASE_SUFFIX, CaseSelection
    from .session_runner import build_generated_test_case_bundles, no_bridgeable_cases_error

    # Bridge once; the runner reuses it.
    select = options.selection()
    bundles, skipped = build_generated_test_case_bundles(
        repo_root, cpu=board.cpu, family=options.family, name_filter=options.test_name,
        limit=options.limit, suite=options.suite, fvp_gate=options.fvp_gate, board_id=board.id,
        select=select,
    )
    if select.case_ids:
        names = [b.case_id.removesuffix(HW_CASE_SUFFIX) for b in bundles] + [t.name for t, _ in skipped]
        missing = select.unmatched_ids(names)
        if missing:
            raise RunRefused(f"No case matches these ids: {', '.join(missing)}")
    if not bundles and not skipped and select != CaseSelection():
        raise RunRefused("No generated case matches --op/--dtype/--case-id.")
    if not bundles:
        raise no_bridgeable_cases_error(
            skipped, cpu=board.cpu, family=options.family, name_filter=options.test_name, suite=options.suite,
        )
    if options.hidden_set is not None:
        bundles += hidden if hidden is not None else hidden_bundles(repo_root, board, options)
    if options.golden_from is not None:
        bundles = golden_bundles(bundles, options.golden_from, allow_failed=options.golden_allow_failed)
    elif options.strict_compare:
        bundles = [strict_bundle(bundle) for bundle in bundles]
    return bundles, skipped


def golden_bundles(bundles: list, golden_dir: Path, *, allow_failed: bool) -> list:
    """Swap in past outputs; refuse misfits."""
    from .case_bundle import golden_bundle, golden_record, golden_usable, input_digest

    if not any((golden_dir / "correctness").glob("*.json")):
        raise RunRefused(f"Golden bundle has no results: {golden_dir}")
    records = {b.case_id: golden_record(b, golden_dir) for b in bundles}
    _refuse("Golden run is missing these cases", [i for i, r in records.items() if r is None])
    if not allow_failed:
        _refuse("Golden run failed these cases", [i for i, r in records.items() if r.get("passed") is not True])
    # Status-only cases use no past output.
    judged = [b for b in bundles if b.expected_status_code is None]
    digests = {b.case_id: records[b.case_id].get("input_digest") for b in judged}
    _refuse("Golden run has no input digest for", [i for i, d in digests.items() if not d])
    _refuse("Golden run used other inputs for", [b.case_id for b in judged if digests[b.case_id] != input_digest(b)])
    _refuse(f"No usable golden output in {golden_dir} for", [b.case_id for b in bundles if not golden_usable(b, golden_dir)])
    return [golden_bundle(bundle, golden_dir) for bundle in bundles]


def _refuse(reason: str, case_ids: list[str]) -> None:
    if case_ids:
        raise RunRefused(f"{reason}: {', '.join(case_ids)}")


def stream_generated_tests(
    repo_root: Path,
    board: BoardSpec,
    serial_no: int,
    *,
    build_dir: Path,
    options: StreamOptions,
    echo: Callable[[str], None],
    progress_to_stderr: bool = False,
    allow_unverified_firmware: bool = False,
    prepared: Optional[tuple[list, list]] = None,
) -> HardwareRunOutcome:
    """Stream the generated suite to already-flashed firmware and write the bundle.

    Preflight: the build dir must carry `hct_build_id.txt` so the session's TARGET_INFO
    can be checked against it; a missing stamp is an error unless
    `allow_unverified_firmware` says the caller knowingly streams to legacy firmware.
    """
    from .entry_coverage import build_coverage, write_coverage
    from .result_bundle import merge_summary
    from .session_runner import run_case_bundles

    session_id = options.session_id or default_session_id(board)

    expected_build_id = read_build_id(build_dir)
    if expected_build_id is None:
        stamp_missing = (
            f"{build_id_path(build_dir)} not found, so the firmware on the board cannot be verified "
            "against this build dir."
        )
        if not allow_unverified_firmware:
            raise RunRefused(
                f"{stamp_missing} Rebuild with `hardware build --board {board.id}` (which stamps it), "
                "or pass --allow-unverified-firmware to stream to legacy firmware unchecked."
            )
        echo(f"[hardware] WARNING: {stamp_missing} Continuing unverified (--allow-unverified-firmware).")

    bundles, skipped = prepared or prepare_bundles(repo_root, board, options)
    # The live progress printer aligns its [N/total] counter and case_id columns from
    # the first printed line instead of widening them as longer names show up mid-run.
    id_width = max(len(b.case_id) for b in bundles)
    counter_passes = counter_passes_for_selection(options.pmu_counters)
    # Refuse results the outbox cannot hold.
    for case_bundle in bundles:
        check_outbox_fits(counter_passes, int(case_bundle.manifest["timing"]["samples"]), case_bundle.case_id)
    echo(
        f"[hardware] Streaming generated tests to {board.id} (serial {serial_no}, session {session_id}, "
        f"firmware build id {expected_build_id or 'unverified'}, "
        f"{len(counter_passes)} PMU pass(es): {', '.join(p.name for p in counter_passes)})..."
    )
    progress = make_live_progress_printer(len(bundles), id_width=id_width, err=progress_to_stderr)

    # Per-case wall clock: the gap between consecutive CASE_COMPLETEs (the first case
    # also absorbs the target reset and TARGET_INFO/catalog exchange).
    case_seconds: Dict[str, float] = {}
    stream_started = time.monotonic()
    last_case_done = stream_started

    def on_case_complete(case) -> None:
        nonlocal last_case_done
        now = time.monotonic()
        case_seconds[case.case_bundle.case_id] = round(now - last_case_done, 4)
        last_case_done = now
        progress(case)

    result, bundle = run_case_bundles(
        repo_root,
        bundles,
        board=board,
        serial_no=serial_no,
        counter_passes=counter_passes,
        session_id=session_id,
        build_dir=build_dir,
        on_case_complete=on_case_complete,
        expected_build_id=expected_build_id,
        compare=options.compare_record(),
    )
    merge_summary(bundle, "selection", resolved_selection(repo_root, board, options))
    timing = {
        "stream_s": round(time.monotonic() - stream_started, 4),
        "batch_count": int(getattr(result, "batch_count", 1)),
        "cases": case_seconds,
    }
    # Unverified firmware: build symbols unknown.
    coverage = build_coverage(repo_root, result.cases, build_dir if expected_build_id else None)
    write_coverage(bundle, coverage, skipped)
    return HardwareRunOutcome(
        session_id=session_id, result=result, bundle=bundle, skipped=skipped, timing=timing, coverage=coverage,
    )


def finalize_timing(outcome: HardwareRunOutcome, *, generate_s: float = 0.0, echo: Callable[[str], None]) -> None:
    """Fold the stage times into outcome.timing, persist them in the bundle's
    session_summary.json, and print the one-line stage summary."""
    timing = dict(outcome.timing)
    timing["generate_s"] = round(generate_s, 4)
    timing["build_s"] = round(outcome.flash.build_seconds if outcome.flash else 0.0, 4)
    timing["flash_s"] = round(outcome.flash.flash_seconds if outcome.flash else 0.0, 4)
    timing["total_s"] = round(timing["generate_s"] + timing["build_s"] + timing["flash_s"] + timing.get("stream_s", 0.0), 4)
    outcome.timing = timing
    write_timing(outcome.bundle, timing)
    case_count = len(timing.get("cases", {}))
    echo(
        f"[hardware] timing: generate {timing['generate_s']:.1f}s  build {timing['build_s']:.1f}s  "
        f"flash {timing['flash_s']:.1f}s  stream {timing.get('stream_s', 0.0):.1f}s  "
        f"({case_count} case(s) in {timing.get('batch_count', 1)} batch(es), total {timing['total_s']:.1f}s)"
    )


def run_hardware_pipeline(
    repo_root: Path,
    board: BoardSpec,
    serial_no: int,
    *,
    options: StreamOptions,
    build_dir: Optional[Path] = None,
    skip_generate: bool = False,
    skip_flash: bool = False,
    force_flash: bool = False,
    jobs: Optional[int] = None,
    force_reconfigure: bool = False,
    echo: Callable[[str], None],
    progress_to_stderr: bool = False,
    allow_unverified_firmware: bool = False,
    app_options: Optional["AppOptions"] = None,
    update_dependencies: bool = False,
) -> HardwareRunOutcome:
    """stage kernels -> generate (board cpu) -> build -> flash unless the board already runs this build -> stream -> bundle."""
    if skip_flash and force_flash:
        raise ValueError("--skip-flash and --force-flash cannot be combined.")
    resolved_build_dir = resolve_build_dir(repo_root, board, build_dir)
    if app_options is None and not (skip_generate and skip_flash):
        from .nsx_app import resolve_options

        # Same resolution as the CLI.
        app_options = resolve_options(nsx_app_dir(resolved_build_dir), repo_root, follow_pin=not skip_flash)

    # Refuse a bad hidden set first.
    hidden = hidden_bundles(repo_root, board, options) if options.hidden_set is not None else None
    generate_s = 0.0
    if skip_generate:
        echo("[hardware] --skip-generate set; reusing existing artifacts/generated_tests.")
    else:
        # Generate against the firmware's kernels.
        if skip_flash:
            if update_dependencies:
                raise RunRefused("--skip-flash cannot update dependencies.")
            # Board keeps the built image.
            kernel_root = built_kernels(board, resolved_build_dir, app_options)
        else:
            kernel_root = stage_kernels(
                board, build_dir=resolved_build_dir, options=app_options, force_sync=force_reconfigure,
                update_dependencies=update_dependencies,
            )
        # Staging did the forced work.
        force_reconfigure = update_dependencies = False
        precision_note = f" float_precision={options.float_precision}" if options.float_precision else ""
        echo(f"[hardware] Generating tests (cpu={board.cpu} suite={options.suite}{precision_note} kernels={kernel_root})...")
        generate_started = time.monotonic()
        generate_tests_for_board(
            repo_root, board, options.suite, float_precision=options.float_precision, cmsis_nn_root=kernel_root,
            select=options.selection(),
        )
        generate_s = time.monotonic() - generate_started

    # Check goldens and hidden cases before touching the board.
    checked = options.golden_from is not None or options.hidden_set is not None
    prepared = prepare_bundles(repo_root, board, options, hidden) if checked else None
    flash: Optional[FlashDecision] = None
    if skip_flash:
        echo("[hardware] --skip-flash set; reusing firmware already running on the board.")
    else:
        flash = flash_firmware(
            board, serial_no, build_dir=resolved_build_dir, jobs=jobs, force_reconfigure=force_reconfigure, force=force_flash,
            options=app_options, update_dependencies=update_dependencies,
        )

    outcome = stream_generated_tests(
        repo_root, board, serial_no, build_dir=resolved_build_dir, options=options,
        echo=echo, progress_to_stderr=progress_to_stderr, allow_unverified_firmware=allow_unverified_firmware,
        prepared=prepared,
    )
    outcome.flash = flash
    if outcome.result is not None:
        finalize_timing(outcome, generate_s=generate_s, echo=echo)
    return outcome
