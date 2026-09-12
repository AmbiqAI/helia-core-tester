"""Orchestration behind `helia_core_tester hardware run` / `hardware stream`.

Everything here calls the Python entry points directly (the same GenerateStep the
top-level `generate` command uses, then the firmware build/flash helpers and the
RTT session runner) -- never a subprocess into the tester's own CLI.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .boards import BoardSpec, default_session_id
from .firmware_build import FlashDecision, flash_firmware, resolve_build_dir
from .measurement import UnsupportedCounterError, counter_passes_for_selection, resolve_counter_selection
from .pmu_catalog import GROUPS, default_selection
from .result_bundle import write_timing
from .run_summary import make_live_progress_printer

PRECISION_SUFFIX = {"fp16": "_f16", "fp32": "_f32"}


def apply_precision(precision: Optional[str], suite: str, test_name: Optional[str]) -> tuple[str, Optional[str]]:
    """Expand `--precision fp16|fp32` into (suite, test_name).

    It forces the float suite and narrows the test-name substring filter to the
    `_f16`/`_f32` suffix. Refuses `--suite both` (the shortcut selects float cases
    only) and an explicit `--test-name` (both are a single substring match, so
    combining them would silently narrow to whichever cases contain both).
    """
    if precision is None:
        return suite, test_name
    if suite == "both":
        raise ValueError("--precision cannot be combined with --suite both (it selects float cases only).")
    suffix = PRECISION_SUFFIX.get(precision.lower())
    if suffix is None:
        raise ValueError(f"--precision must be 'fp16' or 'fp32' (got '{precision}').")
    if test_name:
        raise ValueError("--precision and --test-name cannot be combined (both filter via a single substring match).")
    return "float", suffix


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
    (`mve:all`, `cpu:default`, `mve:ARM_PMU_MVE_STALL,ARM_PMU_MVE_PRED`). Groups keep
    their command-line order, which is the order the PMU passes run in. Unknown groups
    and counter names are rejected here, naming the valid choices.
    """
    selection: PmuSelection = {}
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
            selection[group] = [name.strip() for name in spec.split(",") if name.strip()]
        try:
            resolve_counter_selection({group: selection[group]})
        except UnsupportedCounterError as exc:
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


def generate_tests_for_board(repo_root: Path, board: BoardSpec, suite: str) -> None:
    """Run the generate step for the board's CPU and the requested suite, exactly as
    `helia_core_tester generate --cpu <board.cpu> --suite <suite>` would."""
    from ..core.config import Config
    from ..core.logging import setup_logger
    from ..core.steps import GenerateStep

    config = Config(
        project_root=repo_root,
        cpu=board.cpu,
        suite=suite,
        _explicit_overrides={"project_root", "cpu", "suite"},
    )
    setup_logger(verbosity=config.verbosity)
    result = GenerateStep(config).execute()
    if not (result.success or result.skipped):
        raise RuntimeError(f"Generation failed: {result.message}")


@dataclass
class StreamOptions:
    suite: str = "int"
    family: Optional[str] = None
    test_name: Optional[str] = None
    limit: Optional[int] = None
    # `{group: "all" | "default" | [names]}` -- see parse_pmu_counters().
    pmu_counters: PmuSelection = field(default_factory=default_selection)
    fvp_gate: Optional[str] = None
    session_id: Optional[str] = None


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

    @property
    def failed_case_ids(self) -> list[str]:
        return [c.case_bundle.case_id for c in self.result.cases if not c.comparison.passed]


def stream_generated_tests(
    repo_root: Path,
    board: BoardSpec,
    serial_no: int,
    *,
    build_dir: Path,
    options: StreamOptions,
    echo: Callable[[str], None],
    progress_to_stderr: bool = False,
) -> HardwareRunOutcome:
    """Stream the generated suite to already-flashed firmware and write the bundle."""
    from .hardware_run import build_generated_test_case_bundles, run_apollo510_generated_test_session

    session_id = options.session_id or default_session_id(board)

    # Discover the bridgeable case count/case_ids up front (cheap: just descriptor/header
    # parsing, no hardware I/O) purely so the live progress printer can align its
    # [N/total] counter and case_id columns from the very first printed line instead of
    # widening them as longer names are discovered mid-run.
    preview_bundles, _preview_skipped = build_generated_test_case_bundles(
        repo_root, cpu=board.cpu, family=options.family, name_filter=options.test_name,
        limit=options.limit, suite=options.suite, fvp_gate=options.fvp_gate,
    )
    id_width = max((len(b.case_id) for b in preview_bundles), default=0)
    counter_passes = counter_passes_for_selection(options.pmu_counters)
    echo(
        f"[hardware] Streaming generated tests to {board.id} (serial {serial_no}, session {session_id}, "
        f"{len(counter_passes)} PMU pass(es): {', '.join(p.name for p in counter_passes)})..."
    )
    progress = make_live_progress_printer(len(preview_bundles), id_width=id_width, err=progress_to_stderr)

    # Per-case wall clock: the gap between consecutive CASE_COMPLETEs (the first case of
    # every batch also absorbs that batch's target reset and HELLO/catalog exchange).
    case_seconds: Dict[str, float] = {}
    stream_started = time.monotonic()
    last_case_done = stream_started

    def on_case_complete(case) -> None:
        nonlocal last_case_done
        now = time.monotonic()
        case_seconds[case.case_bundle.case_id] = round(now - last_case_done, 4)
        last_case_done = now
        progress(case)

    result, bundle, skipped = run_apollo510_generated_test_session(
        repo_root,
        serial_no=serial_no,
        board=board,
        counter_passes=counter_passes,
        session_id=session_id,
        build_dir=build_dir,
        family=options.family,
        name_filter=options.test_name,
        limit=options.limit,
        suite=options.suite,
        fvp_gate=options.fvp_gate,
        on_case_complete=on_case_complete,
    )
    timing = {
        "stream_s": round(time.monotonic() - stream_started, 4),
        "batch_count": int(getattr(result, "batch_count", 1)),
        "cases": case_seconds,
    }
    return HardwareRunOutcome(session_id=session_id, result=result, bundle=bundle, skipped=skipped, timing=timing)


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
    jobs: Optional[int] = None,
    force_reconfigure: bool = False,
    echo: Callable[[str], None],
    progress_to_stderr: bool = False,
) -> HardwareRunOutcome:
    """generate (board cpu) -> build -> flash if the ELF changed -> stream -> bundle."""
    resolved_build_dir = resolve_build_dir(repo_root, board, build_dir)

    generate_s = 0.0
    if skip_generate:
        echo("[hardware] --skip-generate set; reusing existing artifacts/generated_tests.")
    else:
        echo(f"[hardware] Generating tests (cpu={board.cpu} suite={options.suite})...")
        generate_started = time.monotonic()
        generate_tests_for_board(repo_root, board, options.suite)
        generate_s = time.monotonic() - generate_started

    flash: Optional[FlashDecision] = None
    if skip_flash:
        echo("[hardware] --skip-flash set; reusing firmware already running on the board.")
    else:
        flash = flash_firmware(
            board, serial_no, build_dir=resolved_build_dir, jobs=jobs, force_reconfigure=force_reconfigure
        )

    outcome = stream_generated_tests(
        repo_root, board, serial_no, build_dir=resolved_build_dir, options=options,
        echo=echo, progress_to_stderr=progress_to_stderr,
    )
    outcome.flash = flash
    if outcome.result is not None:
        finalize_timing(outcome, generate_s=generate_s, echo=echo)
    return outcome
