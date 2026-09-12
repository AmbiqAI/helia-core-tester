"""Orchestration behind `helia_core_tester hardware run` / `hardware stream`.

Everything here calls the Python entry points directly (the same GenerateStep the
top-level `generate` command uses, then the firmware build/flash helpers and the
RTT session runner) -- never a subprocess into the tester's own CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .boards import BoardSpec, default_session_id
from .firmware_build import FlashDecision, flash_firmware, resolve_build_dir
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
    pmu_groups: tuple[str, ...] = ("cpu", "memory", "mve")
    fvp_gate: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class HardwareRunOutcome:
    session_id: str
    result: object
    bundle: Path
    skipped: list
    flash: Optional[FlashDecision] = None

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
    echo(f"[hardware] Streaming generated tests to {board.id} (serial {serial_no}, session {session_id})...")
    on_case_complete = make_live_progress_printer(len(preview_bundles), id_width=id_width, err=progress_to_stderr)

    result, bundle, skipped = run_apollo510_generated_test_session(
        repo_root,
        serial_no=serial_no,
        board=board,
        requested_counter_groups=options.pmu_groups,
        session_id=session_id,
        build_dir=build_dir,
        family=options.family,
        name_filter=options.test_name,
        limit=options.limit,
        suite=options.suite,
        fvp_gate=options.fvp_gate,
        on_case_complete=on_case_complete,
    )
    return HardwareRunOutcome(session_id=session_id, result=result, bundle=bundle, skipped=skipped)


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

    if skip_generate:
        echo("[hardware] --skip-generate set; reusing existing artifacts/generated_tests.")
    else:
        echo(f"[hardware] Generating tests (cpu={board.cpu} suite={options.suite})...")
        generate_tests_for_board(repo_root, board, options.suite)

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
    return outcome
