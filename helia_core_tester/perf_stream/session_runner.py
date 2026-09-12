"""Run case bundles on a board over SEGGER RTT and write the result bundle.

Cases are streamed in batches: each batch is one fresh (reset-on-open) RTT session
and one SESSION_PLAN, sized from what the target announced in TARGET_INFO (cases
per plan, receive-buffer bytes, PMU passes) rather than from mirrored constants.
The batches' results are merged into a single SessionResult and result bundle.

Also owns turning `helia_core_tester generate` output into the CaseBundles a
session streams (`build_generated_test_case_bundles`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

from .boards import DEFAULT_BOARD_ID, BoardSpec, default_session_id, resolve_board
from .case_bundle import CaseBundle, build_abs_s8_case_bundle, build_convolve_s8_case_bundle, load_case_bundle
from .generated_test_bridge import (
    GeneratedTestCase,
    UnsupportedGeneratedTestError,
    bridged_families,
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)
from .measurement import CounterPass, counter_passes_for_selection
from .memory_report import generate_memory_report
from .pmu_catalog import default_selection
from .result_bundle import write_result_bundle
from .session import CaseRunResult, HostSession, SessionResult, TargetLimits
from .transport import JLinkRttTransport, Transport, symbol_address_from_elf
from .wire import session_plan_size
from ..core.config import VALID_SUITE_MODES

OnCaseComplete = Callable[[CaseRunResult], None]


# --- batching --------------------------------------------------------------------------


def take_batch(
    case_bundles: Sequence[CaseBundle],
    counter_passes: Sequence[CounterPass],
    limits: TargetLimits,
) -> list[CaseBundle]:
    """The longest prefix of `case_bundles` that fits one SESSION_PLAN under `limits`:
    at most `max_cases` cases, and an encoded plan (with these PMU passes) within
    `max_plan_bytes`. Order is preserved."""
    batch: list[CaseBundle] = []
    for bundle in case_bundles:
        candidate_ids = [b.case_id for b in batch] + [bundle.case_id]
        if batch and (len(batch) >= limits.max_cases or session_plan_size(candidate_ids, counter_passes) > limits.max_plan_bytes):
            break
        single = session_plan_size([bundle.case_id], counter_passes)
        if single > limits.max_plan_bytes:
            raise ValueError(
                f"Case {bundle.case_id!r} alone needs a {single}-byte SESSION_PLAN with "
                f"{len(counter_passes)} PMU pass(es), over the target's {limits.max_plan_bytes}-byte "
                "receive limit (TARGET_INFO max_rx_payload). Request fewer counters."
            )
        batch.append(bundle)
    return batch


def split_case_bundles_into_batches(
    case_bundles: Sequence[CaseBundle],
    counter_passes: Sequence[CounterPass],
    limits: TargetLimits,
) -> list[list[CaseBundle]]:
    """Greedily pack cases into batches with `take_batch` until none remain."""
    remaining = list(case_bundles)
    batches: list[list[CaseBundle]] = []
    while remaining:
        batch = take_batch(remaining, counter_passes, limits)
        batches.append(batch)
        remaining = remaining[len(batch):]
    return batches


# --- sessions --------------------------------------------------------------------------


def open_rtt_session(
    board: BoardSpec,
    serial_no: int,
    *,
    build_dir: Path,
    counter_passes: Sequence[CounterPass],
) -> tuple[HostSession, Transport, int]:
    """Open a fresh reset-on-open RTT session to the board's flashed firmware. Returns
    the host session, its transport (the caller closes it) and the RTT control-block
    address taken from the ELF in `build_dir`."""
    elf_path = build_dir / "perf_stream" / "hct_benchmark_server.elf"
    rtt_address = symbol_address_from_elf(str(elf_path), "_SEGGER_RTT")
    transport = JLinkRttTransport(
        serial_no=serial_no,
        chip_name=board.jlink_device,
        speed_khz=board.swd_speed_khz,
        rtt_address=rtt_address,
        reset_on_open=True,
        read_timeout_s=10.0,
    )
    return HostSession(transport, counter_passes=counter_passes), transport, rtt_address


def run_case_bundles(
    project_root: Path,
    case_bundles: Sequence[CaseBundle],
    *,
    board: BoardSpec,
    serial_no: int,
    counter_passes: Sequence[CounterPass],
    session_id: str | None = None,
    build_dir: Path | None = None,
    on_case_complete: OnCaseComplete | None = None,
) -> tuple[SessionResult, Path]:
    """Stream `case_bundles` to the board in as many sessions as the target's limits
    require, merge every case into one SessionResult, and write its result bundle.

    Every session starts with the target's TARGET_INFO, so the next batch is cut from
    the remaining cases only once that session's limits are known.
    """
    build_dir = build_dir or board.build_dir(project_root)
    sid = session_id or default_session_id(board)
    counter_passes = tuple(counter_passes)

    remaining = list(case_bundles)
    all_cases: list[CaseRunResult] = []
    all_trace: list[str] = []
    session_complete_cases = 0
    rtt_address = 0
    target_info = None
    limits: TargetLimits | None = None
    batch_index = 0
    while remaining:
        session, transport, rtt_address = open_rtt_session(board, serial_no, build_dir=build_dir, counter_passes=counter_passes)
        batch: list[CaseBundle] = []
        try:
            info = session.handshake()
            target_info = target_info or info
            limits = session.limits
            batch = take_batch(remaining, counter_passes, limits)
            result = session.run_many(batch, on_case_complete=on_case_complete)
        except RuntimeError as exc:
            candidates = [b.case_id for b in (batch or (remaining[: limits.max_cases] if limits else remaining))]
            raise RuntimeError(f"{exc} (batch {batch_index}, candidate case_ids={candidates})") from exc
        finally:
            transport.close()
        all_cases.extend(result.cases)
        all_trace.extend(f"batch{batch_index}:{entry}" for entry in result.protocol_trace)
        session_complete_cases += result.session_complete_cases
        remaining = remaining[len(batch):]
        batch_index += 1

    batch_count = batch_index
    merged_result = SessionResult(
        cases=tuple(all_cases),
        protocol_trace=tuple(all_trace),
        session_complete_cases=session_complete_cases,
        batch_count=batch_count,
        target_info=target_info,
    )

    memory_report = json.loads(generate_memory_report(board, project_root=project_root, build_dir=build_dir).read_text())
    kernel_catalog = json.loads((project_root / "cmake" / "perf_stream" / "kernel_catalog.json").read_text())
    host_log = (
        f"hardware session_id={sid}\n"
        f"board={board.id} chip={board.jlink_device} serial={serial_no} speed_khz={board.swd_speed_khz}\n"
        f"rtt_address=0x{rtt_address:08x}\n"
        f"counter_passes={[p.name for p in counter_passes]}\n"
        f"batch_count={batch_count} max_cases_per_session={limits.max_cases if limits else 0} "
        f"max_session_plan_bytes={limits.max_plan_bytes if limits else 0}\n"
        f"protocol_trace_len={len(merged_result.protocol_trace)}\n"
        f"case_ids={[b.case_id for b in case_bundles]}\n"
    )
    target_log = f"real {board.id} benchmark server over SEGGER RTT ({batch_count} batch(es))\n"
    bundle_root = write_result_bundle(
        merged_result,
        session_id=sid,
        output_root=project_root,
        memory_report=memory_report,
        kernel_catalog=kernel_catalog,
        target_info=board.target_info(),
        host_log_text=host_log,
        target_log_text=target_log,
    )
    return merged_result, bundle_root


def run_demo_session(
    project_root: Path,
    *,
    serial_no: int,
    board: BoardSpec | None = None,
    counter_passes: Sequence[CounterPass] | None = None,
    session_id: str | None = None,
    build_dir: Path | None = None,
) -> tuple[SessionResult, Path]:
    """Two-kernel synthetic demo session (arm_abs_s8 + arm_convolve_s8) on the board;
    library code only, not exposed on the CLI. `counter_passes` defaults to every
    PMU group at its default selection, like the hardware CLI."""
    board = board or resolve_board(DEFAULT_BOARD_ID)
    passes = tuple(counter_passes) if counter_passes is not None else counter_passes_for_selection(default_selection())
    abs_bundle = load_case_bundle(build_abs_s8_case_bundle(project_root, case_id="abs_hw_live").manifest_path)
    conv_bundle = load_case_bundle(build_convolve_s8_case_bundle(project_root, case_id="conv_hw_live").manifest_path)
    return run_case_bundles(
        project_root,
        [abs_bundle, conv_bundle],
        board=board,
        serial_no=serial_no,
        counter_passes=passes,
        session_id=session_id,
        build_dir=build_dir,
    )


# --- generated-test discovery ------------------------------------------------------------


def normalize_suites(suite: str) -> tuple[str, ...]:
    """Expand a --suite value into the concrete generated-test trees to walk.

    "both" runs int and float in a single session. That is safe because the two
    trees live side by side under artifacts/generated_tests/<suite>/<cpu>, every
    case is bridged and FVP-gated against its own suite, and case_ids are unique
    across suites -- so one flash and one result bundle can cover both.
    """
    normalized = str(suite).strip().lower()
    if normalized not in VALID_SUITE_MODES:
        raise ValueError(
            f"Invalid suite: {suite!r} (expected one of: {', '.join(sorted(VALID_SUITE_MODES))})"
        )
    return ("int", "float") if normalized == "both" else (normalized,)


def build_generated_test_case_bundles(
    project_root: Path,
    *,
    cpu: str = "cortex-m55",
    family: str | None = "ConvolutionFunctions",
    name_filter: str | None = None,
    limit: int | None = None,
    suite: str = "int",
    require_fvp_pass: bool = True,
    fvp_gate: str | None = None,
) -> tuple[list[CaseBundle], list[tuple[GeneratedTestCase, str]]]:
    """Discover generated (`helia_core_tester generate`) kernel tests and bridge the
    ones with real perf-stream firmware dispatch support into CaseBundles.

    `family=None` bridges every family with at least one registered builder (see
    `generated_test_bridge.bridged_families()`) instead of a single hardcoded family --
    i.e. runs the complete set of hardware-supported kernels across all families.
    `limit`, if given, is applied per-family (not globally) when `family=None`.
    `suite="int"` (default) discovers under artifacts/generated_tests/int; `suite="float"`
    discovers the FP16/FP32 tree; `suite="both"` walks both in one call, so a single
    flash and a single result bundle cover int and float together. `limit`, when given,
    applies per (suite, family).

    `require_fvp_pass` (default True) is forwarded to
    `build_case_bundle_from_generated_test`'s FVP-pass gate. Set to False to bridge
    every case with real firmware dispatch support regardless of whether a matching
    FVP report exists -- e.g. on hosts that cannot run the (Linux-only) Corstone-300
    FVP model at all, where a fresh FVP report can never be produced locally and the
    gate would otherwise skip every case.

    Returns (bridged_case_bundles, [(skipped_test, reason), ...]).
    """
    families = bridged_families() if family is None else [family]
    bundles: list[CaseBundle] = []
    skipped: list[tuple[GeneratedTestCase, str]] = []
    for suite_name in normalize_suites(suite):
        for fam in families:
            discovered = discover_generated_tests(
                project_root, cpu=cpu, family=fam, name_filter=name_filter, limit=limit, suite=suite_name
            )
            for test in discovered:
                try:
                    bundles.append(build_case_bundle_from_generated_test(
                        project_root, test, require_fvp_pass=require_fvp_pass, fvp_gate=fvp_gate))
                except UnsupportedGeneratedTestError as exc:
                    skipped.append((test, str(exc)))
    return bundles, skipped


def no_bridgeable_cases_error(
    skipped: Sequence[tuple[GeneratedTestCase, str]],
    *,
    cpu: str,
    family: str | None,
    name_filter: str | None,
    suite: str,
) -> RuntimeError:
    """The error to raise when discovery bridged nothing, leading with the reasons
    cases were rejected (an all-FVP-gate rejection in particular is fixed by refreshing
    or bypassing the gate, not by regenerating)."""
    base = (
        f"No bridgeable generated tests found for cpu={cpu} "
        f"family={family if family is not None else '<all bridged families>'} "
        f"name_filter={name_filter!r} suite={suite!r} (skipped {len(skipped)})"
    )
    if not skipped:
        return RuntimeError(f"{base}; run `helia_core_tester generate` first.")
    fvp_skips = [(t, r) for t, r in skipped if "FVP" in r or "artifact" in r]
    detail = "\n".join(f"  - {t.name}: {r}" for t, r in skipped[:5])
    if len(skipped) > 5:
        detail += f"\n  ... and {len(skipped) - 5} more"
    hint = ""
    if len(fvp_skips) == len(skipped):
        stale_only = all("does not match" in r or "no artifact_sha256" in r for _, r in fvp_skips)
        if stale_only:
            # Only --fvp-gate strict blocks on staleness, so the useful advice is
            # "stop being strict", not "bypass the gate".
            hint = (
                "\nEvery case was rejected as stale by --fvp-gate strict. Either refresh the "
                "report (`uv run helia_core_tester build && uv run helia_core_tester run`) or "
                "drop back to --fvp-gate advisory, which runs stale cases and records them as "
                "stale in case_summary.csv."
            )
        else:
            hint = (
                "\nEvery case was rejected by the FVP gate because the FVP recorded a FAILURE "
                "for these exact artifacts -- that is evidence the kernel is wrong, not a stale "
                "report. Investigate before overriding; --fvp-gate off will run them anyway and "
                "record fvp_status=failed in case_summary.csv."
            )
    return RuntimeError(f"{base}:\n{detail}{hint}")
