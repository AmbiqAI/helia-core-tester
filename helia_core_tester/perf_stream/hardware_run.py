"""Real RTT hardware runner for the perf-stream benchmark server (board-keyed; Apollo510 EVB today)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

from .benchmark_firmware_report import generate_benchmark_server_memory_report
from .boards import DEFAULT_BOARD_ID, BoardSpec, default_session_id, resolve_board
from .case_bundle import CaseBundle, build_abs_s8_case_bundle, build_convolve_s8_case_bundle, load_case_bundle
from .generated_test_bridge import (
    GeneratedTestCase,
    UnsupportedGeneratedTestError,
    bridged_families,
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)
from .hctp import HEADER_SIZE
from .measurement import CounterPass, counter_passes_for_selection
from .pmu_catalog import default_selection
from .result_bundle import write_result_bundle
from .session import CaseRunResult, HostSession, SessionResult, load_plan_size
from .transport import JLinkRttTransport, symbol_address_from_elf
from ..core.config import VALID_SUITE_MODES

# Must match HCT_SERVER_MAX_CASES in cmake/perf_stream/benchmark_server_session.h.
# The firmware allocates fixed-size `planned_case_ids`/`planned_kernel_ids` arrays
# sized to this constant and rejects a LOAD_PLAN that names more cases than this
# (handle_load_plan() answers with an ERROR frame). Callers with more cases than
# this (e.g. run_apollo510_generated_test_session over a whole operator family) are
# split into multiple sequential sessions; see _run_case_bundles_in_batches() below.
MAX_CASES_PER_SESSION = 32

# Must match HCT_SERVER_RX_BUFFER_BYTES in benchmark_server_session.h: the firmware
# decodes host frames out of a fixed 2 KiB receive buffer, so a LOAD_PLAN payload
# (case ids can be up to 96 characters, plus one entry per PMU pass) has to fit in
# it too. The target also advertises this bound in HELLO (max_rx_payload) and the
# session refuses to send a plan that exceeds it; the batch splitter below keeps
# every plan under this same constant up front.
FIRMWARE_RX_BUFFER_BYTES = 2048
MAX_LOAD_PLAN_PAYLOAD_BYTES = FIRMWARE_RX_BUFFER_BYTES - HEADER_SIZE


def default_counter_passes() -> tuple[CounterPass, ...]:
    """`cpu:default,memory:default,mve:default` -- the hardware CLI's default."""
    return counter_passes_for_selection(default_selection())


def split_case_bundles_into_batches(
    case_bundles: Sequence[CaseBundle],
    counter_passes: Sequence[CounterPass],
    *,
    max_cases: int = MAX_CASES_PER_SESSION,
    max_plan_bytes: int = MAX_LOAD_PLAN_PAYLOAD_BYTES,
) -> list[list[CaseBundle]]:
    """Greedily pack cases into batches of at most `max_cases` whose encoded LOAD_PLAN
    (with these PMU passes) stays within `max_plan_bytes`. Order is preserved."""
    batches: list[list[CaseBundle]] = []
    current: list[CaseBundle] = []
    for bundle in case_bundles:
        candidate_ids = [b.case_id for b in current] + [bundle.case_id]
        if current and (len(current) >= max_cases or load_plan_size(candidate_ids, counter_passes) > max_plan_bytes):
            batches.append(current)
            current = []
        single = load_plan_size([bundle.case_id], counter_passes)
        if single > max_plan_bytes:
            raise ValueError(
                f"Case {bundle.case_id!r} alone needs a {single}-byte LOAD_PLAN with "
                f"{len(counter_passes)} PMU pass(es), over the firmware's {max_plan_bytes}-byte "
                "receive limit. Request fewer counters."
            )
        current.append(bundle)
    if current:
        batches.append(current)
    return batches


def _run_single_session(
    project_root: Path,
    case_bundles: list[CaseBundle],
    *,
    serial_no: int,
    chip_name: str,
    speed_khz: int,
    counter_passes: Sequence[CounterPass],
    build_dir: Path,
    on_case_complete: Callable[[CaseRunResult], None] | None = None,
) -> tuple[SessionResult, int]:
    """Open one fresh (reset-on-open) RTT session and run exactly one LOAD_PLAN
    worth of case bundles. Callers must keep len(case_bundles) <= MAX_CASES_PER_SESSION
    and the encoded plan within MAX_LOAD_PLAN_PAYLOAD_BYTES (see split_case_bundles_into_batches).
    """
    if len(case_bundles) > MAX_CASES_PER_SESSION:
        raise ValueError(
            f"Cannot run {len(case_bundles)} cases in a single session: firmware "
            f"HCT_SERVER_MAX_CASES={MAX_CASES_PER_SESSION} rejects larger plans. "
            "Split into batches of at most MAX_CASES_PER_SESSION first."
        )
    elf_path = build_dir / "perf_stream" / "hct_benchmark_server.elf"
    rtt_address = symbol_address_from_elf(str(elf_path), "_SEGGER_RTT")

    transport = JLinkRttTransport(
        serial_no=serial_no,
        chip_name=chip_name,
        speed_khz=speed_khz,
        rtt_address=rtt_address,
        reset_on_open=True,
        read_timeout_s=10.0,
    )
    try:
        result = HostSession(transport, counter_passes=counter_passes).run_many(
            case_bundles, on_case_complete=on_case_complete
        )
    finally:
        transport.close()
    return result, rtt_address


def _run_case_bundles_on_apollo510(
    project_root: Path,
    case_bundles: list[CaseBundle],
    *,
    serial_no: int,
    chip_name: str,
    speed_khz: int,
    counter_passes: Sequence[CounterPass],
    session_id: str | None,
    build_dir: Path | None,
    board: BoardSpec,
    on_case_complete: Callable[[CaseRunResult], None] | None = None,
) -> tuple[SessionResult, Path]:
    build_dir = build_dir or board.build_dir(project_root)
    result, rtt_address = _run_single_session(
        project_root,
        case_bundles,
        serial_no=serial_no,
        chip_name=chip_name,
        speed_khz=speed_khz,
        counter_passes=counter_passes,
        build_dir=build_dir,
        on_case_complete=on_case_complete,
    )

    memory_report_path = generate_benchmark_server_memory_report(build_dir=build_dir)
    memory_report = json.loads(memory_report_path.read_text())
    kernel_catalog = json.loads((project_root / "cmake" / "perf_stream" / "kernel_catalog.json").read_text())
    sid = session_id or default_session_id(board)
    host_log = (
        f"hardware session_id={sid}\n"
        f"board={board.id} chip={chip_name} serial={serial_no} speed_khz={speed_khz}\n"
        f"rtt_address=0x{rtt_address:08x}\n"
        f"counter_passes={[p.name for p in counter_passes]}\n"
        f"protocol_trace_len={len(result.protocol_trace)}\n"
        f"case_ids={[b.case_id for b in case_bundles]}\n"
    )
    target_log = f"real {board.id} benchmark server over SEGGER RTT\n"
    bundle_root = write_result_bundle(
        result,
        session_id=sid,
        output_root=project_root,
        memory_report=memory_report,
        kernel_catalog=kernel_catalog,
        target_info=board.target_info(),
        host_log_text=host_log,
        target_log_text=target_log,
    )
    return result, bundle_root


def _run_case_bundles_in_batches(
    project_root: Path,
    case_bundles: list[CaseBundle],
    *,
    serial_no: int,
    chip_name: str,
    speed_khz: int,
    counter_passes: Sequence[CounterPass],
    session_id: str | None,
    build_dir: Path | None,
    board: BoardSpec,
    on_case_complete: Callable[[CaseRunResult], None] | None = None,
) -> tuple[SessionResult, Path]:
    """Like _run_case_bundles_on_apollo510, but transparently splits case_bundles
    into batches (at most MAX_CASES_PER_SESSION cases, LOAD_PLAN within the firmware's
    receive buffer) and runs one fresh (reset-on-open) RTT session per batch, merging
    all cases into a single SessionResult/result bundle.
    """
    build_dir = build_dir or board.build_dir(project_root)
    sid = session_id or default_session_id(board)

    all_cases: list = []
    all_trace: list[str] = []
    session_complete_cases = 0
    rtt_address = 0
    hello = None
    batches = split_case_bundles_into_batches(case_bundles, counter_passes)
    batch_count = len(batches)

    for batch_index, batch in enumerate(batches):
        try:
            result, rtt_address = _run_single_session(
                project_root,
                batch,
                serial_no=serial_no,
                chip_name=chip_name,
                speed_khz=speed_khz,
                counter_passes=counter_passes,
                build_dir=build_dir,
                on_case_complete=on_case_complete,
            )
        except RuntimeError as exc:
            batch_case_ids = [b.case_id for b in batch]
            raise RuntimeError(
                f"{exc} (batch {batch_index}/{batch_count - 1}, candidate case_ids={batch_case_ids})"
            ) from exc
        all_cases.extend(result.cases)
        all_trace.extend(f"batch{batch_index}:{entry}" for entry in result.protocol_trace)
        session_complete_cases += result.session_complete_cases
        hello = hello or result.hello

    merged_result = SessionResult(
        cases=tuple(all_cases),
        protocol_trace=tuple(all_trace),
        session_complete_cases=session_complete_cases,
        batch_count=batch_count,
        hello=hello,
    )

    memory_report_path = generate_benchmark_server_memory_report(build_dir=build_dir)
    memory_report = json.loads(memory_report_path.read_text())
    kernel_catalog = json.loads((project_root / "cmake" / "perf_stream" / "kernel_catalog.json").read_text())
    host_log = (
        f"hardware session_id={sid}\n"
        f"board={board.id} chip={chip_name} serial={serial_no} speed_khz={speed_khz}\n"
        f"rtt_address=0x{rtt_address:08x}\n"
        f"counter_passes={[p.name for p in counter_passes]}\n"
        f"batch_count={batch_count} max_cases_per_session={MAX_CASES_PER_SESSION} "
        f"max_load_plan_bytes={MAX_LOAD_PLAN_PAYLOAD_BYTES}\n"
        f"protocol_trace_len={len(merged_result.protocol_trace)}\n"
        f"case_ids={[b.case_id for b in case_bundles]}\n"
    )
    target_log = f"real {board.id} benchmark server over SEGGER RTT (multi-batch)\n"
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


def run_apollo510_stream_session(
    project_root: Path,
    *,
    serial_no: int,
    board: BoardSpec | None = None,
    chip_name: str | None = None,
    speed_khz: int | None = None,
    counter_passes: Sequence[CounterPass] | None = None,
    session_id: str | None = None,
    build_dir: Path | None = None,
) -> tuple[SessionResult, Path]:
    """Two-kernel synthetic demo session (arm_abs_s8 + arm_convolve_s8); library
    code only, not exposed on the CLI. `chip_name`/`speed_khz` default to the board's."""
    board = board or resolve_board(DEFAULT_BOARD_ID)
    counter_passes = tuple(counter_passes) if counter_passes is not None else default_counter_passes()
    abs_bundle = load_case_bundle(build_abs_s8_case_bundle(project_root, case_id="abs_hw_live").manifest_path)
    conv_bundle = load_case_bundle(build_convolve_s8_case_bundle(project_root, case_id="conv_hw_live").manifest_path)
    return _run_case_bundles_on_apollo510(
        project_root,
        [abs_bundle, conv_bundle],
        serial_no=serial_no,
        chip_name=chip_name or board.jlink_device,
        speed_khz=speed_khz or board.swd_speed_khz,
        counter_passes=counter_passes,
        session_id=session_id,
        build_dir=build_dir,
        board=board,
    )


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
    `build_case_bundle_from_generated_test`'s Phase 2 FVP-pass gate. Set to False to
    bridge every case with real firmware dispatch support regardless of whether a
    matching FVP report exists -- e.g. on hosts that cannot run the (Linux-only)
    Corstone-300 FVP model at all, where a fresh FVP report can never be produced
    locally and the gate would otherwise skip every case.

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


def run_apollo510_generated_test_session(
    project_root: Path,
    *,
    serial_no: int,
    board: BoardSpec | None = None,
    chip_name: str | None = None,
    speed_khz: int | None = None,
    counter_passes: Sequence[CounterPass] | None = None,
    session_id: str | None = None,
    build_dir: Path | None = None,
    cpu: str | None = None,
    family: str | None = "ConvolutionFunctions",
    name_filter: str | None = None,
    limit: int | None = None,
    suite: str = "int",
    require_fvp_pass: bool = True,
    fvp_gate: str | None = None,
    on_case_complete: Callable[[CaseRunResult], None] | None = None,
) -> tuple[SessionResult, Path, list[tuple[GeneratedTestCase, str]]]:
    """Run real `helia_core_tester generate`-produced kernel tests (with their real golden
    data) against connected Apollo510 hardware over the streaming HCTP/RTT session,
    instead of the hand-authored abs/convolve demo cases.

    `board` (default apollo510_evb) supplies the SEGGER device name, SWD speed, CPU
    and result-bundle target info; `chip_name`/`speed_khz`/`cpu` override individual
    fields when given explicitly. `family=None` bridges every family with real firmware
    dispatch support (see `build_generated_test_case_bundles`), i.e. runs the complete
    hardware-supported suite in one session (transparently batched). `suite="int"`
    (default) or `suite="float"` selects which generated-test tree to discover from.
    `require_fvp_pass` (default True) is forwarded to `build_generated_test_case_bundles`
    -- set to False on hosts that cannot run the FVP model at all (see its own docstring).

    `counter_passes` (default: every PMU group at its default selection, see
    pmu_catalog.DEFAULT_SELECTIONS) is the list of PMU passes run per case; each pass
    times the kernel again with up to four (chained, 32-bit) event counters.

    Transparently splits the discovered/bridged cases into batches of at most
    MAX_CASES_PER_SESSION (matching firmware HCT_SERVER_MAX_CASES) whose LOAD_PLAN fits
    the firmware receive buffer, and runs one fresh reset-on-open RTT session per batch,
    merging all cases into a single SessionResult/result bundle.
    """
    board = board or resolve_board(DEFAULT_BOARD_ID)
    cpu = cpu or board.cpu
    counter_passes = tuple(counter_passes) if counter_passes is not None else default_counter_passes()
    bundles, skipped = build_generated_test_case_bundles(
        project_root, cpu=cpu, family=family, name_filter=name_filter, limit=limit, suite=suite,
        require_fvp_pass=require_fvp_pass,
        fvp_gate=fvp_gate,
    )
    if not bundles:
        base = (
            f"No bridgeable generated tests found for cpu={cpu} "
            f"family={family if family is not None else '<all bridged families>'} "
            f"name_filter={name_filter!r} suite={suite!r} (skipped {len(skipped)})"
        )
        if not skipped:
            raise RuntimeError(f"{base}; run `helia_core_tester generate` first.")
        # Cases were discovered but every one was rejected, so "run generate"
        # is the wrong remedy. Lead with the actual reasons -- an all-FVP-gate
        # rejection in particular is fixed by refreshing or bypassing the gate,
        # not by regenerating.
        fvp_skips = [(t, r) for t, r in skipped if "FVP" in r or "artifact" in r]
        detail = "\n".join(f"  - {t.name}: {r}" for t, r in skipped[:5])
        if len(skipped) > 5:
            detail += f"\n  ... and {len(skipped) - 5} more"
        hint = ""
        if len(fvp_skips) == len(skipped):
            stale_only = all("does not match" in r or "no artifact_sha256" in r for _, r in fvp_skips)
            if stale_only:
                # Only --fvp-gate strict blocks on staleness, so the useful
                # advice is "stop being strict", not "bypass the gate".
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
        raise RuntimeError(f"{base}:\n{detail}{hint}")
    result, bundle_root = _run_case_bundles_in_batches(
        project_root,
        bundles,
        serial_no=serial_no,
        chip_name=chip_name or board.jlink_device,
        speed_khz=speed_khz or board.swd_speed_khz,
        counter_passes=counter_passes,
        session_id=session_id,
        build_dir=build_dir,
        board=board,
        on_case_complete=on_case_complete,
    )
    return result, bundle_root, skipped

