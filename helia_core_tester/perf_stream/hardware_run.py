"""Real Apollo510 RTT hardware runner for the perf-stream benchmark server."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import UTC, datetime
from typing import Callable

from .benchmark_firmware_report import generate_benchmark_server_memory_report
from .case_bundle import CaseBundle, build_abs_s8_case_bundle, build_convolve_s8_case_bundle, load_case_bundle
from .generated_test_bridge import (
    GeneratedTestCase,
    UnsupportedGeneratedTestError,
    bridged_families,
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)
from .result_bundle import write_result_bundle
from .session import CaseRunResult, HostSession, SessionResult
from .transport import JLinkRttTransport, symbol_address_from_elf

# Must match HCT_SERVER_MAX_CASES in cmake/perf_stream/benchmark_server_session.h.
# The firmware allocates fixed-size `planned_case_ids`/`planned_kernel_ids` arrays
# sized to this constant and *silently drops* a LOAD_PLAN that names more cases
# than this (handle_load_plan() returns HCTP_STATUS_INVALID_ARGUMENT with no
# reply frame sent) -- so a session with more cases than this will hang the
# host with "Transport stalled without a complete frame." instead of erroring
# cleanly. Callers with more cases than this (e.g. run_apollo510_generated_test_session
# over a whole operator family) must split into multiple sequential sessions;
# see _run_case_bundles_in_batches() below.
MAX_CASES_PER_SESSION = 4


def _run_single_session(
    project_root: Path,
    case_bundles: list[CaseBundle],
    *,
    serial_no: int,
    chip_name: str,
    speed_khz: int,
    requested_counter_groups: tuple[str, ...],
    build_dir: Path,
    on_case_complete: Callable[[CaseRunResult], None] | None = None,
) -> tuple[SessionResult, int]:
    """Open one fresh (reset-on-open) RTT session and run exactly one LOAD_PLAN
    worth of case bundles. Callers must keep len(case_bundles) <= MAX_CASES_PER_SESSION
    or the firmware will silently drop the plan (see MAX_CASES_PER_SESSION above).
    """
    if len(case_bundles) > MAX_CASES_PER_SESSION:
        raise ValueError(
            f"Cannot run {len(case_bundles)} cases in a single session: firmware "
            f"HCT_SERVER_MAX_CASES={MAX_CASES_PER_SESSION} silently drops larger plans. "
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
        result = HostSession(transport, requested_counter_groups=requested_counter_groups).run_many(
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
    requested_counter_groups: tuple[str, ...],
    session_id: str | None,
    build_dir: Path | None,
    session_id_prefix: str,
    on_case_complete: Callable[[CaseRunResult], None] | None = None,
) -> tuple[SessionResult, Path]:
    build_dir = build_dir or (project_root / "build" / "perf_stream" / "benchmark_server_gcc2")
    result, rtt_address = _run_single_session(
        project_root,
        case_bundles,
        serial_no=serial_no,
        chip_name=chip_name,
        speed_khz=speed_khz,
        requested_counter_groups=requested_counter_groups,
        build_dir=build_dir,
        on_case_complete=on_case_complete,
    )

    memory_report_path = generate_benchmark_server_memory_report(build_dir=build_dir)
    memory_report = json.loads(memory_report_path.read_text())
    kernel_catalog = json.loads((project_root / "cmake" / "perf_stream" / "kernel_catalog.json").read_text())
    sid = session_id or f"{session_id_prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    host_log = (
        f"hardware session_id={sid}\n"
        f"chip={chip_name} serial={serial_no} speed_khz={speed_khz}\n"
        f"rtt_address=0x{rtt_address:08x}\n"
        f"requested_counter_groups={requested_counter_groups}\n"
        f"protocol_trace_len={len(result.protocol_trace)}\n"
        f"case_ids={[b.case_id for b in case_bundles]}\n"
    )
    target_log = "real Apollo510 benchmark server over SEGGER RTT\n"
    bundle_root = write_result_bundle(
        result,
        session_id=sid,
        output_root=project_root,
        memory_report=memory_report,
        kernel_catalog=kernel_catalog,
        target_info={"board": "apollo510_evb", "cpu": "cortex-m55", "transport": "jlink-rtt"},
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
    requested_counter_groups: tuple[str, ...],
    session_id: str | None,
    build_dir: Path | None,
    session_id_prefix: str,
    on_case_complete: Callable[[CaseRunResult], None] | None = None,
) -> tuple[SessionResult, Path]:
    """Like _run_case_bundles_on_apollo510, but transparently splits case_bundles
    into batches of at most MAX_CASES_PER_SESSION and runs one fresh (reset-on-open)
    RTT session per batch, merging all cases into a single SessionResult/result bundle.
    """
    build_dir = build_dir or (project_root / "build" / "perf_stream" / "benchmark_server_gcc2")
    sid = session_id or f"{session_id_prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    all_cases: list = []
    all_trace: list[str] = []
    session_complete_cases = 0
    rtt_address = 0
    batch_count = (len(case_bundles) + MAX_CASES_PER_SESSION - 1) // MAX_CASES_PER_SESSION

    for batch_index in range(batch_count):
        start = batch_index * MAX_CASES_PER_SESSION
        batch = case_bundles[start : start + MAX_CASES_PER_SESSION]
        try:
            result, rtt_address = _run_single_session(
                project_root,
                batch,
                serial_no=serial_no,
                chip_name=chip_name,
                speed_khz=speed_khz,
                requested_counter_groups=requested_counter_groups,
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

    merged_result = SessionResult(cases=tuple(all_cases), protocol_trace=tuple(all_trace), session_complete_cases=session_complete_cases)

    memory_report_path = generate_benchmark_server_memory_report(build_dir=build_dir)
    memory_report = json.loads(memory_report_path.read_text())
    kernel_catalog = json.loads((project_root / "cmake" / "perf_stream" / "kernel_catalog.json").read_text())
    host_log = (
        f"hardware session_id={sid}\n"
        f"chip={chip_name} serial={serial_no} speed_khz={speed_khz}\n"
        f"rtt_address=0x{rtt_address:08x}\n"
        f"requested_counter_groups={requested_counter_groups}\n"
        f"batch_count={batch_count} max_cases_per_session={MAX_CASES_PER_SESSION}\n"
        f"protocol_trace_len={len(merged_result.protocol_trace)}\n"
        f"case_ids={[b.case_id for b in case_bundles]}\n"
    )
    target_log = "real Apollo510 benchmark server over SEGGER RTT (multi-batch)\n"
    bundle_root = write_result_bundle(
        merged_result,
        session_id=sid,
        output_root=project_root,
        memory_report=memory_report,
        kernel_catalog=kernel_catalog,
        target_info={"board": "apollo510_evb", "cpu": "cortex-m55", "transport": "jlink-rtt"},
        host_log_text=host_log,
        target_log_text=target_log,
    )
    return merged_result, bundle_root


def run_apollo510_stream_session(
    project_root: Path,
    *,
    serial_no: int,
    chip_name: str = "AP510NFA-CBR",
    speed_khz: int = 4000,
    requested_counter_groups: tuple[str, ...] = ("cpu", "memory", "mve"),
    session_id: str | None = None,
    build_dir: Path | None = None,
) -> tuple[SessionResult, Path]:
    abs_bundle = load_case_bundle(build_abs_s8_case_bundle(project_root, case_id="abs_hw_live").manifest_path)
    conv_bundle = load_case_bundle(build_convolve_s8_case_bundle(project_root, case_id="conv_hw_live").manifest_path)
    return _run_case_bundles_on_apollo510(
        project_root,
        [abs_bundle, conv_bundle],
        serial_no=serial_no,
        chip_name=chip_name,
        speed_khz=speed_khz,
        requested_counter_groups=requested_counter_groups,
        session_id=session_id,
        build_dir=build_dir,
        session_id_prefix="apollo510-live",
    )


def build_generated_test_case_bundles(
    project_root: Path,
    *,
    cpu: str = "cortex-m55",
    family: str | None = "ConvolutionFunctions",
    name_filter: str | None = None,
    limit: int | None = None,
) -> tuple[list[CaseBundle], list[tuple[GeneratedTestCase, str]]]:
    """Discover generated (`helia_core_tester generate`) kernel tests and bridge the
    ones with real perf-stream firmware dispatch support into CaseBundles.

    `family=None` bridges every family with at least one registered builder (see
    `generated_test_bridge.bridged_families()`) instead of a single hardcoded family --
    i.e. runs the complete set of hardware-supported kernels across all families.
    `limit`, if given, is applied per-family (not globally) when `family=None`.

    Returns (bridged_case_bundles, [(skipped_test, reason), ...]).
    """
    families = bridged_families() if family is None else [family]
    bundles: list[CaseBundle] = []
    skipped: list[tuple[GeneratedTestCase, str]] = []
    for fam in families:
        discovered = discover_generated_tests(project_root, cpu=cpu, family=fam, name_filter=name_filter, limit=limit)
        for test in discovered:
            try:
                bundles.append(build_case_bundle_from_generated_test(project_root, test))
            except UnsupportedGeneratedTestError as exc:
                skipped.append((test, str(exc)))
    return bundles, skipped


def run_apollo510_generated_test_session(
    project_root: Path,
    *,
    serial_no: int,
    chip_name: str = "AP510NFA-CBR",
    speed_khz: int = 4000,
    requested_counter_groups: tuple[str, ...] = ("cpu", "memory", "mve"),
    session_id: str | None = None,
    build_dir: Path | None = None,
    cpu: str = "cortex-m55",
    family: str | None = "ConvolutionFunctions",
    name_filter: str | None = None,
    limit: int | None = None,
    on_case_complete: Callable[[CaseRunResult], None] | None = None,
) -> tuple[SessionResult, Path, list[tuple[GeneratedTestCase, str]]]:
    """Run real `helia_core_tester generate`-produced kernel tests (with their real golden
    data) against connected Apollo510 hardware over the streaming HCTP/RTT session,
    instead of the hand-authored abs/convolve demo cases.

    `family=None` bridges every family with real firmware dispatch support (see
    `build_generated_test_case_bundles`), i.e. runs the complete hardware-supported suite
    in one session (transparently batched).

    Transparently splits the discovered/bridged cases into batches of at most
    MAX_CASES_PER_SESSION (matching firmware HCT_SERVER_MAX_CASES) and runs one
    fresh reset-on-open RTT session per batch, merging all cases into a single
    SessionResult/result bundle -- sending more cases than that in one LOAD_PLAN
    causes the firmware to silently drop the plan and hang the host.
    """
    bundles, skipped = build_generated_test_case_bundles(
        project_root, cpu=cpu, family=family, name_filter=name_filter, limit=limit
    )
    if not bundles:
        raise RuntimeError(
            f"No bridgeable generated tests found for cpu={cpu} family={family if family is not None else '<all bridged families>'} "
            f"name_filter={name_filter!r} (skipped {len(skipped)}); run `helia_core_tester generate` first."
        )
    result, bundle_root = _run_case_bundles_in_batches(
        project_root,
        bundles,
        serial_no=serial_no,
        chip_name=chip_name,
        speed_khz=speed_khz,
        requested_counter_groups=requested_counter_groups,
        session_id=session_id,
        build_dir=build_dir,
        session_id_prefix="apollo510-generated-tests",
        on_case_complete=on_case_complete,
    )
    return result, bundle_root, skipped

