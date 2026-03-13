"""FVP process execution and parallel fail-fast controls."""

from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from helia_core_tester.reporting.models import TestResult, TestStatus
from helia_core_tester.reporting.parser import TestResultParser

from .coverage import CoverageContext, process_coverage_output
from .env import REPO_ROOT


@dataclass
class ProcessRecord:
    elf: Path
    cpu: str
    descriptor_name: str
    popen: subprocess.Popen
    start_time: float
    state: str = "running"


class ProcessSupervisor:
    def __init__(self, grace_seconds: float = 2.0, verbosity: int = 0):
        self.grace_seconds = grace_seconds
        self.verbosity = verbosity
        self._active: dict[int, ProcessRecord] = {}
        self._lock = threading.Lock()

    def register(self, record: ProcessRecord) -> None:
        with self._lock:
            self._active[record.popen.pid] = record

    def unregister(self, pid: int) -> None:
        with self._lock:
            self._active.pop(pid, None)

    def snapshot_active(self) -> list[ProcessRecord]:
        with self._lock:
            return list(self._active.values())

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def terminate_all(self, reason: str) -> None:
        records = self.snapshot_active()
        if not records:
            return

        if self.verbosity >= 1:
            print(f"Fail-fast cleanup: terminating {len(records)} active process(es) ({reason})")

        for record in records:
            record.state = f"terminating:{reason}"
            _signal_process_group(record.popen, signal.SIGTERM)

        deadline = time.time() + self.grace_seconds
        while time.time() < deadline:
            remaining = [record for record in records if record.popen.poll() is None]
            if not remaining:
                break
            time.sleep(0.05)

        remaining = [record for record in records if record.popen.poll() is None]
        for record in remaining:
            record.state = f"killing:{reason}"
            _signal_process_group(record.popen, signal.SIGKILL)

        for record in records:
            try:
                record.popen.communicate(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass


@dataclass
class FvpProcessResult:
    exit_code: int
    output: str
    duration: float
    timed_out: bool = False
    launch_error: Optional[str] = None


def _signal_process_group(proc: subprocess.Popen, sig: int) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass
    except Exception:
        pass


def _terminate_process(proc: subprocess.Popen, grace_seconds: float = 1.0) -> None:
    _signal_process_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process_group(proc, signal.SIGKILL)
    try:
        proc.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        pass


def _build_fvp_cmd(fvp_exe: Path, elf: Path, extra_args: List[str]) -> List[str]:
    return [
        str(fvp_exe),
        "-C", "mps3_board.uart0.shutdown_on_eot=1",
        "-C", "mps3_board.visualisation.disable-visualisation=1",
        "-C", "mps3_board.telnetterminal0.start_telnet=0",
        "-C", "mps3_board.uart0.out_file=-",
        "-C", "mps3_board.uart0.unbuffered_output=1",
    ] + extra_args + [str(elf)]


def run_fvp_process(
    fvp_exe: Path,
    elf: Path,
    timeout: float,
    verbosity: int,
    extra_args: List[str],
    env: dict,
    cpu: str,
    descriptor_name: Optional[str],
    supervisor: Optional[ProcessSupervisor] = None,
    stop_event: Optional[threading.Event] = None,
) -> FvpProcessResult:
    if stop_event is not None and stop_event.is_set():
        return FvpProcessResult(exit_code=-1, output="", duration=0.0, launch_error="cancelled")

    cmd = _build_fvp_cmd(fvp_exe=fvp_exe, elf=elf, extra_args=extra_args)
    if verbosity >= 2:
        print(f"Run: {' '.join(cmd)}\n")

    start_time = time.time()
    proc: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=1,
        )
        pid = proc.pid
        if supervisor is not None:
            supervisor.register(
                ProcessRecord(
                    elf=elf,
                    cpu=cpu,
                    descriptor_name=descriptor_name or elf.stem,
                    popen=proc,
                    start_time=start_time,
                )
            )

        try:
            stdout, _ = proc.communicate(timeout=None if timeout <= 0 else timeout)
        except subprocess.TimeoutExpired:
            _terminate_process(proc)
            stdout, _ = proc.communicate()
            return FvpProcessResult(
                exit_code=124,
                output=stdout or "",
                duration=time.time() - start_time,
                timed_out=True,
            )

        return FvpProcessResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            output=stdout or "",
            duration=time.time() - start_time,
        )
    except Exception as exc:
        return FvpProcessResult(
            exit_code=-1,
            output="",
            duration=time.time() - start_time,
            launch_error=str(exc),
        )
    finally:
        if supervisor is not None and pid is not None:
            supervisor.unregister(pid)


def run_fvp_with_reporting(
    fvp_exe: Path,
    elf: Path,
    timeout: float,
    verbosity: int,
    extra_args: List[str],
    env: dict,
    cpu: str,
    descriptor_name: Optional[str] = None,
    coverage_ctx: Optional[CoverageContext] = None,
    supervisor: Optional[ProcessSupervisor] = None,
    stop_event: Optional[threading.Event] = None,
    emit_output: bool = True,
) -> TestResult:
    parser = TestResultParser()

    process_result = run_fvp_process(
        fvp_exe=fvp_exe,
        elf=elf,
        timeout=timeout,
        verbosity=verbosity,
        extra_args=extra_args,
        env=env,
        cpu=cpu,
        descriptor_name=descriptor_name,
        supervisor=supervisor,
        stop_event=stop_event,
    )

    if process_result.launch_error == "cancelled":
        return TestResult(
            test_name=elf.stem,
            status=TestStatus.SKIP,
            duration=0.0,
            cpu=cpu,
            elf_path=elf,
            skip_reason="skipped due to fail-fast cancellation",
            timestamp=datetime.now(),
            descriptor_name=descriptor_name or elf.stem,
        )

    if process_result.timed_out:
        if emit_output:
            print(f"TIMEOUT running {elf}", file=sys.stderr)
        return TestResult(
            test_name=elf.stem,
            status=TestStatus.TIMEOUT,
            duration=process_result.duration,
            cpu=cpu,
            elf_path=elf,
            failure_reason="Test execution timed out",
            timestamp=datetime.now(),
            exit_code=124,
            error_type="timeout",
            descriptor_name=descriptor_name or elf.stem,
        )

    if process_result.launch_error is not None:
        if emit_output:
            print(f"ERROR running {elf}: {process_result.launch_error}", file=sys.stderr)
        return TestResult(
            test_name=elf.stem,
            status=TestStatus.ERROR,
            duration=process_result.duration,
            cpu=cpu,
            elf_path=elf,
            failure_reason=f"Execution error: {process_result.launch_error}",
            timestamp=datetime.now(),
            exit_code=-1,
            error_type="crash",
            descriptor_name=descriptor_name or elf.stem,
        )

    output = process_coverage_output(process_result.output, elf, coverage_ctx, env, verbosity)
    result = parser.parse_fvp_output(
        output,
        elf,
        cpu,
        process_result.duration,
        process_result.exit_code,
        descriptor_name=descriptor_name,
    )
    if emit_output:
        _emit_reporting_result_output(result=result, elf=elf, verbosity=verbosity, raw_output=output)
    return result


def _emit_reporting_result_output(result: TestResult, elf: Path, verbosity: int, raw_output: str) -> None:
    if result.status == TestStatus.PASS:
        print(f"\n\nPASS: {elf}")
        if verbosity >= 3:
            sys.stdout.write(raw_output)
            sys.stdout.flush()
        return

    print(f"FAIL: {elf}")
    if verbosity < 1:
        return

    print("=" * 60)
    print("FAILURE DETAILS:")
    print("=" * 60)

    if result.failure_reason:
        print(f"Reason: {result.failure_reason}")
        print()

    if verbosity >= 2:
        if result.expected_output or result.actual_output:
            print("OUTPUT COMPARISON:")
            if result.expected_output:
                print(f"  Expected (Golden): {result.expected_output}")
            if result.actual_output:
                print(f"  Actual (Got):     {result.actual_output}")
            print()

        if result.output_differences:
            print("DETAILED DIFFERENCES:")
            max_diffs = 20 if verbosity < 3 else len(result.output_differences)
            for diff in result.output_differences[:max_diffs]:
                print(f"  {diff}")
            if len(result.output_differences) > max_diffs:
                print(f"  ... ({len(result.output_differences) - max_diffs} more differences)")
            print()

    max_lines = 20 if verbosity < 3 else len(result.output_lines)
    if result.output_lines:
        print("TEST OUTPUT:")
        for line in result.output_lines[:max_lines]:
            print(f"  {line}")
        if len(result.output_lines) > max_lines and verbosity < 3:
            print("  ... (truncated)")
        print()

    print("=" * 60)


def _resolve_run_jobs(run_jobs: int, total_elves: int) -> int:
    if total_elves <= 1:
        return 1
    if run_jobs == 0:
        run_jobs = os.cpu_count() or 4
    return max(1, min(run_jobs, total_elves))


def run_elf_jobs_with_reporting(
    elf_entries: List[Tuple[Path, str]],
    fvp_exe: Path,
    timeout: float,
    verbosity: int,
    extra_args: List[str],
    env: dict,
    cpu: str,
    coverage_ctx: Optional[CoverageContext],
    run_jobs: int,
    fail_fast: bool,
    supervisor: Optional[ProcessSupervisor],
) -> Tuple[List[TestResult], bool]:
    ordered_entries = sorted(elf_entries, key=lambda item: item[0].name)
    resolved_jobs = _resolve_run_jobs(run_jobs, len(ordered_entries))
    any_fail = False

    if resolved_jobs <= 1:
        results: List[TestResult] = []
        for elf, descriptor_name in ordered_entries:
            result = run_fvp_with_reporting(
                fvp_exe=fvp_exe,
                elf=elf,
                timeout=timeout,
                verbosity=verbosity,
                extra_args=extra_args,
                env=env,
                cpu=cpu,
                descriptor_name=descriptor_name,
                coverage_ctx=coverage_ctx,
                supervisor=supervisor,
                stop_event=None,
                emit_output=True,
            )
            results.append(result)
            if result.status not in (TestStatus.PASS, TestStatus.SKIP):
                any_fail = True
                if fail_fast:
                    break
        return results, any_fail

    stop_event = threading.Event()
    results_by_index: dict[int, TestResult] = {}

    with ThreadPoolExecutor(max_workers=resolved_jobs) as executor:
        future_to_index = {
            executor.submit(
                run_fvp_with_reporting,
                fvp_exe,
                elf,
                timeout,
                verbosity,
                extra_args,
                env,
                cpu,
                descriptor_name,
                coverage_ctx,
                supervisor,
                stop_event,
                False,
            ): idx
            for idx, (elf, descriptor_name) in enumerate(ordered_entries)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            elf, descriptor_name = ordered_entries[idx]
            try:
                result = future.result()
            except CancelledError:
                continue
            except Exception as exc:
                result = TestResult(
                    test_name=elf.stem,
                    status=TestStatus.ERROR,
                    duration=0.0,
                    cpu=cpu,
                    elf_path=elf,
                    failure_reason=f"Execution error: {exc}",
                    timestamp=datetime.now(),
                    exit_code=-1,
                    error_type="crash",
                    descriptor_name=descriptor_name,
                )

            results_by_index[idx] = result
            if result.status not in (TestStatus.PASS, TestStatus.SKIP):
                any_fail = True
                if fail_fast and not stop_event.is_set():
                    stop_event.set()
                    if supervisor is not None:
                        supervisor.terminate_all("fail_fast")
                    for pending in future_to_index:
                        if pending is not future:
                            pending.cancel()

    ordered_results: List[TestResult] = []
    for idx, (elf, _) in enumerate(ordered_entries):
        result = results_by_index.get(idx)
        if result is None:
            continue
        _emit_reporting_result_output(
            result=result,
            elf=elf,
            verbosity=verbosity,
            raw_output="\n".join(result.output_lines),
        )
        ordered_results.append(result)

    return ordered_results, any_fail
