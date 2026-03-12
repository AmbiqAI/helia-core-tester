#!/usr/bin/env python3
"""
build_and_run_fvp.py — Python replacement for CMSIS-NN UnitTest build+run on FVP Corstone-300.

Examples:
  # vanilla (downloads in ./artifacts/downloads, GCC toolchain from downloads, FVP from downloads)
  python3 python_scripts/build_and_run_fvp.py

  # multiple CPUs, less optimization, quiet logs
  python3 python_scripts/build_and_run_fvp.py -c m0,m55 -o "-O2" -q

  # skip downloads/setup (assume paths present), override paths, pass extra CMake defs
  python3 python_scripts/build_and_run_fvp.py -e -u ./artifacts/downloads/ethos-u-core-platform -C ./artifacts/downloads/CMSIS_5 \
    -D CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=ON

  # use Arm Compiler, custom generator, increased timeouts
  python3 python_scripts/build_and_run_fvp.py -a --generator Ninja --timeout-run 180

Notes:
- This script mirrors flags of ns-cmsis-nn/Tests/UnitTest/build_and_run_tests.sh where sensible.
- It expects Linux (same as the bash script).
"""

from __future__ import annotations
import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Import reporting and discovery (package imports only)
from helia_core_tester.core.discovery import (
    find_repo_root,
    find_setup_dependencies_script,
    find_descriptors_dir,
)
from helia_core_tester.core.path_layout import (
    build_dir as canonical_build_dir,
    coverage_report_dir as canonical_coverage_report_dir,
    generated_tests_dir as canonical_generated_tests_dir,
    tests_report_dir as canonical_tests_report_dir,
)
from helia_core_tester.core.cpu_targets import parse_cpu_list
from helia_core_tester.reporting.models import TestResult, TestStatus
from helia_core_tester.reporting.parser import TestResultParser

repo_root = find_repo_root()
ARTIFACTS_DIR = repo_root / "artifacts"
DEFAULT_DL = ARTIFACTS_DIR / "downloads"
DEFAULT_SOURCE = repo_root

FVP_EXE_NAME = "FVP_Corstone_SSE-300_Ethos-U55"
FVP_DIR_X86 = "Linux64_GCC-9.3"
FVP_DIR_AARCH64 = "Linux64_armv8l_GCC-9.3"
GCOV_BEGIN_MARKER = "@@GCOV_BEGIN@@"
GCOV_END_MARKER = "@@GCOV_END@@"
GCOV_BLOCK_PATTERN = re.compile(
    rf"{re.escape(GCOV_BEGIN_MARKER)}(?P<payload>.*?){re.escape(GCOV_END_MARKER)}",
    re.DOTALL,
)


@dataclass
class CoverageContext:
    enabled: bool
    build_dir: Path
    streams_dir: Path
    gcov_tool: Optional[str]
    stream_index: int = 0
    merged_streams: int = 0
    merge_errors: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


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

        for rec in records:
            rec.state = f"terminating:{reason}"
            _signal_process_group(rec.popen, signal.SIGTERM)

        deadline = time.time() + self.grace_seconds
        while time.time() < deadline:
            remaining = [rec for rec in records if rec.popen.poll() is None]
            if not remaining:
                break
            time.sleep(0.05)

        remaining = [rec for rec in records if rec.popen.poll() is None]
        for rec in remaining:
            rec.state = f"killing:{reason}"
            _signal_process_group(rec.popen, signal.SIGKILL)

        for rec in records:
            try:
                rec.popen.communicate(timeout=0.2)
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


class FvpScriptError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def _which_in_env(name: str, env: dict) -> Optional[str]:
    return shutil.which(name, path=env.get("PATH"))


def _resolve_gcov_tool(env: dict) -> Optional[str]:
    for tool in ("arm-none-eabi-gcov-tool", "gcov-tool"):
        resolved = _which_in_env(tool, env)
        if resolved:
            return resolved
    return None


def _resolve_gcov_executable(env: dict) -> Optional[str]:
    for exe in ("arm-none-eabi-gcov", "gcov"):
        resolved = _which_in_env(exe, env)
        if resolved:
            return resolved
    return None


def _tests_report_dir(cpu: str) -> Path:
    return canonical_tests_report_dir(repo_root, cpu)


def _coverage_report_dir(cpu: str) -> Path:
    return canonical_coverage_report_dir(repo_root, cpu)


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


def _run_fvp_process(
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
        print(f"Run: {' '.join(cmd)}")

    start_time = time.time()
    proc: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
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
    except Exception as e:
        return FvpProcessResult(
            exit_code=-1,
            output="",
            duration=time.time() - start_time,
            launch_error=str(e),
        )
    finally:
        if supervisor is not None and pid is not None:
            supervisor.unregister(pid)


def _extract_coverage_blocks(output: str) -> Tuple[List[str], str]:
    blocks: List[str] = []

    def _replace(match: re.Match) -> str:
        payload = match.group("payload")
        hex_payload = re.sub(r"[^0-9A-Fa-f]", "", payload)
        if hex_payload:
            blocks.append(hex_payload)
        return ""

    cleaned_output = GCOV_BLOCK_PATTERN.sub(_replace, output)
    return blocks, cleaned_output


def _new_coverage_context(build_dir: Path, gcov_tool: Optional[str]) -> CoverageContext:
    streams_dir = build_dir / "coverage" / "streams"
    streams_dir.mkdir(parents=True, exist_ok=True)
    return CoverageContext(
        enabled=True,
        build_dir=build_dir,
        streams_dir=streams_dir,
        gcov_tool=gcov_tool,
    )


def _merge_coverage_stream(ctx: CoverageContext, payload: bytes, env: dict, verbosity: int) -> bool:
    if not ctx.gcov_tool:
        return False
    try:
        subprocess.run(
            [ctx.gcov_tool, "merge-stream"],
            input=payload,
            cwd=str(ctx.build_dir),
            env=env,
            check=True,
            stdout=subprocess.DEVNULL if verbosity <= 2 else None,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        if verbosity >= 1:
            err = (e.stderr or b"").decode(errors="replace").strip()
            print(f"Coverage merge failed: {ctx.gcov_tool} merge-stream ({err})", file=sys.stderr)
        return False


def _process_coverage_output(
    output: str,
    elf: Path,
    ctx: Optional[CoverageContext],
    env: dict,
    verbosity: int,
) -> str:
    if not ctx or not ctx.enabled:
        return output

    blocks, cleaned_output = _extract_coverage_blocks(output)
    if not blocks:
        if verbosity >= 2:
            print(f"Coverage: no stream block found in {elf.name}")
        return cleaned_output

    with ctx.lock:
        for block in blocks:
            if len(block) % 2 != 0:
                if verbosity >= 1:
                    print(f"WARNING: skipping malformed coverage payload from {elf}", file=sys.stderr)
                ctx.merge_errors += 1
                continue

            payload = bytes.fromhex(block)
            ctx.stream_index += 1
            stream_path = ctx.streams_dir / f"{elf.stem}_{ctx.stream_index:04d}.gcovstream"
            stream_path.write_bytes(payload)

            if _merge_coverage_stream(ctx, payload, env, verbosity):
                ctx.merged_streams += 1
            else:
                ctx.merge_errors += 1

    if verbosity >= 2:
        print(
            f"Coverage: {elf.name} blocks={len(blocks)} merged={ctx.merged_streams} errors={ctx.merge_errors}"
        )
    return cleaned_output


def _read_cmake_cache_path(build_dir: Path, key: str) -> Optional[Path]:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return None
    prefix = f"{key}:PATH="
    for line in cache.read_text(errors="ignore").splitlines():
        if line.startswith(prefix):
            return Path(line[len(prefix):]).resolve()
    return None


def _generate_coverage_reports(
    cpus: List[str],
    args,
    env: dict,
    source_dir: Path,
    compiler_tag: str,
    verbosity: int,
) -> None:
    if not getattr(args, "coverage", False):
        return

    gcovr = _which_in_env("gcovr", env)
    gcov_exe = _resolve_gcov_executable(env)

    if not gcovr:
        note = ARTIFACTS_DIR / "README.txt"
        note.write_text(
            "gcovr is not installed; .gcda files were merged in build directories.\n"
            "Install gcovr to generate HTML/JSON coverage summaries.\n"
        )
        if verbosity >= 1:
            print(f"Coverage: gcovr not found; wrote {note}")
        return

    for cpu in cpus:
        build_dir = canonical_build_dir(repo_root, cpu, compiler_tag)
        if not build_dir.exists():
            continue

        cpu_dir = _coverage_report_dir(cpu)
        cpu_dir.mkdir(parents=True, exist_ok=True)

        default_cmsis_nn_root = (source_dir / ".." / "..").resolve()
        cmsis_nn_root = _read_cmake_cache_path(build_dir, "CMSIS_NN_ROOT")
        if cmsis_nn_root is None:
            cmsis_nn_root = default_cmsis_nn_root
        elif not cmsis_nn_root.exists():
            if verbosity >= 2:
                print(
                    "Coverage: CMSIS_NN_ROOT from CMake cache does not exist "
                    f"({cmsis_nn_root}); falling back to {default_cmsis_nn_root}"
                )
            cmsis_nn_root = default_cmsis_nn_root

        # Keep this filter mount/path agnostic: gcov source paths may be absolute
        # and differ from host mounts (e.g. /workspaces vs /Users).
        source_filter = r"(^|.*/)Source/.*"

        cmd = [
            gcovr,
            "--root",
            str(cmsis_nn_root),
            "--filter",
            source_filter,
            "--gcov-ignore-parse-errors",
            "suspicious_hits.warn_once_per_file",
            "--object-directory",
            str(build_dir),
            "--txt-summary",
            "--json-summary",
            str(cpu_dir / "summary.json"),
            "--json-summary-pretty",
            "--html-details",
            str(cpu_dir / "index.html"),
            "--lcov",
            str(cpu_dir / "coverage.info"),
            str(build_dir),
        ]
        if gcov_exe:
            cmd.extend(["--gcov-executable", gcov_exe])

        result = subprocess.run(
            cmd,
            cwd=str(build_dir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        (cpu_dir / "summary.txt").write_text((result.stdout or "") + (result.stderr or ""))
        if result.returncode != 0 and verbosity >= 1:
            print(
                f"WARNING: coverage report generation failed for {cpu} (rc={result.returncode})",
                file=sys.stderr,
            )
        elif verbosity >= 1:
            print(f"Coverage report generated: {cpu_dir / 'index.html'}")


def die(msg: str, code: int = 2):
    raise FvpScriptError(msg, exit_code=code)


def is_linux() -> bool:
    return platform.system().lower() == "linux"


def arch_tag() -> str:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    die(f"Unsupported architecture: {m}")


def ensure_exe_on_path(name: str) -> Optional[str]:
    return shutil.which(name)


def call_setup_dependencies(downloads_dir: Path) -> None:
    setup = find_setup_dependencies_script(repo_root)
    if not setup or not setup.exists():
        print("No setup_dependencies.py found; skipping dependency setup.")
        return
    print("Ensuring dependencies via setup_dependencies.py")
    rc = subprocess.call(
        [sys.executable, str(setup), "--downloads-dir", str(downloads_dir)],
        cwd=str(repo_root),
    )
    if rc != 0:
        die(f"Dependency setup failed (rc={rc})")


def prepend_path(p: Path, env: dict) -> None:
    env["PATH"] = str(p) + os.pathsep + env.get("PATH", "")


def _fvp_model_dirs_for_arch(arch: str) -> list[str]:
    if arch == "x86_64":
        return [FVP_DIR_X86, FVP_DIR_AARCH64]
    if arch == "aarch64":
        return [FVP_DIR_AARCH64, FVP_DIR_X86]
    return [FVP_DIR_X86, FVP_DIR_AARCH64]


def _resolve_downloaded_fvp_executable(downloads_dir: Path, arch: str) -> tuple[Optional[Path], list[Path]]:
    models_root = downloads_dir / "corstone300_download" / "models"
    checked: list[Path] = []
    for model_dir in _fvp_model_dirs_for_arch(arch):
        candidate = models_root / model_dir / FVP_EXE_NAME
        checked.append(candidate)
        if candidate.exists():
            return candidate, checked
    return None, checked


def detect_paths(args) -> dict:
    # base
    env = os.environ.copy()
    arch = arch_tag()

    # downloads root
    dl = args.downloads_dir.resolve()

    # ethos-u core platform
    ethos = Path(args.ethos_path).resolve() if args.ethos_path else (dl / "ethos-u-core-platform")
    if not ethos.exists():
        die(f"Ethos-U core platform not found: {ethos}. Run without -e or point -u to a valid path.")

    # cmsis5
    cmsis5 = Path(args.cmsis5_path).resolve() if args.cmsis5_path else (dl / "CMSIS_5")
    if not cmsis5.exists():
        die(f"CMSIS_5 not found: {cmsis5}. Run without -e or point -C to a valid path.")

    # toolchain file & compiler tag
    if args.use_arm_compiler:
        toolchain_file = ethos / "cmake" / "toolchain" / "armclang.cmake"
        compiler_tag = "arm-compiler"
    else:
        toolchain_file = ethos / "cmake" / "toolchain" / "arm-none-eabi-gcc.cmake"
        compiler_tag = "gcc"
        if not args.no_gcc_from_download:
            gcc_bin = dl / "arm_gcc_download" / "bin"
            if not gcc_bin.exists():
                die(f"GCC toolchain not found at {gcc_bin}. Run without -e or install gcc on PATH.")
            prepend_path(gcc_bin, env)

    if not toolchain_file.exists():
        die(f"Toolchain file missing: {toolchain_file}")

    # FVP
    fvp_exe: Optional[Path] = None
    if not args.no_fvp_from_download:
        fvp_exe_candidate, checked_paths = _resolve_downloaded_fvp_executable(dl, arch)
        if fvp_exe_candidate is None:
            checked = ", ".join(str(p) for p in checked_paths)
            die(
                f"FVP not found in downloads (checked: {checked}). "
                "Run with -f to use a system FVP on PATH."
            )
        prepend_path(fvp_exe_candidate.parent, env)
        fvp_exe = fvp_exe_candidate
    else:
        from_path = ensure_exe_on_path(FVP_EXE_NAME)
        if not from_path:
            die(f"{FVP_EXE_NAME} not on PATH (use downloads or add it).")
        fvp_exe = Path(from_path)

    return {
        "env": env,
        "dl": dl,
        "ethos": ethos,
        "cmsis5": cmsis5,
        "toolchain_file": toolchain_file,
        "compiler_tag": compiler_tag,
        "fvp_exe": fvp_exe,
    }


def _get_git_sha(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True
        ).strip()
    except Exception:
        return None


def cmake_configure(source_dir: Path, build_dir: Path, toolchain_file: Path, cpu: str,
                    cmsis5: Path, optimization: str, extra_defs: List[str], generator: Optional[str],
                    generated_tests_dir: Optional[Path], enable_coverage: bool,
                    verbosity: int, env: dict) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    
    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        cmake_cache.unlink()
    
    cmd = [
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_dir),
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
        f"-DTARGET_CPU={cpu}",
        f"-DCMSIS_PATH={cmsis5}",
        f"-DCMSIS_OPTIMIZATION_LEVEL={optimization}",
    ] + [f"-D{d}" for d in extra_defs]
    if generated_tests_dir is not None:
        cmd.append(f"-DGENERATED_TESTS_DIR={generated_tests_dir}")
    if enable_coverage:
        cmd.append("-DENABLE_COVERAGE=ON")
    
    if generator:
        cmd += ["-G", generator]
    if verbosity >= 2:
        print(f"Configure: {' '.join(cmd)}")
    stdout = subprocess.DEVNULL if verbosity <= 1 else None
    rc = subprocess.call(cmd, cwd=str(repo_root), env=env, stdout=stdout, stderr=None)
    if rc != 0:
        die(f"CMake configure failed for {cpu} (rc={rc})")


def cmake_build(build_dir: Path, verbosity: int, env: dict, jobs: Optional[int]) -> None:
    cmd = ["cmake", "--build", str(build_dir)]
    if jobs and jobs > 0:
        # Pass through to the underlying build tool (works for Make/Ninja)
        cmd += ["--", f"-j{jobs}"]
    if verbosity >= 2:
        print(f"Build: {' '.join(cmd)}")
    stdout = subprocess.DEVNULL if verbosity <= 1 else None
    rc = subprocess.call(cmd, cwd=str(repo_root), env=env, stdout=stdout, stderr=None)
    if rc != 0:
        die(f"CMake build failed (rc={rc})")


def find_elves(build_dir: Path) -> List[Path]:
    # First try to find ELF files in a 'tests' subdirectory
    tests_dir = build_dir / "tests"
    if tests_dir.exists():
        return [p for p in tests_dir.rglob("*.elf") if p.is_file()]
    # Fallback to searching the entire build directory
    return [p for p in build_dir.rglob("*.elf") if p.is_file()]


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
    """
    Run FVP with comprehensive result reporting.
    
    Returns:
        TestResult object with detailed test information
    """
    parser = TestResultParser()

    process_result = _run_fvp_process(
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
            descriptor_name=descriptor_name or elf.stem
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
            descriptor_name=descriptor_name or elf.stem
        )

    output = _process_coverage_output(process_result.output, elf, coverage_ctx, env, verbosity)
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
        print(f"PASS: {elf}")
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


def run_fvp(
    fvp_exe: Path,
    elf: Path,
    timeout: float,
    verbosity: int,
    extra_args: List[str],
    env: dict,
    coverage_ctx: Optional[CoverageContext] = None,
    supervisor: Optional[ProcessSupervisor] = None,
    stop_event: Optional[threading.Event] = None,
    emit_output: bool = True,
) -> bool:
    process_result = _run_fvp_process(
        fvp_exe=fvp_exe,
        elf=elf,
        timeout=timeout,
        verbosity=verbosity,
        extra_args=extra_args,
        env=env,
        cpu="unknown",
        descriptor_name=elf.stem,
        supervisor=supervisor,
        stop_event=stop_event,
    )
    if process_result.launch_error == "cancelled":
        return False

    if process_result.timed_out:
        if emit_output:
            print(f"TIMEOUT running {elf}", file=sys.stderr)
        return False

    if process_result.launch_error is not None:
        if emit_output:
            print(f"ERROR running {elf}: {process_result.launch_error}", file=sys.stderr)
        return False

    out = _process_coverage_output(process_result.output, elf, coverage_ctx, env, verbosity)
    zero_failures_pattern = re.compile(r'^0\s+Failures\s*$', re.MULTILINE | re.IGNORECASE)
    success = bool(zero_failures_pattern.search(out))

    if not emit_output:
        return success

    if not success:
        print(f"FAIL: {elf}")
        if verbosity >= 1:
            print("=" * 60)
            print("FAILURE DETAILS:")
            print("=" * 60)
        lines = out.split('\n')
        failure_lines = []
        in_failure_section = False

        for line in lines:
            if any(keyword in line.lower() for keyword in ['fail', 'error', 'assert', 'test']):
                in_failure_section = True
                failure_lines.append(line)
            elif in_failure_section and line.strip():
                failure_lines.append(line)
            elif in_failure_section and not line.strip():
                failure_lines.append(line)
            elif in_failure_section and len(failure_lines) > 20:
                failure_lines.append("... (truncated)")
                break

        if verbosity >= 1:
            max_lines = 20 if verbosity < 3 else len(failure_lines)
            if failure_lines:
                for line in failure_lines[:max_lines]:
                    print(line)
                if len(failure_lines) > max_lines and verbosity < 3:
                    print("... (truncated)")
            else:
                if verbosity >= 2:
                    print("Last 20 lines of output:")
                    for line in lines[-20:]:
                        print(line)
            print("=" * 60)
    elif verbosity >= 1:
        sys.stdout.write(out)
        sys.stdout.flush()

    return success


def parse_cpus(cpu_str: str) -> List[str]:
    return parse_cpu_list(cpu_str)


def _resolve_run_jobs(run_jobs: int, total_elves: int) -> int:
    if total_elves <= 1:
        return 1
    if run_jobs == 0:
        run_jobs = os.cpu_count() or 4
    return max(1, min(run_jobs, total_elves))


def _run_elf_jobs_with_reporting(
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
            except Exception as e:
                result = TestResult(
                    test_name=elf.stem,
                    status=TestStatus.ERROR,
                    duration=0.0,
                    cpu=cpu,
                    elf_path=elf,
                    failure_reason=f"Execution error: {e}",
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


def resolve_generated_tests_dir(source_dir: Path, cpu: str) -> Path:
    return canonical_generated_tests_dir(source_dir, cpu)




def run_tests_with_reporting(cpus: List[str], 
                           source_dir: Path,
                           toolchain_file: Path,
                           cmsis5: Path,
                           fvp_exe: Path,
                           compiler_tag: str,
                           args,
                           env: dict,
                           supervisor: Optional[ProcessSupervisor] = None) -> Tuple[List[TestResult], bool]:
    """
    Run tests with comprehensive reporting.
    
    Returns:
        Tuple of (list of TestResult objects, overall success)
    """
    from helia_core_tester.reporting.generator import ReportGenerator
    from helia_core_tester.reporting.models import TestReport, TestStatus, DescriptorResult
    from helia_core_tester.reporting.descriptor_tracker import DescriptorTracker
    
    all_results: List[TestResult] = []
    any_fail = False
    verbosity = getattr(args, 'verbosity', 0)
    descriptors_dir = find_descriptors_dir()
    tracker = DescriptorTracker(descriptors_dir)
    all_descriptors_dict = tracker.load_all_descriptors()

    for cpu in cpus:
        cpu_start_time = datetime.now()
        if verbosity >= 1:
            print(f"\nTarget: {cpu} (gcc)")
        build_dir = canonical_build_dir(repo_root, cpu, compiler_tag)
        cpu_generated_tests_dir = resolve_generated_tests_dir(source_dir, cpu)
        cpu_tests_report_dir = _tests_report_dir(cpu)
        coverage_ctx = _new_coverage_context(build_dir, getattr(args, "_gcov_tool", None)) if args.coverage else None

        if cpu_tests_report_dir.exists():
            if verbosity >= 1:
                print(f"Removing previous tests report directory: {cpu_tests_report_dir}")
            shutil.rmtree(cpu_tests_report_dir, ignore_errors=True)
        cpu_tests_report_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_build:
            if build_dir.exists():
                if verbosity >= 1:
                    print(f"Removing previous build directory: {build_dir}")
                shutil.rmtree(build_dir, ignore_errors=True)
            # Build first - use the env passed in
            cmake_configure(
                source_dir=source_dir,
                build_dir=build_dir,
                toolchain_file=toolchain_file,
                cpu=cpu,
                cmsis5=cmsis5,
                optimization=args.opt,
                extra_defs=args.cmake_def,
                generator=args.generator,
                generated_tests_dir=cpu_generated_tests_dir,
                enable_coverage=args.coverage,
                verbosity=verbosity,
                env=env,
            )
            cmake_build(build_dir=build_dir, verbosity=verbosity, env=env, jobs=args.jobs)

        if args.no_run:
            continue

        elves = find_elves(build_dir)
        if not elves:
            if verbosity >= 1:
                print(f"(no .elf found under {build_dir}, nothing to run)")
            continue

        elf_entries: List[Tuple[Path, str]] = []
        for elf in elves:
            test_name = elf.stem
            descriptor = tracker.map_test_to_descriptor(test_name, all_descriptors_dict)
            descriptor_name = descriptor.get('name') if descriptor else test_name
            elf_entries.append((elf, descriptor_name))

        cpu_results, cpu_failed = _run_elf_jobs_with_reporting(
            elf_entries=elf_entries,
            fvp_exe=fvp_exe,
            timeout=args.timeout_run,
            verbosity=verbosity,
            extra_args=args.fvp_arg,
            env=env,
            cpu=cpu,
            coverage_ctx=coverage_ctx,
            run_jobs=getattr(args, "run_jobs", 1),
            fail_fast=args.fail_fast,
            supervisor=supervisor,
        )
        all_results.extend(cpu_results)
        any_fail = any_fail or cpu_failed

        cpu_end_time = datetime.now()
        generator = ReportGenerator(output_dir=cpu_tests_report_dir)

        test_result_map: dict[str, TestResult] = {}
        for result in cpu_results:
            desc_name = result.descriptor_name or result.test_name
            if desc_name not in test_result_map or result.status == TestStatus.PASS:
                test_result_map[desc_name] = result

        active_descriptors: set[str] = set(test_result_map.keys())
        for desc_name in all_descriptors_dict.keys():
            tflite_file = cpu_generated_tests_dir / desc_name / f"{desc_name}.tflite"
            includes_dir = cpu_generated_tests_dir / desc_name / "includes"
            model_headers = list(includes_dir.glob(f"{desc_name}_*.h")) if includes_dir.exists() else []
            model_header_old = cpu_generated_tests_dir / desc_name / "includes" / f"{desc_name}_model.h"
            has_model_header = len(model_headers) > 0 or model_header_old.exists()
            elf_path = build_dir / "tests" / f"{desc_name}.elf"
            if tflite_file.exists() or has_model_header or elf_path.exists():
                active_descriptors.add(desc_name)

        descriptor_results: dict[str, DescriptorResult] = {}
        for desc_name in sorted(active_descriptors):
            desc_content = all_descriptors_dict.get(desc_name)
            if not desc_content:
                continue
            test_result = test_result_map.get(desc_name)
            status, failure_stage, failure_reason = tracker.determine_descriptor_status(
                descriptor_name=desc_name,
                test_result=test_result,
                build_dir=build_dir,
                generated_tests_dir=cpu_generated_tests_dir,
            )
            desc_path = tracker.get_descriptor_path(desc_name)
            descriptor_results[desc_name] = DescriptorResult(
                descriptor_name=desc_name,
                descriptor_path=desc_path,
                descriptor_content=desc_content,
                status=status,
                test_result=test_result,
                failure_stage=failure_stage,
                failure_reason=failure_reason,
            )

        for result in cpu_results:
            if not result.descriptor_name or result.descriptor_name in descriptor_results:
                continue
            desc = tracker.map_test_to_descriptor(result.test_name, all_descriptors_dict)
            if not desc:
                continue
            desc_name = desc.get("name", result.descriptor_name)
            if desc_name in descriptor_results:
                continue
            status, failure_stage, failure_reason = tracker.determine_descriptor_status(
                descriptor_name=desc_name,
                test_result=result,
                build_dir=build_dir,
                generated_tests_dir=cpu_generated_tests_dir,
            )
            desc_path = tracker.get_descriptor_path(desc_name)
            descriptor_results[desc_name] = DescriptorResult(
                descriptor_name=desc_name,
                descriptor_path=desc_path,
                descriptor_content=desc,
                status=status,
                test_result=result,
                failure_stage=failure_stage,
                failure_reason=failure_reason,
            )

        metadata = {
            "cpu": cpu,
            "optimization": args.opt,
            "compiler": "arm-compiler" if args.use_arm_compiler else "gcc",
            "toolchain_file": str(toolchain_file),
            "cmsis5_path": str(cmsis5),
            "fvp_exe": str(fvp_exe),
            "downloads_dir": str(args.downloads_dir),
            "source_dir": str(source_dir),
            "generated_tests_dir": str(cpu_generated_tests_dir),
            "tests_report_dir": str(cpu_tests_report_dir),
            "git_sha": _get_git_sha(source_dir),
        }
        report = TestReport(
            run_id=f"run_{cpu}_{cpu_start_time.strftime('%Y%m%d_%H%M%S')}",
            start_time=cpu_start_time,
            end_time=cpu_end_time,
            cpu=cpu,
            descriptor_results=descriptor_results,
            all_descriptors=list(all_descriptors_dict.values()),
            project_root=source_dir,
            metadata=metadata,
        )
        report_formats = getattr(args, "report_formats", None) or ["json"]
        generated_files = generator.generate_reports(report, report_formats)
        if not getattr(args, "quiet", False):
            print(
                f"Summary ({cpu}): total={report.total_tests} "
                f"passed={report.passed} failed={report.failed} skipped={report.skipped} "
                f"duration={report.duration:.2f}s"
            )
        if verbosity >= 1:
            for format_type, file_path in generated_files.items():
                print(f"{cpu} {format_type.upper()} report: {file_path}")

        if cpu_failed and args.fail_fast and verbosity >= 1:
            print("Stopping early due to failure (--fail-fast).")

        if any_fail and args.fail_fast:
            break

    _generate_coverage_reports(cpus, args, env, source_dir, compiler_tag, verbosity)
    all_results = sorted(all_results, key=lambda r: (r.cpu, r.test_name))
    return all_results, not any_fail


def _main_impl(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Build and run helia-core kernels unit tests on FVP Corstone-300 (Python).")
    ap.add_argument("-c", "--cpu", default="cortex-m55", help="Comma-separated cores, e.g. m0,m4,m55")
    ap.add_argument("-o", "--opt", default="-Ofast", help="Optimization level passed via CMSIS_OPTIMIZATION_LEVEL")
    ap.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3], default=0,
                   help="Output verbosity level (0=minimal, 1=progress, 2=commands, 3=debug)")
    ap.add_argument("-b", "--no-build", action="store_true", help="Skip build (only run)")
    ap.add_argument("-r", "--no-run", action="store_true", help="Skip run (only build)")
    ap.add_argument("-e", "--no-setup", action="store_true", help="Skip dependency setup")
    ap.add_argument("-a", "--use-arm-compiler", action="store_true", help="Use Arm Compiler (default: GCC)")
    ap.add_argument("-p", "--no-venv", action="store_true", help="(Kept for parity; no effect on CMake build)")
    ap.add_argument("-f", "--no-fvp-from-download", action="store_true", help="Do NOT use downloaded FVP; use FVP from PATH")
    ap.add_argument("-g", "--no-gcc-from-download", action="store_true", help="Do NOT use downloaded GCC; use system GCC")
    ap.add_argument("-u", "--ethos-path", type=Path, help="Override ethos-u-core-platform path")
    ap.add_argument("-C", "--cmsis5-path", type=Path, help="Override CMSIS_5 path")
    ap.add_argument("-D", "--cmake-def", action="append", default=[], help="Extra -DVAR=VAL for CMake (repeatable)")
    ap.add_argument("--coverage", action="store_true", help="Enable ns-cmsis-nn coverage instrumentation and gcda merge")
    ap.add_argument("--downloads-dir", type=Path, default=DEFAULT_DL, help="Downloads directory (default: ./artifacts/downloads)")
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE, help="CMake source dir (UnitTest root)")
    ap.add_argument("--generator", help="CMake generator (e.g. Ninja)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4, help="Parallel build jobs")
    ap.add_argument("--run-jobs", type=int, default=1, help="Parallel FVP test jobs (0 = auto/use all host cores)")
    ap.add_argument("--timeout-run", type=float, default=0.0, help="Per-test timeout in seconds (0 = none)")
    ap.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False, help="Stop on first failure")
    ap.add_argument("--fvp-arg", action="append", default=[], help="Extra args to pass to the FVP (repeatable)")
    # Reporting options
    ap.add_argument("--no-report", action="store_true", help="Disable comprehensive test reporting (enabled by default)")
    ap.add_argument("--report-formats", nargs="+", choices=["json", "html", "md", "junit"], default=["json"], 
                   help="Report formats to generate (default: json)")
    ap.add_argument("--quiet", action="store_true", help="Quiet mode (no output)")
    args = ap.parse_args(argv)

    if not is_linux():
        die("This script supports Linux only (matching the original bash script).")

    # Optional setup of downloads
    if not args.no_setup:
        call_setup_dependencies(args.downloads_dir)

    # Resolve paths / env
    ctx = detect_paths(args)
    env = ctx["env"]
    toolchain_file = ctx["toolchain_file"]
    compiler_tag = ctx["compiler_tag"]
    fvp_exe = ctx["fvp_exe"]
    cmsis5 = ctx["cmsis5"]
    source_dir = args.source_dir.resolve()

    if not source_dir.exists():
        die(f"CMake source dir not found: {source_dir}")

    try:
        cpus = parse_cpus(args.cpu)
    except ValueError as e:
        die(str(e))

    if args.run_jobs < 0:
        die(f"--run-jobs must be >= 0, got {args.run_jobs}")

    if args.coverage and args.use_arm_compiler:
        die("--coverage is only supported with GCC builds")
    if args.coverage:
        gcov_tool = _resolve_gcov_tool(env)
        if not gcov_tool:
            die("Coverage requested but no gcov-tool found on PATH (expected arm-none-eabi-gcov-tool).")
        setattr(args, "_gcov_tool", gcov_tool)
    
    # Use reporting if enabled (default: enabled, unless --no-report is set)
    enable_reporting = not args.no_report
    verbosity = getattr(args, 'verbosity', 0)
    supervisor = ProcessSupervisor(verbosity=verbosity)
    try:
        if enable_reporting:
            _, success = run_tests_with_reporting(
                cpus=cpus,
                source_dir=source_dir,
                toolchain_file=toolchain_file,
                cmsis5=cmsis5,
                fvp_exe=fvp_exe,
                compiler_tag=compiler_tag,
                args=args,
                env=env,
                supervisor=supervisor,
            )
            if success:
                if verbosity >= 1:
                    print("\nAll requested builds/runs completed successfully.")
                return 0
            return 1

        any_fail = False
        for cpu in cpus:
            if verbosity >= 1:
                print(f"\nTarget: {cpu} ({compiler_tag})")
            build_dir = canonical_build_dir(repo_root, cpu, compiler_tag)
            cpu_generated_tests_dir = resolve_generated_tests_dir(source_dir, cpu)
            coverage_ctx = _new_coverage_context(build_dir, getattr(args, "_gcov_tool", None)) if args.coverage else None

            if not args.no_build:
                cmake_configure(
                    source_dir=source_dir,
                    build_dir=build_dir,
                    toolchain_file=toolchain_file,
                    cpu=cpu,
                    cmsis5=cmsis5,
                    optimization=args.opt,
                    extra_defs=args.cmake_def,
                    generator=args.generator,
                    generated_tests_dir=cpu_generated_tests_dir,
                    enable_coverage=args.coverage,
                    verbosity=verbosity,
                    env=env,
                )
                cmake_build(build_dir=build_dir, verbosity=verbosity, env=env, jobs=args.jobs)

            if args.no_run:
                continue

            elves = find_elves(build_dir)
            if not elves:
                if verbosity >= 1:
                    print(f"(no .elf found under {build_dir}, nothing to run)")
                continue

            elf_entries = [(elf, elf.stem) for elf in elves]
            cpu_results, cpu_failed = _run_elf_jobs_with_reporting(
                elf_entries=elf_entries,
                fvp_exe=fvp_exe,
                timeout=args.timeout_run,
                verbosity=verbosity,
                extra_args=args.fvp_arg,
                env=env,
                cpu=cpu,
                coverage_ctx=coverage_ctx,
                run_jobs=args.run_jobs,
                fail_fast=args.fail_fast,
                supervisor=supervisor,
            )
            any_fail = any_fail or cpu_failed
            if cpu_failed and args.fail_fast:
                if verbosity >= 1:
                    print("Stopping early due to failure (--fail-fast).")
                break

        if any_fail:
            return 1

        _generate_coverage_reports(cpus, args, env, source_dir, compiler_tag, verbosity)

        if verbosity >= 1:
            print("\nAll requested builds/runs completed successfully.")
        return 0
    finally:
        supervisor.terminate_all("shutdown")


def main(argv: List[str]) -> int:
    try:
        return _main_impl(argv)
    except FvpScriptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
