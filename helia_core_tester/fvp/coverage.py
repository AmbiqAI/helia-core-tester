"""Coverage stream processing and report generation for FVP runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from helia_core_tester.core.path_layout import build_dir as canonical_build_dir
from helia_core_tester.core.path_layout import coverage_report_dir as canonical_coverage_report_dir

from .cmake import read_cmake_cache_path
from .env import ARTIFACTS_DIR, REPO_ROOT, resolve_gcov_executable


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


def new_coverage_context(build_dir: Path, gcov_tool: Optional[str]) -> CoverageContext:
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
    except subprocess.CalledProcessError as exc:
        if verbosity >= 1:
            err = (exc.stderr or b"").decode(errors="replace").strip()
            print(f"Coverage merge failed: {ctx.gcov_tool} merge-stream ({err})", file=sys.stderr)
        return False


def process_coverage_output(
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


def _coverage_report_dir(cpu: str, suite: str) -> Path:
    return canonical_coverage_report_dir(REPO_ROOT, cpu, suite=suite)


def generate_coverage_reports(
    cpus: List[str],
    suite: str,
    args,
    env: dict,
    source_dir: Path,
    compiler_tag: str,
    verbosity: int,
) -> None:
    if not getattr(args, "coverage", False):
        return
    report_suite = getattr(args, "coverage_report_suite", None) or suite

    gcovr = shutil.which("gcovr", path=env.get("PATH"))
    gcov_exe = resolve_gcov_executable(env)

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
        cpu_dir = _coverage_report_dir(cpu, report_suite)
        if cpu_dir.exists():
            shutil.rmtree(cpu_dir, ignore_errors=True)
        cpu_dir.mkdir(parents=True, exist_ok=True)

        cpu_build_dir = canonical_build_dir(REPO_ROOT, cpu, compiler_tag, suite=suite)
        if not cpu_build_dir.exists():
            continue

        default_cmsis_nn_root = (source_dir / ".." / "..").resolve()
        cmsis_nn_root = read_cmake_cache_path(cpu_build_dir, "CMSIS_NN_ROOT")
        if cmsis_nn_root is None:
            cmsis_nn_root = default_cmsis_nn_root
        elif not cmsis_nn_root.exists():
            if verbosity >= 2:
                print(
                    "Coverage: CMSIS_NN_ROOT from CMake cache does not exist "
                    f"({cmsis_nn_root}); falling back to {default_cmsis_nn_root}"
                )
            cmsis_nn_root = default_cmsis_nn_root

        source_filter = r"(^|.*/)Source/.*"

        cmd = [
            gcovr,
            "--root",
            str(cmsis_nn_root),
            "--filter",
            source_filter,
            "--gcov-ignore-parse-errors",
            "suspicious_hits.warn_once_per_file",
            "--gcov-ignore-errors=no_working_dir_found",
            "--decisions",
            "--object-directory",
            str(cpu_build_dir),
            "--txt-summary",
            "--json-summary",
            str(cpu_dir / "summary.json"),
            "--json-summary-pretty",
            "--html-details",
            str(cpu_dir / "index.html"),
            "--lcov",
            str(cpu_dir / "coverage.info"),
            str(cpu_build_dir),
        ]
        if gcov_exe:
            cmd.extend(["--gcov-executable", gcov_exe])

        result = subprocess.run(
            cmd,
            cwd=str(cpu_build_dir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        (cpu_dir / "summary.txt").write_text((result.stdout or "") + (result.stderr or ""))
        if result.returncode != 0 and verbosity >= 1:
            print(f"WARNING: coverage report generation failed for {cpu} (rc={result.returncode})", file=sys.stderr)
        elif verbosity >= 1:
            print(f"Coverage report generated: {cpu_dir / 'index.html'}")
