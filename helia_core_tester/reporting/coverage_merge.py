"""
Merged multi-CPU coverage reporting with expected-zero classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import json
import html
import shutil
import subprocess

from helia_core_tester.core.cpu_targets import parse_cpu_list
from helia_core_tester.core.path_layout import coverage_merged_dir, coverage_report_dir


@dataclass
class CoverageMergeReport:
    project_root: Path
    cpus: List[str]
    coverage_inputs: Dict[str, str]
    missing_coverage_inputs: Dict[str, str]
    expected_zero_config: Optional[Path]
    merged_lcov_path: Path
    summary_json_path: Path
    summary_md_path: Path
    summary_html_path: Path
    html_generator: str
    html_generation_note: Optional[str]
    total_lh: int
    total_lf: int
    overall_line_rate: float
    file_coverage: List[Dict[str, object]]
    covered_files: List[str]
    zero_reachable_files: List[str]
    expected_zero_files: List[str]
    expected_zero_but_covered_files: List[str]
    expected_zero_missing_files: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "cpus": self.cpus,
            "coverage_inputs": self.coverage_inputs,
            "missing_coverage_inputs": self.missing_coverage_inputs,
            "expected_zero_config": str(self.expected_zero_config) if self.expected_zero_config else None,
            "merged_lcov_path": str(self.merged_lcov_path),
            "summary_json_path": str(self.summary_json_path),
            "summary_md_path": str(self.summary_md_path),
            "summary_html_path": str(self.summary_html_path),
            "html_generator": self.html_generator,
            "html_generation_note": self.html_generation_note,
            "total_lh": self.total_lh,
            "total_lf": self.total_lf,
            "overall_line_rate": self.overall_line_rate,
            "file_coverage": self.file_coverage,
            "covered_files": self.covered_files,
            "zero_reachable_files": self.zero_reachable_files,
            "expected_zero_files": self.expected_zero_files,
            "expected_zero_but_covered_files": self.expected_zero_but_covered_files,
            "expected_zero_missing_files": self.expected_zero_missing_files,
            "counts": {
                "covered": len(self.covered_files),
                "zero_reachable": len(self.zero_reachable_files),
                "expected_zero": len(self.expected_zero_files),
                "expected_zero_but_covered": len(self.expected_zero_but_covered_files),
                "expected_zero_missing": len(self.expected_zero_missing_files),
            },
        }


def _normalize_rel_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _relative_source_path(source_path: str, project_root: Path) -> str:
    raw = source_path.strip()
    source = Path(raw)
    try:
        return _normalize_rel_path(str(source.resolve().relative_to(project_root.resolve())))
    except Exception:
        pass

    normalized = raw.replace("\\", "/")
    for marker in ("/Source/", "/Include/"):
        idx = normalized.rfind(marker)
        if idx != -1:
            return _normalize_rel_path(normalized[idx + 1 :])

    return _normalize_rel_path(normalized)


def _parse_lcov(lcov_path: Path) -> Dict[str, Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    current_path: Optional[str] = None

    for raw_line in lcov_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("SF:"):
            current_path = line[3:].strip()
            records[current_path] = {
                "lines": {},
                "lf": None,
                "lh": None,
            }
            continue

        if not current_path:
            continue

        if line.startswith("DA:"):
            payload = line[3:].split(",")
            if len(payload) >= 2:
                try:
                    line_no = int(payload[0])
                    hit_count = int(payload[1])
                except ValueError:
                    continue
                lines = records[current_path]["lines"]
                lines[line_no] = lines.get(line_no, 0) + hit_count
            continue

        if line.startswith("LF:"):
            try:
                records[current_path]["lf"] = int(line[3:])
            except ValueError:
                pass
            continue

        if line.startswith("LH:"):
            try:
                records[current_path]["lh"] = int(line[3:])
            except ValueError:
                pass
            continue

        if line == "end_of_record":
            current_path = None

    return records


def _load_expected_zero(expected_zero_path: Optional[Path]) -> Tuple[Set[str], Optional[Path]]:
    if expected_zero_path is None:
        return set(), None
    if not expected_zero_path.exists():
        return set(), expected_zero_path

    try:
        payload = json.loads(expected_zero_path.read_text())
    except Exception:
        return set(), expected_zero_path

    values = payload.get("expected_zero_files", [])
    if not isinstance(values, list):
        return set(), expected_zero_path

    normalized = {_normalize_rel_path(str(v)) for v in values if str(v).strip()}
    return normalized, expected_zero_path


def _write_merged_lcov(
    merged_lcov_path: Path,
    project_root: Path,
    merged_lines: Dict[str, Dict[int, int]],
    merged_lf: Dict[str, int],
) -> None:
    out: List[str] = []
    for rel in sorted(merged_lines.keys()):
        file_path = project_root / rel
        sf = str(file_path.resolve()) if file_path.exists() else rel
        lines = merged_lines[rel]
        lf = max(merged_lf.get(rel, 0), len(lines))
        lh = sum(1 for c in lines.values() if c > 0)

        out.append("TN:")
        out.append(f"SF:{sf}")
        for line_no in sorted(lines.keys()):
            out.append(f"DA:{line_no},{lines[line_no]}")
        out.append(f"LF:{lf}")
        out.append(f"LH:{lh}")
        out.append("end_of_record")

    merged_lcov_path.write_text("\n".join(out) + ("\n" if out else ""))


def _write_markdown(report: CoverageMergeReport, path: Path) -> None:
    def fmt(items: Iterable[str]) -> str:
        values = list(items)
        if not values:
            return "(none)"
        return "\n".join(f"- {v}" for v in values)

    content = [
        "# Merged Coverage Triage",
        f"- CPUs: `{', '.join(report.cpus)}`",
        f"- Coverage inputs found: `{len(report.coverage_inputs)}`",
        f"- Coverage inputs missing: `{len(report.missing_coverage_inputs)}`",
        f"- Overall merged line coverage: `{report.total_lh}/{report.total_lf}` ({report.overall_line_rate:.2f}%)",
        f"- HTML generator: `{report.html_generator}`",
        f"- HTML generation note: `{report.html_generation_note}`" if report.html_generation_note else "- HTML generation note: (none)",
        f"- Expected-zero config: `{report.expected_zero_config}`" if report.expected_zero_config else "- Expected-zero config: (none)",
        "",
        "## Covered (hit on at least one CPU)",
        fmt(report.covered_files),
        "",
        "## Zero-Hit Reachable (not expected-zero)",
        fmt(report.zero_reachable_files),
        "",
        "## Zero-Hit Expected (orphan/known-unreachable)",
        fmt(report.expected_zero_files),
        "",
        "## Expected-Zero But Covered (allowlist cleanup candidates)",
        fmt(report.expected_zero_but_covered_files),
        "",
        "## Expected-Zero Entries Missing from Coverage Inputs",
        fmt(report.expected_zero_missing_files),
        "",
        "## Missing Coverage Inputs",
        fmt(f"{cpu}: {path_str}" for cpu, path_str in sorted(report.missing_coverage_inputs.items())),
        "",
    ]
    path.write_text("\n".join(content))


def _write_html(report: CoverageMergeReport, path: Path) -> None:
    rows: List[str] = []
    for item in report.file_coverage:
        file_name = html.escape(str(item["file"]))
        lh = int(item["lh"])
        lf = int(item["lf"])
        line_rate = float(item["line_rate"])
        classification = html.escape(str(item["classification"]))
        covered_on_cpus = ", ".join(item.get("covered_on_cpus", []))
        rows.append(
            "<tr>"
            f"<td><code>{file_name}</code></td>"
            f"<td>{lh}</td>"
            f"<td>{lf}</td>"
            f"<td>{line_rate:.2f}%</td>"
            f"<td>{classification}</td>"
            f"<td>{html.escape(covered_on_cpus)}</td>"
            "</tr>"
        )

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Merged Coverage Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin: 0 0 10px 0; }}
    .meta {{ margin: 6px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f7f7f7; }}
    tr:nth-child(even) {{ background: #fbfbfb; }}
    code {{ white-space: nowrap; }}
  </style>
</head>
<body>
  <h1>Merged Coverage Report</h1>
  <div class="meta"><strong>CPUs:</strong> {html.escape(", ".join(report.cpus))}</div>
  <div class="meta"><strong>Overall line coverage:</strong> {report.total_lh}/{report.total_lf} ({report.overall_line_rate:.2f}%)</div>
  <div class="meta"><strong>Covered files:</strong> {len(report.covered_files)}</div>
  <div class="meta"><strong>Zero-hit reachable files:</strong> {len(report.zero_reachable_files)}</div>
  <div class="meta"><strong>Expected-zero files:</strong> {len(report.expected_zero_files)}</div>
  <h2>Per-file Coverage</h2>
  <table>
    <thead>
      <tr>
        <th>File</th>
        <th>LH</th>
        <th>LF</th>
        <th>Line %</th>
        <th>Classification</th>
        <th>Covered on CPUs</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(content)


def _line_rate(lh: int, lf: int) -> float:
    if lf <= 0:
        return 0.0
    return (float(lh) / float(lf)) * 100.0


def _resolve_gcov_executable(root: Path, build_dirs: List[Path]) -> Optional[str]:
    # Prefer the toolchain downloaded by helia_core_tester.
    candidate = root / "artifacts" / "downloads" / "arm_gcc_download" / "bin" / "arm-none-eabi-gcov"
    if candidate.exists():
        return str(candidate)

    # Fallback: infer from CMake cache tool paths.
    for build_dir in build_dirs:
        addr2line = _read_cmake_cache_path(build_dir, "CMAKE_ADDR2LINE")
        if addr2line and addr2line.exists():
            inferred = addr2line.parent / "arm-none-eabi-gcov"
            if inferred.exists():
                return str(inferred)

    # Last resort: PATH lookup.
    for exe in ("arm-none-eabi-gcov", "gcov"):
        resolved = shutil.which(exe)
        if resolved:
            return resolved
    return None


def _read_cmake_cache_path(build_dir: Path, key: str) -> Optional[Path]:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return None
    prefix = f"{key}:PATH="
    for line in cache.read_text(errors="ignore").splitlines():
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            if value:
                return Path(value).resolve()
    return None


def _resolve_cmsis_nn_root(tester_root: Path, build_dirs: List[Path]) -> Path:
    if (tester_root / "Source").exists():
        return tester_root

    for build_dir in build_dirs:
        cmsis_nn_root = _read_cmake_cache_path(build_dir, "CMSIS_NN_ROOT")
        if cmsis_nn_root and cmsis_nn_root.exists() and (cmsis_nn_root / "Source").exists():
            return cmsis_nn_root

    fallback = (tester_root / ".." / "..").resolve()
    if (fallback / "Source").exists():
        return fallback
    return tester_root


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True, (result.stdout or "").strip()
    output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    return False, output


def _try_write_gcovr_html(
    root: Path,
    cpus: List[str],
    suites: List[str],
    out_dir: Path,
) -> Tuple[bool, str]:
    gcovr = shutil.which("gcovr")
    if not gcovr:
        return False, "gcovr not found on PATH"

    source_filter = r"(^|.*/)Source/.*"
    gcov_exe: Optional[str] = None
    trace_dir = out_dir / "gcovr-traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_files: List[Path] = []
    build_dirs: List[Path] = []
    errors: List[str] = []

    for suite in suites:
        for cpu in cpus:
            build_dir = root / "artifacts" / f"build-{suite}-{cpu}-gcc"
            if not build_dir.exists():
                errors.append(f"{suite}:{cpu}: build dir missing ({build_dir})")
                continue
            build_dirs.append(build_dir)

    if not build_dirs:
        return False, "; ".join(errors) if errors else "no build directories found"

    gcovr_root = _resolve_cmsis_nn_root(root, build_dirs)
    gcov_exe = _resolve_gcov_executable(root, build_dirs)

    for suite in suites:
        for cpu in cpus:
            build_dir = root / "artifacts" / f"build-{suite}-{cpu}-gcc"
            if not build_dir.exists():
                continue
            trace_path = trace_dir / f"{suite}_{cpu}.json"
            cmd = [
                gcovr,
                "--root",
                str(gcovr_root),
                "--filter",
                source_filter,
                "--gcov-ignore-parse-errors",
                "suspicious_hits.warn_once_per_file",
                "--gcov-ignore-errors=no_working_dir_found",
                "--merge-mode-functions",
                "merge-use-line-min",
                "--object-directory",
                str(build_dir),
                "--json",
                str(trace_path),
                str(build_dir),
            ]
            if gcov_exe:
                cmd.extend(["--gcov-executable", gcov_exe])

            ok, details = _run_cmd(cmd, cwd=build_dir)
            if ok and trace_path.exists():
                trace_files.append(trace_path)
            else:
                errors.append(f"{suite}:{cpu}: {details}")

    if not trace_files:
        if errors:
            return False, "gcovr trace generation failed for all CPUs: " + " | ".join(errors)
        return False, "gcovr trace generation failed for all CPUs"

    html_path = out_dir / "index.html"
    cmd = [
        gcovr,
        "--root",
        str(gcovr_root),
        "--filter",
        source_filter,
        "--gcov-ignore-parse-errors",
        "suspicious_hits.warn_once_per_file",
        "--gcov-ignore-errors=no_working_dir_found",
        "--merge-mode-functions",
        "merge-use-line-min",
        "--html-details",
        str(html_path),
        "--html-title",
        f"Merged Coverage ({', '.join(suites)} x {', '.join(cpus)})",
    ]
    for trace_file in trace_files:
        cmd.extend(["--add-tracefile", str(trace_file)])

    ok, details = _run_cmd(cmd, cwd=out_dir)
    if not ok or not html_path.exists():
        return False, f"gcovr html generation failed: {details}"

    return True, "generated via gcovr --add-tracefile"


def run_coverage_merge(
    project_root: Path,
    cpus: str | Iterable[str],
    suites: Iterable[str] | None = None,
    report_dir: Optional[Path] = None,
    expected_zero_config: Optional[Path] = None,
) -> Tuple[int, CoverageMergeReport]:
    root = Path(project_root).resolve()
    cpu_list = parse_cpu_list(cpus)
    suite_list = ["int", "float"] if suites is None else [str(item).strip().lower() for item in suites]
    suite_list = [item for item in suite_list if item in {"int", "float", "float-mve"}]
    if not suite_list:
        suite_list = ["int", "float"]

    if expected_zero_config is None:
        expected_zero_config = root / "assets" / "coverage_expected_zero.json"
    expected_zero_set, expected_zero_path = _load_expected_zero(expected_zero_config)

    out_dir = (Path(report_dir).resolve() if report_dir else coverage_merged_dir(root))
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_lines: Dict[str, Dict[int, int]] = {}
    merged_lf: Dict[str, int] = {}
    sources_with_hits: Dict[str, Set[str]] = {}
    coverage_inputs: Dict[str, str] = {}
    missing_inputs: Dict[str, str] = {}

    for suite in suite_list:
        for cpu in cpu_list:
            # float-mve coverage is produced only for cortex-m55 and is optional:
            # skip other CPUs and do not treat its absence as a missing required input.
            optional_suite = suite == "float-mve"
            if optional_suite and cpu != "cortex-m55":
                continue

            source_key = f"{suite}:{cpu}"
            lcov_path = coverage_report_dir(root, cpu, suite=suite) / "coverage.info"
            if not lcov_path.exists():
                if not optional_suite:
                    missing_inputs[source_key] = str(lcov_path)
                continue

            coverage_inputs[source_key] = str(lcov_path)
            records = _parse_lcov(lcov_path)
            for source_path, record in records.items():
                rel = _relative_source_path(source_path, root)
                lines = record["lines"]

                if rel not in merged_lines:
                    merged_lines[rel] = {}
                if rel not in sources_with_hits:
                    sources_with_hits[rel] = set()

                for line_no, count in lines.items():
                    merged_lines[rel][line_no] = merged_lines[rel].get(line_no, 0) + count

                if record.get("lf") is not None:
                    merged_lf[rel] = max(merged_lf.get(rel, 0), int(record["lf"]))
                else:
                    merged_lf[rel] = max(merged_lf.get(rel, 0), len(merged_lines[rel]))

                if any(count > 0 for count in lines.values()) or int(record.get("lh") or 0) > 0:
                    sources_with_hits[rel].add(source_key)

    covered_files: List[str] = []
    zero_reachable_files: List[str] = []
    expected_zero_files: List[str] = []
    expected_zero_but_covered_files: List[str] = []
    file_coverage: List[Dict[str, object]] = []
    total_lh = 0
    total_lf = 0

    all_files = set(merged_lines.keys())
    for rel in sorted(all_files):
        line_hits = merged_lines.get(rel, {})
        lf = max(merged_lf.get(rel, 0), len(line_hits))
        lh = sum(1 for count in line_hits.values() if count > 0)
        total_lh += lh
        total_lf += lf
        covered = lh > 0
        expected_zero = rel in expected_zero_set
        classification = "covered"

        if covered:
            covered_files.append(rel)
            if expected_zero:
                expected_zero_but_covered_files.append(rel)
                classification = "expected_zero_but_covered"
        else:
            if expected_zero:
                expected_zero_files.append(rel)
                classification = "expected_zero"
            else:
                zero_reachable_files.append(rel)
                classification = "zero_reachable"

        file_coverage.append(
            {
                "file": rel,
                "lh": lh,
                "lf": lf,
                "line_rate": round(_line_rate(lh, lf), 2),
                "classification": classification,
                "covered_on_cpus": sorted(sources_with_hits.get(rel, set())),
            }
        )

    expected_zero_missing_files = sorted(expected_zero_set - all_files)
    overall_line_rate = round(_line_rate(total_lh, total_lf), 2)

    merged_lcov_path = out_dir / "coverage_merged.info"
    _write_merged_lcov(merged_lcov_path, root, merged_lines, merged_lf)

    report = CoverageMergeReport(
        project_root=root,
        cpus=cpu_list,
        coverage_inputs=coverage_inputs,
        missing_coverage_inputs=missing_inputs,
        expected_zero_config=expected_zero_path,
        merged_lcov_path=merged_lcov_path,
        summary_json_path=out_dir / "coverage_merged_summary.json",
        summary_md_path=out_dir / "coverage_merged_summary.md",
        summary_html_path=out_dir / "index.html",
        html_generator="builtin",
        html_generation_note=None,
        total_lh=total_lh,
        total_lf=total_lf,
        overall_line_rate=overall_line_rate,
        file_coverage=file_coverage,
        covered_files=covered_files,
        zero_reachable_files=zero_reachable_files,
        expected_zero_files=expected_zero_files,
        expected_zero_but_covered_files=expected_zero_but_covered_files,
        expected_zero_missing_files=expected_zero_missing_files,
    )

    gcovr_ok, gcovr_note = _try_write_gcovr_html(root, cpu_list, suite_list, out_dir)
    if gcovr_ok:
        report.html_generator = "gcovr"
        report.html_generation_note = gcovr_note
    else:
        report.html_generator = "builtin"
        report.html_generation_note = gcovr_note
        _write_html(report, report.summary_html_path)

    _write_markdown(report, report.summary_md_path)
    report.summary_json_path.write_text(json.dumps(report.to_dict(), indent=2))

    if not coverage_inputs:
        exit_code = 1
    elif len(suite_list) == 1:
        exit_code = 0 if not missing_inputs else 1
    else:
        covered_cpus = {
            key.split(":", 1)[1]
            for key in coverage_inputs.keys()
            if ":" in key
        }
        exit_code = 0 if all(cpu in covered_cpus for cpu in cpu_list) else 1
    return exit_code, report
