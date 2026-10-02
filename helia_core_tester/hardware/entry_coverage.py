"""CMSIS-NN entry points: public, timed, and deployed.

Timed: each unrejected case's registry `cmsis_function` plus the
arm_* names in its manifest capabilities. Deployed comes from
assets/deployed_entry_points.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from .adapter_specs import timed_kernel_calls
from .kernel_registry import load_kernel_registry
from .pathutil import write_text_lf

DEPLOYED_PATH = Path("assets/deployed_entry_points.json")
COVERAGE_FILE = "coverage.json"
NO_ADAPTER = "no_hardware_adapter"


def public_entry_points(include_dir: Path) -> list[str]:
    """Kernels the public headers declare, minus setup calls."""
    headers = sorted(include_dir.glob("arm_nnfunctions*.h"))
    return timed_kernel_calls("\n".join(h.read_text(encoding="utf-8") for h in headers))


def built_include_dir(build_dir: Optional[Path]) -> Optional[Path]:
    """Include/ of the kernels the build compiled."""
    from .firmware_build import nsx_app_dir
    from .nsx_app import kernel_dir, saved_options

    if build_dir is None:
        return None
    app_dir = nsx_app_dir(build_dir)
    options = saved_options(app_dir)
    if options is None:
        return None
    include = (options.cmsis_nn_root or kernel_dir(app_dir, options)) / "Include"
    return include if include.is_dir() else None


def build_coverage(project_root: Path, cases: Sequence, build_dir: Optional[Path]) -> dict[str, Any]:
    """The bundle's coverage section for `cases`."""
    by_id = {entry.kernel_id: entry.cmsis_function for entry in load_kernel_registry(project_root)}
    timed_set: set[str] = set()
    for case in cases:
        if case.rejection is not None or not case.samples:
            continue
        bundle = case.case_bundle
        # Capabilities name per-axis variants.
        capabilities = bundle.manifest.get("required_target_capabilities") or ()
        timed_set.update(name for name in capabilities if name.startswith("arm_"))
        if bundle.kernel_id in by_id:
            timed_set.add(by_id[bundle.kernel_id])
    timed = sorted(timed_set)
    deployed = json.loads((project_root / DEPLOYED_PATH).read_text(encoding="utf-8"))
    deployed_names = deployed["entry_points"]
    include = built_include_dir(build_dir)
    public = public_entry_points(include) if include is not None else None
    deployed_timed = [name for name in deployed_names if name in timed]
    return {
        "schema": "hct.hardware.coverage",
        "schema_version": 1,
        "public": {
            "include_dir": str(include) if include is not None else None,
            "entry_points": public,
            "timed": None if public is None else len(set(public) & set(timed)),
            "total": None if public is None else len(public),
        },
        "timed_entry_points": timed,
        "deployed": {
            "source": deployed["source"],
            "entry_points": deployed_names,
            "missing": [name for name in deployed_names if name not in timed],
        },
        "deployed_entry_points_timed": len(deployed_timed),
        "deployed_entry_points_total": len(deployed_names),
    }


def write_coverage(bundle_root: Path, coverage: dict, skipped: Sequence[tuple]) -> Path:
    """Add coverage.json and skipped cases to a bundle."""
    path = bundle_root / COVERAGE_FILE
    write_text_lf(path, json.dumps(coverage, indent=2))
    manifest_path = bundle_root / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("artifacts", {})["coverage"] = COVERAGE_FILE
    write_text_lf(manifest_path, json.dumps(manifest, indent=2))
    summary_path = bundle_root / "session_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["skipped_cases"] = [
        {"case_id": test.name, "family": test.family, "suite": test.suite, "reason": reason}
        for test, reason in skipped
    ]
    write_text_lf(summary_path, json.dumps(summary, indent=2))
    return path


def coverage_line(coverage: dict) -> str:
    """One line: deployed and public entry points timed."""
    line = (
        f"Coverage: {coverage['deployed_entry_points_timed']}/{coverage['deployed_entry_points_total']} "
        "deployed entry points timed"
    )
    public = coverage["public"]
    if public["total"] is not None:
        line += f", {public['timed']}/{public['total']} public"
    return line


def coverage_totals(coverage: Optional[dict]) -> Optional[dict[str, Any]]:
    """Counts and missing names for the run summary."""
    if coverage is None:
        return None
    return {
        "deployed_entry_points_timed": coverage["deployed_entry_points_timed"],
        "deployed_entry_points_total": coverage["deployed_entry_points_total"],
        "deployed_missing": coverage["deployed"]["missing"],
        "public_entry_points_timed": coverage["public"]["timed"],
        "public_entry_points_total": coverage["public"]["total"],
    }
