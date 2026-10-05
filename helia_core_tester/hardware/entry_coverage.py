"""CMSIS-NN entry points: public, timed, and deployed.

Public: kernels the build linked (kernel_symbol_refs.inc).
Timed: each unrejected case's arm_* capability, else its registry
`cmsis_function`. Deployed: assets/deployed_entry_points.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from .adapter_specs import is_setup_call
from .kernel_registry import load_kernel_registry
from .pathutil import write_text_lf
from .result_bundle import merge_summary

DEPLOYED_PATH = Path("assets/deployed_entry_points.json")
COVERAGE_FILE = "coverage.json"
NO_ADAPTER = "no_hardware_adapter"
SYMBOL_REFS = "kernel_symbol_refs.inc"
_SYMBOL_REF = re.compile(r'^\{ "(arm_\w+)"', re.M)


def built_entry_points(build_dir: Optional[Path]) -> tuple[Optional[Path], Optional[list[str]]]:
    """Public kernels the build linked."""
    from .firmware_build import IMAGE_SUBDIR

    if build_dir is None:
        return None, None
    path = build_dir / IMAGE_SUBDIR / SYMBOL_REFS
    if not path.is_file():
        return None, None
    names = _SYMBOL_REF.findall(path.read_text(encoding="utf-8"))
    return path, sorted({name for name in names if not is_setup_call(name)})


def build_coverage(project_root: Path, cases: Sequence, build_dir: Optional[Path]) -> dict[str, Any]:
    """The bundle's coverage section for `cases`."""
    by_id = {entry.kernel_id: entry.cmsis_function for entry in load_kernel_registry(project_root)}
    timed_set: set[str] = set()
    for case in cases:
        if case.rejection is not None or not case.samples:
            continue
        bundle = case.case_bundle
        # Capabilities name the exact variant.
        capabilities = bundle.manifest.get("required_target_capabilities") or ()
        named = {name for name in capabilities if name.startswith("arm_")}
        if not named and bundle.kernel_id in by_id:
            named = {by_id[bundle.kernel_id]}
        timed_set.update(named)
    timed = sorted(timed_set)
    deployed = json.loads((project_root / DEPLOYED_PATH).read_text(encoding="utf-8"))
    deployed_names = deployed["entry_points"]
    refs, public = built_entry_points(build_dir)
    deployed_timed = [name for name in deployed_names if name in timed]
    return {
        "schema": "hct.hardware.coverage",
        "schema_version": 1,
        "public": {
            "source": str(refs) if refs is not None else None,
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
    merge_summary(bundle_root, "skipped_cases", [
        {"case_id": test.name, "family": test.family, "suite": test.suite, "reason": reason}
        for test, reason in skipped
    ])
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
