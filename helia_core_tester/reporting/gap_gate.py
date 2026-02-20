"""
Gap gate for generated tests vs descriptors.

Detects descriptor generation gaps and manifest inconsistencies, then
returns a process-style exit code (0 = ok, 1 = unexpected gap).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import json

from helia_core_tester.generation.io.descriptors import load_all_descriptors


ALLOWLIST_DESCRIPTOR_NAMES: Set[str] = {
    "argmax_axis0_s16",
    "argmax_axis1_s8",
    "argmax_axis2_s16",
    "argmax_axis3_s8",
    "argmin_axis0_s8",
    "argmin_axis1_s16",
    "argmin_axis2_s8",
    "argmin_axis3_s16",
    "batch_to_space_nd_basic_s8",
    "batch_to_space_nd_block41_s8",
    "batch_to_space_nd_crops_s8",
    "batch_to_space_nd_int16",
    "concatenation_const_vector_s8",
    "concatenation_width_dual_s8",
    "depth_to_space_block2_s8",
    "depth_to_space_block3_s8",
    "depth_to_space_block4_s8",
    "depth_to_space_wide_s8",
    "equal_tensor_s16",
    "equal_tensor_s8",
    "greater_equal_tensor_s16",
    "greater_equal_tensor_s8",
    "greater_tensor_s16",
    "greater_tensor_s8",
    "less_broadcast_s8",
    "less_equal_tensor_s16",
    "less_equal_tensor_s8",
    "less_tensor_s16",
    "less_tensor_s8",
    "not_equal_tensor_s16",
    "not_equal_tensor_s8",
    "pack_axis0_triple_s8",
    "pack_axis1_quad_s8",
    "pack_axis_last_pair_s8",
    "shape_image_s8",
    "shape_vector_s16",
    "slice_center_crs8",
    "slice_shrink_axis_s8",
    "space_to_batch_nd_basic_s8",
    "space_to_batch_nd_int16",
    "space_to_batch_nd_padded_height_s8",
    "space_to_batch_nd_wide_padding_s8",
    "space_to_depth_block2_s8",
    "space_to_depth_block3_s8",
    "space_to_depth_block4_s8",
    "space_to_depth_rectangular_s8",
    "split_channels_pairs_s8",
    "split_height_triple_s8",
    "split_v_channels_mixed_s8",
    "split_v_height_tail_s8",
    "split_v_width_balanced_s8",
    "split_width_quads_s8",
    "transpose_time_batch_s8",
    "unpack_axis1_triple_s8",
    "unpack_axis2_pair_s8",
    "unpack_axis_last_quad_s8",
}


ALLOWLIST_OPERATORS: Set[str] = {
    "ArgMax",
    "ArgMin",
    "BatchToSpaceND",
    "Concatenation",
    "DepthToSpace",
    "Equal",
    "Greater",
    "GreaterEqual",
    "Less",
    "LessEqual",
    "NotEqual",
    "Pack",
    "Shape",
    "Slice",
    "SpaceToBatchND",
    "SpaceToDepth",
    "Split",
    "Transpose",
    "Unpack",
}


@dataclass
class GapReport:
    cpu: str
    generated_tests_dir: Path
    report_dir: Path
    manifest_path: Optional[Path]
    descriptor_without_tflite: List[str]
    manifest_missing_descriptor: List[str]
    tflite_missing_manifest_entry: List[str]
    operators_zero_in_manifest: List[str]
    allowlisted_descriptor_gaps: List[str]
    unexpected_descriptor_gaps: List[str]
    allowlisted_zero_manifest_ops: List[str]
    unexpected_zero_manifest_ops: List[str]
    manifest_missing: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "cpu": self.cpu,
            "generated_tests_dir": str(self.generated_tests_dir),
            "report_dir": str(self.report_dir),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "manifest_missing": self.manifest_missing,
            "descriptor_without_tflite": self.descriptor_without_tflite,
            "manifest_missing_descriptor": self.manifest_missing_descriptor,
            "tflite_missing_manifest_entry": self.tflite_missing_manifest_entry,
            "operators_zero_in_manifest": self.operators_zero_in_manifest,
            "allowlisted_descriptor_gaps": self.allowlisted_descriptor_gaps,
            "unexpected_descriptor_gaps": self.unexpected_descriptor_gaps,
            "allowlisted_zero_manifest_ops": self.allowlisted_zero_manifest_ops,
            "unexpected_zero_manifest_ops": self.unexpected_zero_manifest_ops,
        }


def resolve_generated_tests_dir(project_root: Path, cpu: str) -> Path:
    base = project_root / "artifacts" / "generated_tests"
    cpu_specific = base / cpu
    if cpu_specific.exists():
        return cpu_specific
    return base


def load_manifest(generated_tests_dir: Path) -> Tuple[Optional[Path], Dict[str, object]]:
    manifest_path = generated_tests_dir / "manifest.json"
    if not manifest_path.exists():
        return None, {}
    try:
        data = json.loads(manifest_path.read_text())
        if not isinstance(data, dict):
            return manifest_path, {}
        return manifest_path, data
    except Exception:
        return manifest_path, {}


def scan_generated_dirs(generated_tests_dir: Path) -> Dict[str, Dict[str, bool]]:
    result: Dict[str, Dict[str, bool]] = {}
    if not generated_tests_dir.exists():
        return result
    for entry in generated_tests_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        descriptor = entry / "descriptor.yaml"
        tflite = entry / f"{name}.tflite"
        includes_dir = entry / "includes"
        has_headers = includes_dir.exists() and any(includes_dir.glob("*.h"))
        has_c_sources = any(entry.glob("*.c"))
        result[name] = {
            "has_descriptor": descriptor.exists(),
            "has_tflite": tflite.exists(),
            "has_headers": has_headers,
            "has_c_sources": has_c_sources,
        }
    return result


def load_active_descriptors(project_root: Path) -> Dict[str, Dict[str, object]]:
    descriptors_dir = project_root / "assets" / "descriptors"
    if not descriptors_dir.exists():
        return {}
    descs = load_all_descriptors(str(descriptors_dir))
    by_name: Dict[str, Dict[str, object]] = {}
    for d in descs:
        name = d.get("name")
        if name:
            by_name[name] = d
    return by_name


def _manifest_tests(manifest: Dict[str, object]) -> List[Dict[str, object]]:
    tests = manifest.get("tests")
    if isinstance(tests, list):
        return [t for t in tests if isinstance(t, dict)]
    return []


def _manifest_name_set(manifest: Dict[str, object]) -> Set[str]:
    return {t.get("name") for t in _manifest_tests(manifest) if isinstance(t.get("name"), str)}


def _manifest_operator_counts(manifest: Dict[str, object]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in _manifest_tests(manifest):
        op = t.get("operator")
        if not isinstance(op, str):
            continue
        counts[op] = counts.get(op, 0) + 1
    return counts


def _descriptor_operator_counts(descriptors: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for desc in descriptors.values():
        op = desc.get("operator")
        if not isinstance(op, str):
            continue
        counts[op] = counts.get(op, 0) + 1
    return counts


def compute_gaps(project_root: Path, cpu: str, generated_tests_dir: Optional[Path] = None) -> GapReport:
    gen_dir = generated_tests_dir or resolve_generated_tests_dir(project_root, cpu)
    manifest_path, manifest = load_manifest(gen_dir)
    manifest_missing = manifest_path is None

    dir_scan = scan_generated_dirs(gen_dir)
    descriptors = load_active_descriptors(project_root)

    manifest_names = _manifest_name_set(manifest)

    descriptor_without_tflite = sorted(
        name for name, meta in dir_scan.items()
        if meta.get("has_descriptor")
        and not meta.get("has_tflite")
        and not (meta.get("has_headers") or meta.get("has_c_sources"))
    )

    manifest_missing_descriptor = sorted(
        name for name in manifest_names
        if not (gen_dir / name).exists()
    )

    tflite_missing_manifest_entry = sorted(
        name for name, meta in dir_scan.items()
        if meta.get("has_tflite") and name not in manifest_names
    )

    descriptor_ops = _descriptor_operator_counts(descriptors)
    manifest_ops = _manifest_operator_counts(manifest)
    operators_zero_in_manifest = sorted(
        op for op, count in descriptor_ops.items()
        if count > 0 and manifest_ops.get(op, 0) == 0
    )

    allowlisted_descriptor_gaps = sorted(
        name for name in descriptor_without_tflite if name in ALLOWLIST_DESCRIPTOR_NAMES
    )
    unexpected_descriptor_gaps = sorted(
        name for name in descriptor_without_tflite if name not in ALLOWLIST_DESCRIPTOR_NAMES
    )

    allowlisted_zero_manifest_ops = sorted(
        op for op in operators_zero_in_manifest if op in ALLOWLIST_OPERATORS
    )
    unexpected_zero_manifest_ops = sorted(
        op for op in operators_zero_in_manifest if op not in ALLOWLIST_OPERATORS
    )

    report_dir = project_root / "artifacts" / "reports"
    return GapReport(
        cpu=cpu,
        generated_tests_dir=gen_dir,
        report_dir=report_dir,
        manifest_path=manifest_path,
        descriptor_without_tflite=descriptor_without_tflite,
        manifest_missing_descriptor=manifest_missing_descriptor,
        tflite_missing_manifest_entry=tflite_missing_manifest_entry,
        operators_zero_in_manifest=operators_zero_in_manifest,
        allowlisted_descriptor_gaps=allowlisted_descriptor_gaps,
        unexpected_descriptor_gaps=unexpected_descriptor_gaps,
        allowlisted_zero_manifest_ops=allowlisted_zero_manifest_ops,
        unexpected_zero_manifest_ops=unexpected_zero_manifest_ops,
        manifest_missing=manifest_missing,
    )


def _write_json(report: GapReport, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"gap_report_{report.cpu}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path


def _write_md(report: GapReport, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"gap_report_{report.cpu}.md"

    def _list(items: Iterable[str]) -> str:
        items = list(items)
        if not items:
            return "(none)"
        return "\n".join(f"- {i}" for i in items)

    content = [
        "# Gap Report",
        f"- CPU: `{report.cpu}`",
        f"- Generated tests dir: `{report.generated_tests_dir}`",
        f"- Manifest: `{report.manifest_path}`" if report.manifest_path else "- Manifest: (missing)",
        "",
        "## Descriptor Gaps (descriptor.yaml but no .tflite or generated headers)",
        _list(report.descriptor_without_tflite),
        "",
        "## Manifest Entries Missing Descriptor Directory",
        _list(report.manifest_missing_descriptor),
        "",
        "## TFLite Missing Manifest Entry",
        _list(report.tflite_missing_manifest_entry),
        "",
        "## Operators Present in Descriptors but Missing in Manifest",
        _list(report.operators_zero_in_manifest),
        "",
        "## Allowlisted Descriptor Gaps",
        _list(report.allowlisted_descriptor_gaps),
        "",
        "## Unexpected Descriptor Gaps",
        _list(report.unexpected_descriptor_gaps),
        "",
        "## Allowlisted Zero-Manifest Operators",
        _list(report.allowlisted_zero_manifest_ops),
        "",
        "## Unexpected Zero-Manifest Operators",
        _list(report.unexpected_zero_manifest_ops),
        "",
    ]
    path.write_text("\n".join(content))
    return path


def run_gap_check(
    project_root: Path,
    cpu: str,
    report_dir: Optional[Path] = None,
    generated_tests_dir: Optional[Path] = None,
) -> Tuple[int, GapReport, Tuple[Path, Path]]:
    report = compute_gaps(project_root, cpu, generated_tests_dir=generated_tests_dir)
    out_dir = report_dir or report.report_dir
    json_path = _write_json(report, out_dir)
    md_path = _write_md(report, out_dir)

    unexpected = bool(report.unexpected_descriptor_gaps or report.unexpected_zero_manifest_ops)
    hard_fail = bool(report.manifest_missing_descriptor or report.tflite_missing_manifest_entry or report.manifest_missing)
    exit_code = 1 if (unexpected or hard_fail) else 0

    return exit_code, report, (json_path, md_path)
