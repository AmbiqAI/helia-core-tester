"""Phase 1 sidecar validation: confirm the structured JSON sidecar emitted
alongside each generated test case's .c/.h files is a faithful, in-sync
reflection of what was actually rendered -- not an independently re-derived
copy that could silently drift (the exact failure mode that motivated this
work: 22 hand-written regex extractors in generated_test_bridge.py each
re-parsing generated C source to recover kernel name/args/tolerance).

These tests exercise the real generation pipeline (OperationBase._write_op_outputs)
end-to-end for a couple of representative BasicMathFunctions cases and assert
the sidecar's kernel_fn/tolerance agree with what's baked into the generated
.c file's kernel call and HELIA_VALIDATE_OUTPUTS(...) invocation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _generate_one(tmp_path: Path, op: str, seed: int = 500) -> Path:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "helia_core_tester/generation/test_ops.py::test_generation",
            "-q",
            "--cpu",
            "cortex-m55",
            "--suite",
            "int",
            "--op",
            op,
            "--seed",
            str(seed),
            "--generated-tests-dir",
            str(tmp_path),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return tmp_path


def _find_case_dir(root: Path, family: str, case_name: str) -> Path:
    case_dir = root / family / case_name
    assert case_dir.is_dir(), f"expected {case_dir} to exist after generation"
    return case_dir


@pytest.mark.parametrize(
    ("op", "family", "case_name", "op_suffix"),
    [
        ("Add", "BasicMathFunctions", "add_default_s8", "add"),
        ("Add", "BasicMathFunctions", "add_default_s16", "add"),
    ],
)
def test_sidecar_kernel_fn_matches_generated_c_call(
    tmp_path: Path, op: str, family: str, case_name: str, op_suffix: str
) -> None:
    root = _generate_one(tmp_path, op)
    case_dir = _find_case_dir(root, family, case_name)
    sidecar = json.loads((case_dir / f"{case_name}_{op_suffix}.sidecar.json").read_text())
    c_source = (case_dir / f"{case_name}_{op_suffix}.c").read_text()

    kernel_fn = sidecar["kernel_fn"]
    assert kernel_fn, "sidecar must record the resolved kernel function name"
    assert f"= {kernel_fn}(" in c_source, (
        f"sidecar kernel_fn {kernel_fn!r} not found as the actual kernel call "
        f"site in the generated .c file -- sidecar has drifted from reality"
    )


@pytest.mark.parametrize(
    ("op", "family", "case_name", "op_suffix"),
    [
        ("Add", "BasicMathFunctions", "add_default_s8", "add"),
        ("Add", "BasicMathFunctions", "add_default_s16", "add"),
    ],
)
def test_sidecar_tolerance_matches_generated_validate_outputs_call(
    tmp_path: Path, op: str, family: str, case_name: str, op_suffix: str
) -> None:
    root = _generate_one(tmp_path, op)
    case_dir = _find_case_dir(root, family, case_name)
    sidecar = json.loads((case_dir / f"{case_name}_{op_suffix}.sidecar.json").read_text())
    c_source = (case_dir / f"{case_name}_{op_suffix}.c").read_text()

    sidecar_tolerance = sidecar["scalars"]["validation_tolerance"]

    # HELIA_VALIDATE_OUTPUTS(mode, actual, expected, size, tolerance, atol, rtol, max_reports, failures)
    match = re.search(
        r"HELIA_VALIDATE_OUTPUTS\(\s*\w+,.*?,.*?,.*?,\s*(-?\d+),",
        c_source,
        re.DOTALL,
    )
    assert match, "expected a HELIA_VALIDATE_OUTPUTS(...) call in the generated .c file"
    rendered_tolerance = int(match.group(1))

    assert sidecar_tolerance == rendered_tolerance, (
        f"sidecar tolerance {sidecar_tolerance} does not match the tolerance "
        f"actually baked into HELIA_VALIDATE_OUTPUTS(...) ({rendered_tolerance}) "
        f"-- sidecar has drifted from what was rendered"
    )


def test_sidecar_comparison_matches_hardware_bridge_resolve_comparison(tmp_path: Path) -> None:
    """The sidecar's `comparison` field is meant to be the same value the
    hardware perf-stream bridge manifest would compute via resolve_comparison()
    -- confirm this by recomputing it directly from the case's descriptor.yaml.
    """
    import yaml

    from helia_core_tester.generation.io.dtypes import resolve_comparison

    root = _generate_one(tmp_path, "Add")
    case_dir = _find_case_dir(root, "BasicMathFunctions", "add_default_s16")
    sidecar = json.loads((case_dir / "add_default_s16_add.sidecar.json").read_text())
    desc = yaml.safe_load((case_dir / "descriptor.yaml").read_text())

    expected_comparison = resolve_comparison(desc)
    assert sidecar["comparison"] == expected_comparison
