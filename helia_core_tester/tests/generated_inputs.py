"""Find the generated input a test consumes, generating it when absent (#150).

`artifacts/` is gitignored and a working tree usually holds a partial corpus: whatever
families the last `generate` produced, and none at all in CI. A test that names its case
(`name_filter`) uses the on-disk copy when present; otherwise the matching descriptors are
generated into a per-session temp tree, so the test runs in CI too. A test that needs a
whole family, or names a case no descriptor matches, skips naming it. A case directory
that is present but cannot be discovered (no or unreadable descriptor.yaml) is a broken
artifact and fails instead.
"""

from __future__ import annotations

import atexit
import functools
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from helia_core_tester.core.config import Config
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test
from helia_core_tester.hardware.generated_test_bridge import GeneratedTestCase, discover_generated_tests

REPO_ROOT = Path(__file__).resolve().parents[2]


def generated_family_dir(project_root: Path, *, suite: str, cpu: str, family: str) -> Path:
    return project_root / "artifacts" / "generated_tests" / suite / cpu / family


def _first_unreadable_descriptor(root: Path) -> Path | str:
    for descriptor in sorted(root.glob("*/descriptor.yaml")):
        try:
            yaml.safe_load(descriptor.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            return descriptor
    return f"under {root}"


@functools.cache
def _session_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="hct-generated-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


@functools.cache
def _descriptors() -> tuple[dict, ...]:
    return tuple(load_all_descriptors(str(REPO_ROOT / "assets" / "descriptors")))


def generate_matching(*, cpu: str, family: str, name_filter: str, suite: str) -> Path:
    """Generate matching descriptors into the session tree; return its root."""
    root = _session_root()
    out_dir = root / "artifacts" / "generated_tests" / suite / cpu
    for desc in _descriptors():
        is_float = str(desc.get("_descriptor_suite", "")).lower() == "float"
        if desc["_family"] != family or is_float != (suite == "float") or name_filter not in desc["name"]:
            continue
        if not (out_dir / family / desc["name"] / "descriptor.yaml").is_file():
            generate_test(desc, str(out_dir), seed=Config.seed, cpu=cpu)
    return root


def discover_or_skip(
    project_root: Path,
    *,
    cpu: str = "cortex-m55",
    family: str = "ConvolutionFunctions",
    name_filter: str | None = None,
    limit: int | None = None,
    suite: str = "int",
) -> list[GeneratedTestCase]:
    """discover_generated_tests(), generating a named case absent on disk."""
    root = generated_family_dir(project_root, suite=suite, cpu=cpu, family=family)
    try:
        cases = discover_generated_tests(
            project_root, cpu=cpu, family=family, name_filter=name_filter, limit=limit, suite=suite
        )
    except yaml.YAMLError as exc:
        pytest.fail(f"generated case descriptor {_first_unreadable_descriptor(root)} is unreadable: {exc}")
    if cases:
        return cases
    present = sorted(
        directory.name
        for directory in (root.iterdir() if root.is_dir() else ())
        if directory.is_dir() and (name_filter is None or name_filter in directory.name)
    )
    if present:
        pytest.fail(
            f"generated case(s) {present} under {root} are present but not discoverable "
            "(missing or unreadable descriptor.yaml)"
        )
    if name_filter is not None:
        generated = generate_matching(cpu=cpu, family=family, name_filter=name_filter, suite=suite)
        cases = discover_generated_tests(generated, cpu=cpu, family=family, name_filter=name_filter, limit=limit, suite=suite)
        if cases:
            return cases
    matching = f" matching {name_filter!r}" if name_filter else ""
    pytest.skip(
        f"no generated {suite}/{cpu}/{family} case{matching} under artifacts/generated_tests "
        "(no descriptor matches, or run `helia_core_tester generate`)"
    )
