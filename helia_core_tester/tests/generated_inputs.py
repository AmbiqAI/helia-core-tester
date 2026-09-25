"""Skip a test only when the generated input it consumes is absent (#150).

`artifacts/` is gitignored and a working tree usually holds a partial corpus: whatever
families the last `generate` produced. A test that needs one generated case skips, naming
that case, when it was never generated. A case directory that is present but cannot be
discovered (no or unreadable descriptor.yaml) is a broken artifact and fails instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from helia_core_tester.hardware.generated_test_bridge import GeneratedTestCase, discover_generated_tests


def generated_family_dir(project_root: Path, *, suite: str, cpu: str, family: str) -> Path:
    return project_root / "artifacts" / "generated_tests" / suite / cpu / family


def discover_or_skip(
    project_root: Path,
    *,
    cpu: str = "cortex-m55",
    family: str = "ConvolutionFunctions",
    name_filter: str | None = None,
    limit: int | None = None,
    suite: str = "int",
) -> list[GeneratedTestCase]:
    """discover_generated_tests(), skipping when no matching case was generated."""
    root = generated_family_dir(project_root, suite=suite, cpu=cpu, family=family)
    try:
        cases = discover_generated_tests(
            project_root, cpu=cpu, family=family, name_filter=name_filter, limit=limit, suite=suite
        )
    except yaml.YAMLError as exc:
        pytest.fail(f"a generated case descriptor.yaml under {root} is unreadable: {exc}")
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
    matching = f" matching {name_filter!r}" if name_filter else ""
    pytest.skip(
        f"no generated {suite}/{cpu}/{family} case{matching} under artifacts/generated_tests "
        "(run `helia_core_tester generate` for it)"
    )
