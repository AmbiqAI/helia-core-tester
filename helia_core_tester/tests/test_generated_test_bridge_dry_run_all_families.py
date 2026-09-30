"""Regression check: run build_case_bundle_from_generated_test() against
every real generated test case under artifacts/generated_tests/{int,float}/cortex-m55,
with require_fvp_pass=False (this sandbox has no FVP reports), and confirm every
case either bridges successfully or raises UnsupportedGeneratedTestError with a
reason (never an unexpected exception type). This is a host-side dry run of the
full generation -> bridge pipeline across every family/operator, not just the
per-builder unit tests.
"""

from pathlib import Path

import pytest

from helia_core_tester.hardware import generated_test_bridge as gtb
from helia_core_tester.tests.generated_inputs import discover_or_skip

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _all_families():
    params = []
    for suite in ("int", "float"):
        root = _PROJECT_ROOT / "artifacts/generated_tests" / suite / "cortex-m55"
        if root.is_dir():
            params += [(suite, p.name) for p in sorted(root.iterdir()) if p.is_dir()]
    return params


@pytest.mark.parametrize(("suite", "family"), _all_families())
def test_bridge_dry_run_over_all_generated_cases_in_family(tmp_path, suite, family):
    cases = discover_or_skip(_PROJECT_ROOT, cpu="cortex-m55", family=family, suite=suite)
    bridged = 0
    skipped = 0
    for case in cases:
        try:
            gtb.build_case_bundle_from_generated_test(
                _PROJECT_ROOT, case, output_root=tmp_path / family, require_fvp_pass=False
            )
            bridged += 1
        except gtb.UnsupportedGeneratedTestError:
            skipped += 1
    assert bridged + skipped == len(cases)
