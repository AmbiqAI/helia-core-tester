"""Phase 6 regression check: run build_case_bundle_from_generated_test() against
every real generated test case under artifacts/generated_tests/int/cortex-m55,
with require_fvp_pass=False (this sandbox has no FVP reports), and confirm every
case either bridges successfully or raises UnsupportedGeneratedTestError with a
reason (never an unexpected exception type). This is a host-side dry run of the
full generation -> bridge pipeline across every family/operator, not just the
per-builder unit tests.
"""

from pathlib import Path

import pytest

from helia_core_tester.perf_stream import generated_test_bridge as gtb

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _all_families():
    root = _PROJECT_ROOT / "artifacts/generated_tests/int/cortex-m55"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


@pytest.mark.parametrize("family", _all_families())
def test_bridge_dry_run_over_all_generated_cases_in_family(tmp_path, family):
    cases = gtb.discover_generated_tests(_PROJECT_ROOT, cpu="cortex-m55", family=family)
    if not cases:
        pytest.skip(f"No generated cases found for family {family}")
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
