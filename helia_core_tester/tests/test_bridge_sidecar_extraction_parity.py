"""Phase 3 of the generation/bridge unification plan: verify that the
BasicMathFunctions bridge builders' sidecar-based scalar extraction path
(reading structured JSON emitted by `_write_op_outputs()`) produces
byte-identical `serialized_scalar_parameters` to the legacy regex-based
extraction path that parses generated `.c` source directly.

Every migrated builder below falls back to the regex path when no sidecar is
present (for op-generator files not yet routed through `_write_op_outputs()`),
so this test protects that migration invariant: whichever path runs, the
bridged case manifest must be identical. If this test ever fails after adding
a new BasicMathFunctions op or changing a builder, it usually means the
sidecar's `scalars` dict and the positional regex extraction have drifted out
of sync -- treat that as a real bug, not a flaky test.
"""

from pathlib import Path

import pytest
import yaml

from helia_core_tester.perf_stream import generated_test_bridge as gtb

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GEN_ROOT = _PROJECT_ROOT / "artifacts/generated_tests/int/cortex-m55"

_CASES = [
    ("BasicMathFunctions", "abs_rescale_s8"),
    ("BasicMathFunctions", "abs_default_s16"),
    ("BasicMathFunctions", "add_default_s8"),
    ("BasicMathFunctions", "add_default_s16"),
    ("BasicMathFunctions", "sub_row_scalar1_s8"),
    ("BasicMathFunctions", "mul_default_s8"),
    ("BasicMathFunctions", "mul_default_s16"),
    ("BasicMathFunctions", "squared_difference_ident_s8"),
    ("BasicMathFunctions", "squared_difference_ident_s16"),
    ("BasicMathFunctions", "squared_difference_scalar_input1_s8"),
    ("BasicMathFunctions", "argmax_axis2_s16"),
    ("BasicMathFunctions", "mean_channels_case_02_s16"),
    ("BasicMathFunctions", "reduce_max_batch_s16"),
]


@pytest.mark.parametrize("family, op_name", _CASES)
def test_sidecar_and_regex_extraction_paths_agree(tmp_path, family, op_name):
    """For a real generated artifact with a sidecar, temporarily hide the
    sidecar and confirm the resulting CaseBundle's scalar parameters are
    identical to the sidecar-present run."""
    case_dir = _GEN_ROOT / family / op_name
    if not case_dir.is_dir():
        pytest.skip(f"{case_dir} not present in this local artifacts tree (run generation first)")
    desc = yaml.safe_load((case_dir / "descriptor.yaml").read_text())
    generated_test = gtb.GeneratedTestCase(
        name=op_name, cpu="cortex-m55", family=family, directory=case_dir, descriptor=desc
    )
    builder = gtb._BUILDERS[(family, desc["operator"])]

    bundle_with_sidecar = builder(_PROJECT_ROOT, generated_test, output_root=tmp_path / "with_sidecar")

    sidecar_paths = list(case_dir.glob("*.sidecar.json"))
    assert sidecar_paths, f"{op_name}: expected a sidecar to exist for this migrated op"
    renamed = []
    for sidecar_path in sidecar_paths:
        sidecar_path.rename(sidecar_path.with_suffix(".bak"))
        renamed.append(sidecar_path)
    try:
        bundle_without_sidecar = builder(_PROJECT_ROOT, generated_test, output_root=tmp_path / "without_sidecar")
    finally:
        for sidecar_path in renamed:
            sidecar_path.with_suffix(".bak").rename(sidecar_path)

    assert (
        bundle_with_sidecar.manifest["serialized_scalar_parameters"]
        == bundle_without_sidecar.manifest["serialized_scalar_parameters"]
    )
    assert bundle_with_sidecar.manifest["kernel_id"] == bundle_without_sidecar.manifest["kernel_id"]
