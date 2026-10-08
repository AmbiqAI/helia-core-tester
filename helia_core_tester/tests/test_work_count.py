from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.hardware.case_bundle import BlobInfo, CaseBundle
from helia_core_tester.hardware.work_count import case_work, per_unit


def _bundle(operator: str, scalars: dict | None = None, **shapes: tuple[int, ...]) -> CaseBundle:
    blobs = tuple(
        BlobInfo(index, role, "S8", len(dims), dims, 0, 1, 0, "", Path(role))
        for index, (role, dims) in enumerate(shapes.items(), start=1)
    )
    manifest = {"operator": operator, "serialized_scalar_parameters": scalars or {}}
    return CaseBundle(Path("."), Path("case_manifest.json"), manifest, blobs)


@pytest.mark.parametrize(
    ("operator", "scalars", "shapes", "macs"),
    [
        # 3x3 conv, 2 in, 4 out, 3x6 output.
        ("Convolve", None, {"input_0": (1, 3, 6, 2), "weights": (3, 3, 2, 4), "expected_output": (1, 3, 6, 4)}, 3 * 6 * 4 * 9 * 2),
        # Two groups halve the weights' input depth.
        ("Convolve", None, {"input_0": (1, 4, 4, 4), "weights": (3, 3, 2, 4), "expected_output": (1, 4, 4, 4)}, 16 * 4 * 9 * 2),
        # Depthwise, multiplier 2.
        ("DepthwiseConv", None, {"input_0": (1, 8, 8, 3), "weights": (1, 3, 3, 6), "expected_output": (1, 8, 8, 6)}, 64 * 6 * 9),
        ("FullyConnected", None, {"input_0": (3, 20), "weights": (6, 20), "expected_output": (3, 6)}, 3 * 6 * 20),
        # Each input element scatters a 3x3x3 kernel.
        ("TransposeConv", None, {"input_0": (1, 4, 4, 2), "weights": (3, 3, 3, 2), "expected_output": (1, 8, 8, 3)}, 32 * 27),
        ("BatchMatMul", None, {"input_0": (1, 1, 4, 3), "input_1": (1, 1, 2, 3), "expected_output": (1, 1, 4, 2)}, 4 * 2 * 3),
        # adj_x moves K to the rows.
        ("BatchMatMul", {"adj_x": 1}, {"input_0": (1, 1, 3, 4), "input_1": (1, 1, 3, 5), "expected_output": (1, 1, 4, 5)}, 4 * 5 * 3),
    ],
)
def test_mac_kernels_count_dense_macs(operator: str, scalars: dict | None, shapes: dict, macs: int) -> None:
    assert case_work(_bundle(operator, scalars, **shapes)) == {"macs": macs, "ops": 2 * macs}


def test_other_kernels_count_ops() -> None:
    pool = _bundle("AvgPool", {"pool_h": 3, "pool_w": 1}, input_0=(1, 20, 1, 21), expected_output=(1, 7, 1, 21))
    assert case_work(pool) == {"macs": None, "ops": 7 * 21 * 3}
    assert case_work(_bundle("ReduceSum", input_0=(1, 4, 5, 6), expected_output=(1, 1, 1, 6))) == {"macs": None, "ops": 120}
    assert case_work(_bundle("Add", input_0=(1, 2, 3, 4), input_1=(1, 1, 1, 4), expected_output=(1, 2, 3, 4))) == {"macs": None, "ops": 24}
    # Data movement has no op count.
    assert case_work(_bundle("Transpose", input_0=(1, 2, 3, 4), expected_output=(1, 4, 3, 2))) == {"macs": None, "ops": None}


def test_per_unit_skips_missing_work() -> None:
    assert per_unit(100.0, 400) == 0.25
    assert per_unit(100.0, None) is None and per_unit(100.0, 0) is None and per_unit(None, 4) is None
