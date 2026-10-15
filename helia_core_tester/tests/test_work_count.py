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


# LSTM blobs are flat: input B*T*I, weights 4*H*(I+H), output B*T*H.
def _lstm(batch: int, steps: int, inputs: int, hidden: int) -> CaseBundle:
    return _bundle(
        "LSTMUnidirectional", input_0=(batch * steps * inputs,), weights=(4 * hidden * (inputs + hidden),),
        bias=(4 * hidden,), expected_output=(batch * steps * hidden,),
    )


@pytest.mark.parametrize(
    ("batch", "steps", "inputs", "hidden"),
    [(1, 1, 1, 1), (1, 2, 3, 2), (2, 25, 10, 5), (3, 7, 1, 16), (1, 1, 32, 1), (4, 9, 13, 13)],
)
def test_lstm_counts_four_gates_per_step(batch: int, steps: int, inputs: int, hidden: int) -> None:
    macs = batch * steps * 4 * hidden * (inputs + hidden)
    assert case_work(_lstm(batch, steps, inputs, hidden)) == {"macs": macs, "ops": 2 * macs}


@pytest.mark.parametrize(
    "shapes",
    [
        {"weights": (40,), "expected_output": (4,)},
        {"input_0": (6,), "expected_output": (4,)},
        {"input_0": (6,), "weights": (40,)},
        # Weights not 4*H*(I+H) for any H.
        {"input_0": (6,), "weights": (41,), "expected_output": (4,)},
        {"input_0": (6,), "weights": (40,), "expected_output": (5,)},
        {"input_0": (6,), "weights": (0,), "expected_output": (4,)},
        {"input_0": (0,), "weights": (40,), "expected_output": (4,)},
    ],
)
def test_lstm_without_consistent_shapes_has_no_macs(shapes: dict) -> None:
    assert case_work(_bundle("LSTMUnidirectional", **shapes)) == {"macs": None, "ops": None}


def test_svdf_counts_feature_and_time_products() -> None:
    # 2 batches, 4 inputs, 6 filters, 3 time steps.
    bundle = _bundle("SVDF", input_0=(2, 4, 1, 1), input_1=(2, 6, 3, 1), input_2=(6, 3, 1, 1),
                     weights=(6, 4, 1, 1), expected_output=(2, 3, 1, 1))
    assert case_work(bundle) == {"macs": 2 * 6 * (4 + 3), "ops": 2 * 2 * 6 * (4 + 3)}


@pytest.mark.parametrize(
    "shapes",
    [
        {"input_2": (6, 3), "weights": (6, 4)},
        {"input_0": (2, 4), "weights": (6, 4)},
        {"input_0": (2, 4), "input_2": (6, 3)},
        {"input_0": (8,), "input_2": (6, 3), "weights": (6, 4)},
        # Feature width must match the input, filters must match the time weights.
        {"input_0": (2, 5), "input_2": (6, 3), "weights": (6, 4)},
        {"input_0": (2, 4), "input_2": (5, 3), "weights": (6, 4)},
        {"input_0": (0, 4), "input_2": (6, 3), "weights": (6, 4)},
        {"input_0": (2, 4), "input_2": (6, 0), "weights": (6, 4)},
    ],
)
def test_svdf_without_consistent_shapes_has_no_macs(shapes: dict) -> None:
    assert case_work(_bundle("SVDF", expected_output=(2, 3), **shapes)) == {"macs": None, "ops": None}


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
