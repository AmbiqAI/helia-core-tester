"""The width-only ReduceSum cases take the kernel's generic route, and the goldens the harness
ships for them agree with the real interpreter rather than only with our own reference (#131)."""

import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor

TESTER_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTORS = (
    TESTER_ROOT / "assets" / "descriptors" / "BasicMathFunctions" / "reduce_sum_float.yaml"
)

_NEW_CASES = (
    "reduce_sum_float_axis_h_adapter_f32",
    "reduce_sum_float_axis_h_adapter_tail_f16",
    "reduce_sum_float_axis_w_c128_f32",
    "reduce_sum_float_axis_w_rows_tail_f32",
    "reduce_sum_float_axis_w_rows_tail_f16",
)
_SUFFIX_CONTROLS = ("reduce_sum_float_axis_c_f32", "reduce_sum_float_axis_hwc_f32")


def _descriptors():
    return {desc["name"]: desc for desc in load_descriptor(str(DESCRIPTORS))}


def _flatten_suffix_start(in_dims, axis_arr):
    """Transcription of arm_reduce_get_flatten_suffix_start_from_arrays in ns-cmsis-nn
    Include/arm_nnsupportfunctions.h. A size-one dimension counts as reduced, which is why this
    cannot be simplified to "the reduced axes form a contiguous suffix": [1, 3, 4, 1] reducing
    only w takes the flatten helper, and that shorthand would say otherwise."""
    axis_mask = (
        (axis_arr[0] & 1) << 3 | (axis_arr[1] & 1) << 2 | (axis_arr[2] & 1) << 1 | (axis_arr[3] & 1)
    )
    input_mask = (
        int(in_dims[0] == 1) << 3
        | int(in_dims[1] == 1) << 2
        | int(in_dims[2] == 1) << 1
        | int(in_dims[3] == 1)
    )
    union_mask = axis_mask | input_mask
    for bit, start, keep in ((0x8, 0, 0xF), (0x4, 1, 0x7), (0x2, 2, 0x3), (0x1, 3, 0x1)):
        if axis_mask & bit:
            return start if (union_mask & keep) == keep else -1
    return -1


def _route_of(desc):
    axes = set(desc["axes"])
    axis_arr = [1 if i in axes else 0 for i in range(4)]
    return "flatten" if _flatten_suffix_start(desc["input_shape"], axis_arr) >= 0 else "generic"


def test_the_transcribed_helper_folds_size_one_dimensions_in():
    # The distinguishing behaviour, pinned so a future simplification of the helper is caught.
    assert _flatten_suffix_start([1, 3, 4, 1], [0, 0, 1, 0]) == 2
    assert _flatten_suffix_start([1, 3, 4, 5], [0, 0, 1, 0]) == -1


@pytest.mark.parametrize("name", _NEW_CASES)
def test_new_cases_take_the_generic_route(name):
    assert _route_of(_descriptors()[name]) == "generic"


@pytest.mark.parametrize("name", _SUFFIX_CONTROLS)
def test_the_suffix_controls_still_take_the_flatten_route(name):
    assert _route_of(_descriptors()[name]) == "flatten"


def test_the_shape_the_runtime_actually_passes_is_covered():
    # The model reduces the sequence axis of [1, 64, 128]. heliaRT pads a rank-3 shape at the
    # end, so the kernel is called with [1, 64, 128, 1] reducing h and retaining w, not with a
    # width reduction. A route guard written for one axis does not fire for the other, so the
    # literal layout is pinned here rather than the arithmetic alone.
    desc = _descriptors()["reduce_sum_float_axis_h_adapter_f32"]
    assert desc["input_shape"] == [1, 64, 128, 1]
    assert desc["axes"] == [1]


def test_the_width_layout_is_kept_as_an_equivalent_control():
    desc = _descriptors()["reduce_sum_float_axis_w_c128_f32"]
    assert desc["input_shape"] == [1, 1, 64, 128]
    assert desc["axes"] == [2]


def _c_floats(source: str, symbol: str) -> np.ndarray:
    body = source.split(f"{symbol}[]")[1].split("{", 1)[1].split("};")[0]
    values = re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", body)
    return np.array([float(v) for v in values], dtype=np.float32)


@pytest.mark.parametrize(
    "name",
    (
        "reduce_sum_float_axis_h_adapter_f32",
        "reduce_sum_float_axis_w_c128_f32",
        "reduce_sum_float_axis_w_rows_tail_f32",
    ),
)
def test_shipped_golden_agrees_with_the_real_interpreter(tmp_path, name):
    """Runs the generated model through LiteRT on the generated input and compares against the
    generated golden, so what is pinned is the artefact the suite ships rather than a second
    statement of the same formula. That only holds with the seed the suite itself generates
    with, hence default_seed_for_case. Float32 only: the emitted model is float32 while an f16
    golden is rounded to half, so the two are not directly comparable."""
    pytest.importorskip("ai_edge_litert.interpreter")
    from helia_core_tester.generation.ops.BasicMathFunctions.reduce_sum import OpReduceSum
    from helia_core_tester.generation.test_ops import default_seed_for_case
    from helia_core_tester.generation.utils.litert_utils import run_inference_litert

    desc = _descriptors()[name]
    op = OpReduceSum(desc, default_seed_for_case(name))
    model_path = tmp_path / f"{name}.tflite"
    op.convert_to_tflite(op.build_keras_model(), str(model_path), 0)
    op.generate_c_files(tmp_path)

    header = next(tmp_path.rglob(f"{name}_reduce_sum.h")).read_text()
    inputs = _c_floats(header, f"{name}_input").reshape(desc["input_shape"])
    golden = _c_floats(header, f"{name}_expected_output")

    actual = np.asarray(run_inference_litert(str(model_path), inputs)).reshape(-1)
    assert actual.size == golden.size
    assert actual == pytest.approx(golden, abs=5e-5, rel=2e-5)
