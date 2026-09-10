"""Which route each float ReduceSum case takes, pinned against the kernel's own guards, and
the shipped goldens checked against the real interpreter rather than only against our own
reference. Reducing a spatial axis alone takes the channel-preserving spatial path added by
ns-cmsis-nn#488; reducing one together with the channel axis is the only thing left that
reaches the generic traversal (#131, #488)."""

import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor

TESTER_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTORS = (
    TESTER_ROOT / "assets" / "descriptors" / "BasicMathFunctions" / "reduce_sum_float.yaml"
)

# The spatial path claims these: batch and channel retained, a spatial axis reduced.
_SPATIAL_CASES = (
    "reduce_sum_float_axis_h_adapter_f32",
    "reduce_sum_float_axis_h_adapter_tail_f16",
    "reduce_sum_float_axis_w_c128_f32",
    "reduce_sum_float_axis_w_rows_tail_f32",
    "reduce_sum_float_axis_w_rows_tail_f16",
)
# Reducing a spatial axis together with the channel axis fails that guard and leaves no
# contiguous suffix, so these are the only cases in the family that reach the generic body.
_GENERIC_CASES = (
    "reduce_sum_float_axis_hc_f32",
    "reduce_sum_float_axis_hc_f16",
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


def _takes_spatial_path(in_dims, axis_arr):
    """Transcription of the channel-preserving spatial guard in ns-cmsis-nn
    Source/BasicMathFunctions/arm_reduce_sum_f32.c, whose float16 twin carries the same clauses
    in a different order. It is tried before the flatten helper, so a shape that satisfies it
    never reaches either the flatten or the generic body.

    Two groups of clauses are omitted deliberately. The overflow check needs more than 2^31
    elements, which no generated case can carry. The output-dimension agreement clauses are
    satisfied by construction, because build_reduce_output_dims emits exactly the dimensions the
    kernel expects for every descriptor, including the ones setting keepdims false."""
    reduces_spatial = bool(axis_arr[1] or axis_arr[2])
    return (
        not axis_arr[0]
        and not axis_arr[3]
        and reduces_spatial
        and all(d > 0 for d in in_dims)
    )


def _route_of(desc):
    axes = set(desc["axes"])
    axis_arr = [1 if i in axes else 0 for i in range(4)]
    if _takes_spatial_path(desc["input_shape"], axis_arr):
        return "spatial"
    return "flatten" if _flatten_suffix_start(desc["input_shape"], axis_arr) >= 0 else "generic"


def test_the_spatial_guard_needs_the_channel_axis_retained():
    # The distinguishing property, pinned so a future simplification is caught: reducing h
    # alone is spatial, reducing h with c is not.
    assert _takes_spatial_path([1, 3, 4, 5], [0, 1, 0, 0])
    assert not _takes_spatial_path([1, 3, 4, 5], [0, 1, 0, 1])


def test_the_transcribed_helper_folds_size_one_dimensions_in():
    # The distinguishing behaviour, pinned so a future simplification of the helper is caught.
    assert _flatten_suffix_start([1, 3, 4, 1], [0, 0, 1, 0]) == 2
    assert _flatten_suffix_start([1, 3, 4, 5], [0, 0, 1, 0]) == -1


@pytest.mark.parametrize("name", _GENERIC_CASES)
def test_the_mixed_axis_cases_take_the_generic_route(name):
    assert _route_of(_descriptors()[name]) == "generic"


@pytest.mark.parametrize("name", _SPATIAL_CASES)
def test_the_shape_cases_take_the_spatial_route(name):
    # They were written before that path existed, when these shapes fell to the generic body.
    assert _route_of(_descriptors()[name]) == "spatial"


def test_the_mixed_axis_pair_is_the_only_generic_cover_in_the_family():
    generic = {
        name
        for name, desc in _descriptors().items()
        if name.startswith("reduce_sum_float") and _route_of(desc) == "generic"
    }
    assert generic == set(_GENERIC_CASES)


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
