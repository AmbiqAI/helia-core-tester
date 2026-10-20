"""The int8 bilinear resize golden: the C reference's resize_bilinear_s8 (TFLite's ResizeBilinearInteger)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from helia_core_tester.generation.ops.ReshapeFunctions.resize_bilinear import OpResizeBilinear
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings

_DESCRIPTORS = Path(__file__).resolve().parents[2] / "assets" / "descriptors" / "ReshapeFunctions" / "resize_bilinear.yaml"


def _resize(x, size, align=False, half=False):
    return get_bindings().run(
        "resize_bilinear_s8", {"align_corners": int(align), "half_pixel_centers": int(half)},
        {"input": np.ascontiguousarray(x, dtype=np.int8)},
        {"output": (x.shape[0], size[0], size[1], x.shape[3])})["output"]


def _axis(value, scale, half, size):
    scaled = value * scale + scale // 2 - (1 << 9) if half else value * scale
    lo = max(int(scaled / (1 << 10)), 0)
    hi = min(int((scaled + (1 << 10) - 1) / (1 << 10)), size - 1)
    return scaled, lo, hi


def _scale(in_size, out_size, align):
    if align and out_size > 1:
        return ((1 << 10) * (in_size - 1) + (out_size - 1) // 2) // (out_size - 1)
    return ((1 << 10) * in_size + out_size // 2) // out_size


def _model(x, size, align, half):
    """TFLite reference_ops::ResizeBilinearInteger, element by element (C division truncates toward zero)."""
    n, ih, iw, c = x.shape
    oh, ow = size
    sh, sw = _scale(ih, oh, align), _scale(iw, ow, align)
    out = np.zeros((n, oh, ow, c), np.int8)
    for b in range(n):
        for y in range(oh):
            ys, y0, y1 = _axis(y, sh, half, ih)
            fy = ys - (1 << 10) * y0
            for xo in range(ow):
                xs, x0, x1 = _axis(xo, sw, half, iw)
                fx = xs - (1 << 10) * x0
                for ch in range(c):
                    total = (int(x[b, y0, x0, ch]) * ((1 << 10) - fy) * ((1 << 10) - fx)
                             + int(x[b, y1, x0, ch]) * fy * ((1 << 10) - fx)
                             + int(x[b, y0, x1, ch]) * ((1 << 10) - fy) * fx
                             + int(x[b, y1, x1, ch]) * fy * fx)
                    rounding = (1 << 19) if total > 0 else -(1 << 19)
                    out[b, y, xo, ch] = int((total + rounding) / (1 << 20))
    return out


@pytest.mark.parametrize(
    "align,half,expected",
    [
        # The values LiteRT's builtin reference kernel gave for this input before the C reference.
        (False, False, [0, 6, 12, 12]),
        (False, True, [0, 3, 9, 12]),
        (True, False, [0, 4, 8, 12]),
    ],
)
def test_coordinate_options_match_the_litert_values(align, half, expected):
    x = np.array([0, 12], dtype=np.int8).reshape(1, 1, 2, 1)
    np.testing.assert_array_equal(_resize(x, (1, 4), align, half).reshape(-1), expected)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("align,half", [(False, False), (False, True), (True, False)])
def test_reference_matches_the_integer_model(seed, align, half):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(1, 3)), int(rng.integers(1, 6)), int(rng.integers(1, 6)), int(rng.integers(1, 4)))
    size = (int(rng.integers(1, 11)), int(rng.integers(1, 11)))
    x = rng.integers(-128, 128, shape, dtype=np.int8)
    np.testing.assert_array_equal(_resize(x, size, align, half), _model(x, size, align, half))


@pytest.mark.parametrize("value", [-128, 127])
def test_a_constant_input_resizes_to_itself_at_the_dtype_limits(value):
    x = np.full((1, 3, 2, 2), value, np.int8)
    for align, half in ((False, False), (False, True), (True, False)):
        assert np.all(_resize(x, (7, 5), align, half) == value)


@pytest.mark.parametrize(
    "params",
    [{"align_corners": 1, "half_pixel_centers": 1}, {"align_corners": 2, "half_pixel_centers": 0}],
)
def test_reference_rejects_invalid_options(params):
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        get_bindings().run("resize_bilinear_s8", params, {"input": np.zeros((1, 2, 2, 1), np.int8)},
                           {"output": (1, 4, 4, 1)})


def test_reference_rejects_a_channel_count_change():
    with pytest.raises(ReferenceKernelError, match="E_SHAPE"):
        get_bindings().run("resize_bilinear_s8", {"align_corners": 0, "half_pixel_centers": 0},
                           {"input": np.zeros((1, 2, 2, 3), np.int8)}, {"output": (1, 4, 4, 2)})


def _op(desc):
    return OpResizeBilinear(dict(desc), seed=500, target_cpu="cortex-m55")


@pytest.mark.parametrize(
    "desc,match",
    [
        ({"activation_dtype": "S16"}, "require S8"),
        ({"hint": {"kernel": "arm_resize_bilinear_s16"}}, "unsupported bilinear entry"),
        ({"size": [0, 4]}, "size must be two positive dims"),
        ({"input_shape": [1, 2, 2]}, "input_shape must be four positive dims"),
        ({"align_corners": True, "half_pixel_centers": True}, "cannot both be set"),
    ],
)
def test_bilinear_rejects_invalid_descriptors_before_generating(tmp_path, desc, match):
    base = {"name": "rb", "operator": "ResizeBilinear", "input_shape": [1, 1, 2, 1], "size": [1, 4]}
    with pytest.raises(ValueError, match=match):
        _op({**base, **desc}).generate_c_files(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("desc", [d for d in yaml.safe_load_all(_DESCRIPTORS.read_text()) if d],
                         ids=lambda d: d["name"])
def test_descriptor_golden_is_the_recorded_reference_call(tmp_path, desc):
    op = _op(desc)
    op.generate_c_files(tmp_path)
    call = op.reference
    assert call is not None and call.entry == "resize_bilinear_s8"
    x = call.inputs["input"]
    want = _model(x, desc["size"], bool(desc.get("align_corners")), bool(desc.get("half_pixel_centers")))
    np.testing.assert_array_equal(call.output(), want)
    assert not list(tmp_path.rglob("*.tflite"))
