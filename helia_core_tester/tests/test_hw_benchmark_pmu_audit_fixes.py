"""
Regression tests for the .codex-audit remediation pass on feature/hw-benchmark-pmu.

Covers:
  F003 - depthwise_conv.yaml zero-sized filter multiplier fixed to 1, plus
         positive-dimension / channel-agreement validation for DepthwiseConv.
  F004 - S4 "no_bias" depthwise descriptors actually declare use_bias: false.
  F009 - NNActivationS16._simulate_activation() writes through a real view
         instead of a throwaway copy from .flatten().
  F010 - load_all_descriptors() fails atomically (DescriptorLoadError) unless
         best_effort=True is explicitly requested.
  F012 - setup_dependencies.download_file() requires and verifies a pinned
         SHA-256 digest and fails closed on mismatch.
  F014 - depthwise_conv.py no longer has hard-coded agent-debug side effects.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import (
    DescriptorLoadError,
    load_all_descriptors,
    load_descriptor,
)
from helia_core_tester.generation.ops.ActivationFunctions.nn_activation_s16 import (
    OpNNActivationS16,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _depthwise_descriptors():
    path = _repo_root() / "assets" / "descriptors" / "ConvolutionFunctions" / "depthwise_conv.yaml"
    return {d["name"]: d for d in load_descriptor(str(path))}


# ---------------------------------------------------------------------------
# F003
# ---------------------------------------------------------------------------

def test_f003_depthwise_fast_bias_s16_has_nonzero_multiplier():
    descs = _depthwise_descriptors()
    desc = descs["depthwise_conv_fast_test_bias_s16"]
    assert desc["filter_shape"][-1] == 1, "depth multiplier must be positive (was 0)"
    assert all(d > 0 for d in desc["filter_shape"])


def test_f003_depthwise_conv_rejects_non_positive_filter_dims():
    bad_desc = {
        "operator": "DepthwiseConv",
        "name": "depthwise_conv_bad_zero_multiplier",
        "activation_dtype": "S16",
        "weight_dtype": "S8",
        "activation": "NONE",
        "hint": {"call_style": "per_tensor"},
        "input_shape": [1, 4, 4, 8],
        "filter_shape": [2, 2, 8, 0],
        "strides": [1, 1],
        "padding": "VALID",
        "use_bias": True,
    }
    from helia_core_tester.generation.io.descriptors import _validate_and_normalize_descriptor

    with pytest.raises(ValueError, match="non-positive"):
        _validate_and_normalize_descriptor(bad_desc)


def test_f003_depthwise_conv_rejects_channel_mismatch():
    bad_desc = {
        "operator": "DepthwiseConv",
        "name": "depthwise_conv_bad_channel_mismatch",
        "activation_dtype": "S16",
        "weight_dtype": "S8",
        "activation": "NONE",
        "hint": {"call_style": "per_tensor"},
        "input_shape": [1, 4, 4, 8],
        "filter_shape": [2, 2, 4, 1],  # filter channel dim (4) != input channels (8)
        "strides": [1, 1],
        "padding": "VALID",
        "use_bias": True,
    }
    from helia_core_tester.generation.io.descriptors import _validate_and_normalize_descriptor

    with pytest.raises(ValueError, match="input-channel dimension"):
        _validate_and_normalize_descriptor(bad_desc)


# ---------------------------------------------------------------------------
# F004
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "depthwise_conv_eveninch_evenmult2_no_bias_s4",
        "depthwise_conv_oddinch_mult1_no_bias_s4",
    ],
)
def test_f004_s4_no_bias_descriptors_disable_bias(name):
    descs = _depthwise_descriptors()
    assert descs[name]["use_bias"] is False


# ---------------------------------------------------------------------------
# F009
# ---------------------------------------------------------------------------

def test_f009_activation_s16_writes_through_view_not_copy():
    op = OpNNActivationS16.__new__(OpNNActivationS16)
    # Avoid depending on the real sigmoid table file location.
    op._load_sigmoid_table = lambda: list(range(256))

    input_data = np.array([[100, -100, 0, 3000, -3000]], dtype=np.int32)
    out = op._simulate_activation(input_data, left_shift=0, act_type="SIGMOID")

    assert out.dtype == np.int16
    assert out.shape == input_data.shape
    # Prior to the fix, out.flatten() returned a copy, so `out` itself was
    # never written and stayed as np.empty_like garbage; a real bug fix must
    # produce a result that is not trivially uniform for varied inputs.
    assert len(set(out.flatten().tolist())) > 1


def test_f009_activation_s16_deterministic_across_calls():
    op = OpNNActivationS16.__new__(OpNNActivationS16)
    op._load_sigmoid_table = lambda: list(range(256))
    input_data = np.array([[1, 2, 3, -4, -5]], dtype=np.int32)

    out1 = op._simulate_activation(input_data, left_shift=0, act_type="TANH")
    out2 = op._simulate_activation(input_data, left_shift=0, act_type="TANH")
    assert np.array_equal(out1, out2)


# ---------------------------------------------------------------------------
# F010
# ---------------------------------------------------------------------------

def test_f010_load_all_descriptors_strict_by_default_raises_aggregate_error(tmp_path):
    (tmp_path / "BasicMathFunctions").mkdir()
    good = tmp_path / "BasicMathFunctions" / "abs.yaml"
    good.write_text(
        "operator: Abs\n"
        "name: abs_ok_s8\n"
        "activation_dtype: S8\n"
        "weight_dtype: S8\n"
        "input_shape: [1, 4]\n"
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text("operator: DepthwiseConv\n")  # missing required fields -> load error

    with pytest.raises(DescriptorLoadError) as exc_info:
        load_all_descriptors(str(tmp_path))

    failures = exc_info.value.failures
    assert len(failures) == 1
    assert str(bad) in failures[0][0]


def test_f010_load_all_descriptors_best_effort_returns_partial_list(tmp_path):
    (tmp_path / "BasicMathFunctions").mkdir()
    good = tmp_path / "BasicMathFunctions" / "abs.yaml"
    good.write_text(
        "operator: Abs\n"
        "name: abs_ok_s8\n"
        "activation_dtype: S8\n"
        "weight_dtype: S8\n"
        "input_shape: [1, 4]\n"
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text("operator: DepthwiseConv\n")

    descriptors = load_all_descriptors(str(tmp_path), best_effort=True)
    names = {d["name"] for d in descriptors}
    assert "abs_ok_s8" in names
    assert len(descriptors) == 1


# ---------------------------------------------------------------------------
# F012
# ---------------------------------------------------------------------------

def test_f012_download_file_rejects_checksum_mismatch(tmp_path):
    from helia_core_tester.scripts.setup_dependencies import (
        ChecksumMismatchError,
        download_file,
    )

    fake_bytes = b"not the expected content"

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            nonlocal fake_bytes
            chunk, fake_bytes = fake_bytes[:n] if n > 0 else fake_bytes, fake_bytes[n:] if n > 0 else b""
            return chunk

    dest = tmp_path / "download.bin"
    wrong_sha256 = "0" * 64

    with patch("urllib.request.urlopen", return_value=_FakeResponse()):
        with pytest.raises(ChecksumMismatchError):
            download_file("https://example.invalid/file", dest, "test file", wrong_sha256)

    assert not dest.exists(), "mismatched download must be deleted, not left on disk"


def test_f012_download_file_accepts_matching_checksum(tmp_path):
    import hashlib

    from helia_core_tester.scripts.setup_dependencies import download_file

    payload = b"hello world"
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    class _FakeResponse:
        def __init__(self, data: bytes):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            if n <= 0:
                chunk, self._data = self._data, b""
                return chunk
            chunk, self._data = self._data[:n], self._data[n:]
            return chunk

    dest = tmp_path / "download.bin"
    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        download_file("https://example.invalid/file", dest, "test file", expected_sha256)

    assert dest.read_bytes() == payload


def test_f012_pinned_hashes_exist_for_both_architectures():
    from helia_core_tester.scripts.setup_dependencies import DEFAULT_GCC_VERSION, PINNED_SHA256

    for arch in ("x86_64", "aarch64"):
        for key in (("arm_gcc", DEFAULT_GCC_VERSION, arch), ("corstone300", arch)):
            digest = PINNED_SHA256[key]
            assert len(digest) == 64
            int(digest, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# F014
# ---------------------------------------------------------------------------

def test_f014_depthwise_conv_generation_has_no_debug_log_side_effect():
    source = (
        _repo_root()
        / "helia_core_tester"
        / "generation"
        / "ops"
        / "ConvolutionFunctions"
        / "depthwise_conv.py"
    ).read_text()

    assert "debug.log" not in source
    assert "workspaces/cmsis-aot-tester" not in source
    assert "agent log" not in source


# ---------------------------------------------------------------------------
# F011 (dismissed as a false positive; the existing half-pixel formula
# matches CMSIS-NN's GetNearestNeighbor / TFLite reference semantics, which
# intentionally omit the -0.5 correction used only for bilinear resize).
# These tests pin the current, correct behavior with known-index vectors so
# any future regression toward the incorrect "-0.5 offset" formula is caught.
# ---------------------------------------------------------------------------

from helia_core_tester.generation.ops.ReshapeFunctions.resize_nearest_neighbor import (
    OpResizeNearestNeighbor,
)


def test_f011_half_pixel_upsampling_known_indices():
    # in_size=2, out_size=4, half_pixel_centers=True: scale = 2/4 = 0.5
    # scaled = (out_idx + 0.5) * 0.5 -> floor
    # out_idx: 0 -> 0.25 -> 0 | 1 -> 0.75 -> 0 | 2 -> 1.25 -> 1 | 3 -> 1.75 -> 1
    expected = [0, 0, 1, 1]
    actual = [
        OpResizeNearestNeighbor._nearest_index(i, in_size=2, out_size=4, align_corners=False, half_pixel_centers=True)
        for i in range(4)
    ]
    assert actual == expected


def test_f011_half_pixel_downsampling_known_indices():
    # in_size=4, out_size=2, half_pixel_centers=True: scale = 4/2 = 2.0
    # out_idx: 0 -> (0+0.5)*2=1.0 -> floor 1 | 1 -> (1+0.5)*2=3.0 -> floor 3
    expected = [1, 3]
    actual = [
        OpResizeNearestNeighbor._nearest_index(i, in_size=4, out_size=2, align_corners=False, half_pixel_centers=True)
        for i in range(2)
    ]
    assert actual == expected


def test_f011_align_corners_known_indices():
    # in_size=3, out_size=5, align_corners=True: scale = (3-1)/(5-1) = 0.5
    # scaled = out_idx * 0.5, rounded: 0,0.5->0 or 1(banker's/np.round->0),1,1.5->2,2
    expected = [
        int(np.round(i * 0.5)) for i in range(5)
    ]
    actual = [
        OpResizeNearestNeighbor._nearest_index(i, in_size=3, out_size=5, align_corners=True, half_pixel_centers=False)
        for i in range(5)
    ]
    assert actual == expected


def test_f011_no_half_pixel_no_align_corners_known_indices():
    # in_size=4, out_size=2, plain nearest (no half-pixel, no align-corners): scale = 4/2 = 2.0
    # scaled = out_idx * 2.0 -> floor: 0 -> 0, 1 -> 2
    expected = [0, 2]
    actual = [
        OpResizeNearestNeighbor._nearest_index(i, in_size=4, out_size=2, align_corners=False, half_pixel_centers=False)
        for i in range(2)
    ]
    assert actual == expected


def test_f011_half_pixel_formula_omits_bilinear_style_minus_half_offset():
    # Regression guard: the formula must remain (out_idx + 0.5) * scale with NO
    # subsequent "-0.5" term. Introducing that offset (as bilinear resize does)
    # would be the *wrong* fix for nearest-neighbor and was the original,
    # incorrect audit suggestion for F011.
    in_size, out_size = 8, 4
    scale = in_size / out_size
    for out_idx in range(out_size):
        correct = int(np.floor((out_idx + 0.5) * scale))
        incorrect_with_minus_half = int(np.floor((out_idx + 0.5) * scale - 0.5))
        actual = OpResizeNearestNeighbor._nearest_index(
            out_idx, in_size=in_size, out_size=out_size, align_corners=False, half_pixel_centers=True
        )
        assert actual == correct
        if correct != incorrect_with_minus_half:
            assert actual != incorrect_with_minus_half
