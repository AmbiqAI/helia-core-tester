"""Data-scaled float TransposeConv tolerance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from helia_core_tester.core.config import Config
from helia_core_tester.generation import random_shapes as rs
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.io.dtypes import default_comparison_for_dtype
from helia_core_tester.generation.ops._shared import sum_bound as sb
from helia_core_tester.hardware.comparison import FLOAT_HEADROOM
from helia_core_tester.hardware.generated_test_bridge import _comparison_from_generated_source

TCF32, TCF16 = ("TransposeConv", "FP32"), ("TransposeConv", "FP16")
PARAMS = {"stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0,
          "activation_min": -np.inf, "activation_max": np.inf}


def _scatter(x: np.ndarray, w: np.ndarray, params: dict, out_hw: tuple) -> tuple:
    """Float64 sums and term counts."""
    out = np.zeros((x.shape[0], *out_hw, w.shape[0]))
    count = np.zeros(out.shape, int)
    for b, iy, ix, ky, kx in np.ndindex(x.shape[0], x.shape[1], x.shape[2], w.shape[1], w.shape[2]):
        oy = iy * params["stride_h"] + ky - params["pad_h"]
        ox = ix * params["stride_w"] + kx - params["pad_w"]
        if 0 <= oy < out_hw[0] and 0 <= ox < out_hw[1]:
            out[b, oy, ox] += w[:, ky, kx].astype(np.float64) @ x[b, iy, ix].astype(np.float64)
            count[b, oy, ox] += x.shape[3]
    return out.ravel(), count.ravel()


def _case(rng: np.random.Generator, shape: tuple, kernel: tuple, cout: int, scale: float, dtype=np.float32):
    """Random data at the given scale."""
    x = rng.uniform(-scale, scale, shape).astype(dtype)
    w = rng.uniform(-0.2, 0.2, (cout, *kernel, shape[3])).astype(dtype)
    b = rng.uniform(-scale / 2, scale / 2, cout).astype(dtype)
    return x, w, b


@pytest.mark.parametrize("params", [PARAMS, {**PARAMS, "stride_h": 2, "stride_w": 3, "pad_h": 1, "pad_w": 2}])
def test_terms_match_scatter(params: dict) -> None:
    x, w, _ = _case(np.random.default_rng(0), (2, 3, 4, 5), (3, 4), 3, 1.0)
    out_hw = (6, 9)
    terms, count = sb.tconv_terms(x, w, params, out_hw)
    sums, expected = _scatter(x, w, params, out_hw)
    assert np.allclose(terms.sum(axis=1), sums, rtol=0, atol=1e-12)
    assert np.array_equal(count, expected)


def test_bound_scales_with_terms() -> None:
    # The failing hidden wide case, in miniature.
    x, w, b = _case(np.random.default_rng(1), (1, 1, 54, 26), (1, 8), 27, 352.7)
    terms, count = sb.tconv_terms(x, w, PARAMS, (1, 61))
    rows = np.tile(b.astype(np.float64), 61)
    ref = (terms.sum(axis=1) + rows).astype(np.float32).reshape(1, 1, 61, 27)
    base = default_comparison_for_dtype("FP32")
    atol = sb.tconv_tolerance(x, w, b, PARAMS, ref, base["rtol"], base["atol"])
    scale = np.sqrt(np.square(terms).sum(axis=1) + np.square(rows))
    assert atol == pytest.approx(sb.SUM_BOUND * float(np.max(np.sqrt(count + 1) * np.finfo(np.float32).eps * scale)))
    golden = ref.astype(np.float64).ravel()
    drift = np.abs(sb.reordered_sums(terms, rows, np.float32)["reversed"] - golden)
    # The fixed tolerance fails this order.
    assert np.any(drift > base["atol"] + base["rtol"] * np.abs(golden))
    assert np.all(drift <= sb.DRAW_SHARE * (atol + base["rtol"] * np.abs(golden)))


def test_small_data_keeps_default() -> None:
    x, w, b = _case(np.random.default_rng(2), (1, 2, 2, 1), (1, 1), 1, 1.0)
    ref = (np.einsum("nhwc,oc->nhwo", x.astype(np.float64), w[:, 0, 0]) + b).astype(np.float32)
    assert sb.tconv_tolerance(x, w, b, PARAMS, ref, 2e-5, 5e-5) == 5e-5


def _f16_rounded(terms: np.ndarray, rows: np.ndarray, cin: int) -> np.ndarray:
    """Accumulator through _Float16 every tap."""
    acc = rows.astype(np.float32)
    for tap in range(terms.shape[1] // cin):
        part = terms[:, tap * cin:(tap + 1) * cin].astype(np.float32).sum(axis=1, dtype=np.float32)
        acc = (acc + part).astype(np.float16).astype(np.float32)
    return acc


@pytest.mark.parametrize(("shape", "kernel", "low"), [
    ((1, 8, 8, 96), (8, 8), 0.0), ((1, 8, 8, 96), (8, 8), -1.0), ((1, 5, 5, 32), (5, 5), -1.0),
    ((1, 1, 54, 26), (1, 8), -352.7),
])
def test_f16_loss_still_fails(shape: tuple, kernel: tuple, low: float) -> None:
    # Long sums: the hardest case to catch.
    rng = np.random.default_rng(3)
    x = rng.uniform(low, abs(low) or 1.0, shape).astype(np.float32)
    w = rng.uniform(-0.05, 0.05, (1, *kernel, shape[3])).astype(np.float32)
    b = np.float32([0.25])
    out_hw = (shape[1] + kernel[0] - 1, shape[2] + kernel[1] - 1)
    terms, _ = sb.tconv_terms(x, w, PARAMS, out_hw)
    rows = np.tile(b.astype(np.float64), terms.shape[0])
    ref = (terms.sum(axis=1) + rows).astype(np.float32)
    atol = sb.tconv_tolerance(x, w, b, PARAMS, ref.reshape(1, *out_hw, 1), 2e-5, 5e-5)
    loss = np.abs(_f16_rounded(terms, rows, shape[3]) - ref) / (atol + 2e-5 * np.abs(ref))
    assert loss.max() > 1 / FLOAT_HEADROOM


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_zero_output_still_fails(dtype) -> None:
    # Long FP16 sums once allowed zeros.
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, (1, 7, 7, 39)).astype(dtype)
    w = rng.uniform(-0.15, 0.15, (1, 7, 6, 39)).astype(dtype)
    b = np.zeros(1, dtype)
    terms, _ = sb.tconv_terms(x, w, PARAMS, (13, 12))
    ref = terms.sum(axis=1).astype(dtype)
    base = default_comparison_for_dtype("FP16" if dtype == np.float16 else "FP32")
    atol = sb.tconv_tolerance(x, w, b, PARAMS, ref.reshape(1, 13, 12, 1), base["rtol"], base["atol"])
    golden = np.abs(ref.astype(np.float64))
    assert (golden / (atol + base["rtol"] * golden)).max() > 5


def test_f16_checks_f16_sums() -> None:
    x, w, b = _case(np.random.default_rng(4), (1, 4, 4, 16), (3, 3), 4, 1.0, np.float16)
    terms, _ = sb.tconv_terms(x, w, PARAMS, (6, 6))
    sums = sb.reordered_sums(terms, np.tile(b.astype(np.float64), 36), np.float16)
    assert set(sums) == {"reversed", "pairwise", "f16_reversed"} and sums["f16_reversed"].dtype == np.float16


def test_drift_refuses_tolerance() -> None:
    x, w, b = _case(np.random.default_rng(5), (1, 3, 3, 8), (2, 2), 2, 1.0)
    terms, _ = sb.tconv_terms(x, w, PARAMS, (4, 4))
    ref = (terms.sum(axis=1) + np.tile(b.astype(np.float64), 16)).astype(np.float32).reshape(1, 4, 4, 2)
    assert sb.tconv_tolerance(x, w, b, PARAMS, ref, 2e-5, 5e-5) is not None
    ref.flat[3] += 1.0
    assert sb.tconv_tolerance(x, w, b, PARAMS, ref, 2e-5, 5e-5) is None


def test_nonfinite_lanes_skip_check() -> None:
    x, w, b = _case(np.random.default_rng(6), (1, 3, 3, 4), (2, 2), 2, 1.0)
    x[0, 1, 1, 2] = np.inf
    terms, _ = sb.tconv_terms(np.where(np.isfinite(x), x, 0), w, PARAMS, (4, 4))
    ref = (terms.sum(axis=1) + np.tile(b.astype(np.float64), 16)).astype(np.float32).reshape(1, 4, 4, 2)
    # A finite golden the token reaches.
    ref[0, 1, 1, 0] = 0.0
    assert sb.tconv_tolerance(x, w, b, PARAMS, ref, 2e-5, 5e-5) is not None


def test_only_float_tconv_draws_bound() -> None:
    cases = rs.sample_cases(4, 1, ops=rs.OPS)
    assert {(c["operator"], c["activation_dtype"]) for c in cases if c.get("sum_bound")} == {TCF32, TCF16}
    assert all(c.get("sum_bound") is not True for c in cases if c["activation_dtype"] not in ("FP32", "FP16"))


def _generate(tmp_path: Path, key: tuple, n: int = 14) -> dict[str, Path]:
    """Generated edge cases, by edge."""
    from helia_core_tester.generation.test_ops import generate_test

    descriptors = load_all_descriptors(str(rs.prepare_shapes(tmp_path, n, 5, "cortex-m55", (key,))))
    tree = tmp_path / "artifacts" / "generated_tests" / "float" / "cortex-m55"
    dirs = {}
    for desc in descriptors:
        if desc.get("edge") in ("wide", "long_acc"):
            generate_test(desc, str(tree), seed=Config.seed, cpu="cortex-m55")
            dirs[desc["edge"]] = tree / "ConvolutionFunctions" / desc["name"]
    return dirs


@pytest.mark.parametrize("key", [TCF32, TCF16])
def test_descriptor_records_tolerance(tmp_path: Path, key: tuple) -> None:
    dirs = _generate(tmp_path, key)
    base = default_comparison_for_dtype(key[1])
    for edge, directory in dirs.items():
        desc = yaml.safe_load((directory / "descriptor.yaml").read_text())
        resolved = desc["resolved_comparison"]
        assert resolved["atol"] >= base["atol"] and resolved["rtol"] == base["rtol"]
        if edge == "long_acc" and key == TCF16:
            assert resolved == {"mode": "float", **desc["comparison"]} and resolved["atol"] > base["atol"]
        source = next(directory.glob("*_transpose_conv.c")).read_text()
        assert _comparison_from_generated_source(source)["atol"] == pytest.approx(resolved["atol"], rel=1e-6)


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first, second = _generate(tmp_path / "a", TCF32), _generate(tmp_path / "b", TCF32)
    for edge, directory in first.items():
        for name in ("descriptor.yaml", f"{directory.name}_transpose_conv.c"):
            assert (directory / name).read_bytes() == (second[edge] / name).read_bytes()


def test_redraw_is_seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from helia_core_tester.generation.ops.ConvolutionFunctions import transpose_conv

    real, calls = transpose_conv.tconv_tolerance, []

    def refuse_first(x, *args):
        calls.append(x.copy())
        return None if len(calls) % 2 else real(x, *args)

    monkeypatch.setattr(transpose_conv, "tconv_tolerance", refuse_first)
    first = _generate(tmp_path / "a", TCF32)["wide"]
    taken = calls[1]
    calls.clear()
    second = _generate(tmp_path / "b", TCF32)["wide"]
    assert not np.array_equal(calls[0], taken) and np.array_equal(calls[1], taken)
    assert (first / "descriptor.yaml").read_bytes() == (second / "descriptor.yaml").read_bytes()


def test_endless_drift_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from helia_core_tester.generation.ops.ConvolutionFunctions import transpose_conv

    monkeypatch.setattr(transpose_conv, "tconv_tolerance", lambda *args: None)
    with pytest.raises(ValueError, match="reordered sums exceed half the tolerance"):
        _generate(tmp_path, TCF32)
