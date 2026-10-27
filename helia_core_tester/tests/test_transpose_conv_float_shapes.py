"""Hidden float TransposeConv shapes: draws, goldens, bridge."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from helia_core_tester.core.config import Config
from helia_core_tester.core.steps.generate import GenerateStep
from helia_core_tester.generation import random_shapes as rs
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops._shared.quant_knobs import clamp_golden
from helia_core_tester.generation.utils.template_context import FLOAT_TCONV_MIN_CTX
from helia_core_tester.hardware.adapter_specs import render_generated_adapters_source

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TCF32, TCF16 = ("TransposeConv", "FP32"), ("TransposeConv", "FP16")
SECRET = "s3cret-hidden-seed-0123"


def test_float_keys_get_their_routes() -> None:
    assert rs.select_ops("TransposeConv", "fp32") == (TCF32,) and rs.select_ops("TransposeConv", "FP16") == (TCF16,)
    assert rs.select_ops("TransposeConv", None) == (("TransposeConv", "S16"), TCF32, TCF16, ("TransposeConv", "S8"))
    assert [rs.GENERATORS[k].stream for k in (TCF32, TCF16)] == [4, 5]
    layer = rs.Layer(2, 3, 4, 5, 3, 3, 2, 2, transpose=True)
    routes = {dtype: rs.layer_route("TransposeConv", layer, True, dtype) for dtype in ("S16", "FP32", "FP16")}
    assert routes == {"S16": rs.TC16_ROUTE, "FP32": "arm_transpose_conv_f32", "FP16": "arm_transpose_conv_f16"}


@pytest.mark.parametrize("cpu", ["cortex-m55", "cortex-m4"])
def test_tc16_hidden_set_unchanged(cpu: str) -> None:
    # Pinned before float joined.
    cases = rs.hidden_cases(12, SECRET.encode(), cpu, (("TransposeConv", "S16"),))
    digest = hashlib.sha256(yaml.safe_dump_all(cases, sort_keys=False).encode()).hexdigest()
    assert digest == "497c5857bac516c0be6498c6ce11e835656e6ee4ca242e2c57215e04dd725971"


def test_float_suite_plumbing() -> None:
    assert rs.op_suites("TransposeConv", "FP32") == {"float"} and rs.op_suites("TransposeConv", "S16") == {"int"}
    assert rs.op_suites("TransposeConv", None) == {"int", "float"}
    config = Config(project_root=PROJECT_ROOT, random_shapes=2, suite="both", op_filter="TransposeConv",
                    dtype_filter="FP32")
    assert {suite for _, suite, _ in GenerateStep(config)._targets()} == {"float"}
    # Default int keeps the int draws.
    config = Config(project_root=PROJECT_ROOT, random_shapes=2)
    assert {suite for _, suite, _ in GenerateStep(config)._targets()} == {"int"}


@pytest.mark.parametrize("key", [TCF32, TCF16])
@pytest.mark.parametrize("seed", range(6))
def test_float_edges_hold(key: tuple[str, str], seed: int) -> None:
    lanes = rs.TCF_LANES[key[1]]
    cases = rs.sample_cases(14, seed, ops=(key,))
    edges = [c for c in cases if "edge" in c]
    assert [c["edge"] for c in edges] == list(rs.TCF_EDGES) and len(edges) == rs.edge_count(14, len(rs.TCF_EDGES))
    for case in cases:
        n, h, w, cin = case["input_shape"]
        kh, kw, cout, _ = case["filter_shape"]
        sh, sw = case["strides"]
        edge = case.get("edge")
        assert case["suite"] == "float" and case["weight_dtype"] == key[1]
        assert set(case["tensor_dtypes"].values()) == {key[1]}
        assert (n == 2) == (edge == "batch2")
        if edge == "long_acc":
            assert kh * kw * cin >= 25 * 32 and case["activation"] == "NONE"
        elif edge == "nonfinite":
            assert case["nonfinite_policy"] == "mask" and len(case["nonfinite_tokens"]) == 1
            assert 0 <= case["nonfinite_positions"][0] < h * w * cin and min(h, w) > max(kh, kw)
        elif edge == "clamp":
            low, high = case["activation_min"], case["activation_max"]
            assert low < 0 < high and case["activation"] == "NONE"
            cast = np.float16 if key[1] == "FP16" else np.float32
            assert float(cast(low)) == low and float(cast(high)) == high
        elif edge == "odd_tail":
            assert cin % lanes and cout % lanes and cout > lanes
        elif edge == "wide":
            low, high = case["input_range"]
            assert low == -high and not 0.1 < high < 10
        elif edge == "gaps":
            assert sw > kw
        elif edge == "same_offset":
            assert case["padding"] == "SAME" and (kh - sh) % 2 and (kw - sw) % 2
        elif edge == "overlap":
            assert min(kh, kw) >= 5


def test_float_mixed_draws_vary() -> None:
    cases = [c for c in rs.sample_cases(60, 3, ops=(TCF32, TCF16)) if "edge" not in c]
    assert {c["activation"] for c in cases} == {"NONE", "RELU", "RELU6"}
    assert {c["padding"] for c in cases} == {"SAME", "VALID"}
    assert any(c["use_bias"] for c in cases) and any(not c["use_bias"] for c in cases)
    assert any("input_range" in c for c in cases)


def test_float_clamp_golden() -> None:
    out = np.array([-2.0, -0.5, np.nan, 0.5, 2.0], np.float16)
    clamped = clamp_golden({"activation_min": -1.0, "activation_max": 1.0}, out)
    assert clamped.dtype == np.float16 and np.array_equal(clamped, [-1, -0.5, np.nan, 0.5, 1], equal_nan=True)
    assert clamp_golden({"activation_max": 0.25}, out).tolist()[:2] == [-2.0, -0.5]


def test_float_firmware_takes_batches() -> None:
    source = render_generated_adapters_source()
    body = source[source.index("run_transpose_conv_once(hct_server_session_t"):]
    body = body[:body.index("run_transpose_conv_s16_once")]
    assert "output_dims.n != input_dims.n" in body and "input_dims.n != 1 || output_dims.n != 1" in body


def _reference(blobs: dict, params: dict) -> np.ndarray:
    """Float64 scatter TransposeConv."""
    x, w, out = blobs["input_0"], blobs["weights"], blobs["expected_output"]
    acc = np.zeros(out.shape, np.float64)
    for b, iy, ix, fy, fx in np.ndindex(x.shape[0], x.shape[1], x.shape[2], w.shape[1], w.shape[2]):
        oy, ox = iy * params["stride_h"] - params["pad_h"] + fy, ix * params["stride_w"] - params["pad_w"] + fx
        if 0 <= oy < out.shape[1] and 0 <= ox < out.shape[2]:
            acc[b, oy, ox] += w[:, fy, fx, :].astype(np.float64) @ x[b, iy, ix].astype(np.float64)
    bits = [np.uint32(params[f"float_activation_{k}_bits"]).view(np.float32) for k in ("min", "max")]
    return np.clip(acc + blobs.get("bias", 0), *bits)


@pytest.mark.parametrize("key", [TCF32, TCF16])
def test_float_goldens_bridge(tmp_path: Path, key: tuple[str, str]) -> None:
    from helia_core_tester.generation.test_ops import generate_test
    from helia_core_tester.hardware.generated_test_bridge import (
        build_case_bundle_from_generated_test, discover_generated_tests,
    )

    descriptors = load_all_descriptors(str(rs.prepare_shapes(tmp_path, 14, 5, "cortex-m55", (key,))))
    tree = tmp_path / "artifacts" / "generated_tests" / "float" / "cortex-m55"
    for desc in descriptors:
        if "edge" in desc:
            generate_test(desc, str(tree), seed=Config.seed, cpu="cortex-m55")
    dtype = np.float16 if key[1] == "FP16" else np.float32
    seen = set()
    for case in discover_generated_tests(PROJECT_ROOT, tests_root=tmp_path, suite="float"):
        bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, case, output_root=tmp_path / "b",
                                                       require_fvp_pass=False)
        blobs = {b.role: np.frombuffer(b.path.read_bytes(), dtype).reshape(b.dimensions) for b in bundle.blobs}
        compare = bundle.manifest["correctness_comparison"]
        expected = blobs["expected_output"].astype(np.float64)
        mask = np.array(compare.get("nonfinite_mask", [0] * expected.size), bool).reshape(expected.shape)
        reference = _reference(blobs, bundle.manifest["serialized_scalar_parameters"])
        error = np.abs(expected - reference)[~mask]
        assert np.all(error <= compare["atol"] + compare["rtol"] * np.abs(reference[~mask]))
        # A few KB fit any candidate.
        assert bundle.manifest["scratch_buffer"]["bytes"] >= FLOAT_TCONV_MIN_CTX + 1024
        edge = case.descriptor["edge"]
        seen.add(edge)
        if edge == "batch2":
            assert expected.shape[0] == 2 and not np.array_equal(expected[0], expected[1])
        elif edge == "clamp":
            low, high = case.descriptor["activation_min"], case.descriptor["activation_max"]
            assert np.any(expected == low) and np.any(expected == high)
        elif edge == "nonfinite":
            assert 0 < mask.sum() < mask.size
    assert seen == set(rs.TCF_EDGES)


def _float_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **filters) -> Path:
    """One float generation pass, faked."""
    import helia_core_tester.generation.test_ops as generation_module

    generated = tmp_path / "artifacts" / "generated_tests" / "float" / "cortex-m55"
    desc = {"name": "rs1_tcf32_000", "operator": "TransposeConv", "activation_dtype": "FP32", "weight_dtype": "FP32",
            "suite": "float", "_family": "ConvolutionFunctions", "_parity_kind": "cmsis"}
    monkeypatch.setattr(generation_module, "find_repo_root", lambda: tmp_path)
    for name in ("prepare_shapes", "prepare_hidden"):
        monkeypatch.setattr(rs, name, lambda *_args: tmp_path)
    monkeypatch.setattr(generation_module, "load_all_descriptors", lambda _path: [desc])

    def _fake_generate_test(desc, out_dir, **_kwargs):
        test_dir = Path(out_dir) / desc["_family"] / desc["name"]
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / f"{desc['name']}.tflite").write_bytes(b"\x01")

    monkeypatch.setattr(generation_module, "generate_test", _fake_generate_test)
    generation_module.test_generation({
        "op": None, "dtype": None, "wtype": None, "name": None, "limit": None, "seed": 1, "cpu": "cortex-m55",
        "suite": "float", "float_precision": "both", "generated_tests_dir": str(generated), "random_shapes": 2,
        "shape_seed": 1, **filters,
    })
    return generated


def test_mixed_op_draw_splits_by_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The float pass holds no Convolve.
    _float_pass(tmp_path, monkeypatch, op="Convolve,TransposeConv")


def test_hidden_reuse_drops_other_suites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(rs.SECRET_ENV, SECRET)
    hidden = tmp_path / "hidden"
    old = hidden / "artifacts" / "generated_tests" / "int" / "cortex-m55" / "ConvolutionFunctions" / "h0123456789ab"
    old.mkdir(parents=True)
    _float_pass(tmp_path, monkeypatch, op="TransposeConv", dtype="FP32", hidden_dir=str(hidden), shape_seed=None)
    assert not old.parent.parent.exists()
    # A both-suite draw keeps int.
    old.mkdir(parents=True)
    _float_pass(tmp_path, monkeypatch, op="TransposeConv", hidden_dir=str(hidden), shape_seed=None)
    assert old.is_dir()
