"""Seeded random conv shapes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import helia_core_tester.generation.test_ops as generation_module
from helia_core_tester.core.config import Config
from helia_core_tester.core.steps.generate import GenerateStep
from helia_core_tester.generation import random_shapes as rs
from helia_core_tester.generation.golden_check import DegenerateGoldenError
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops._shared.quant_knobs import clamp_golden, value_range
from helia_core_tester.hardware.generated_test_bridge import HW_CASE_SUFFIX

SEEDS = (0, 1, 7, 12345)


def _layer(case: dict) -> rs.Layer:
    kh, kw, cin, last = case["filter_shape"]
    dw = case["operator"] == "DepthwiseConv"
    return rs.Layer(
        case["input_shape"][1], case["input_shape"][2], cin, cin * last if dw else last, kh, kw,
        *case["strides"], *case["dilation"], padding=case["padding"], mult=last if dw else 1,
    )


def test_same_seed_same_cases() -> None:
    assert rs.sample_cases(20, 5) == rs.sample_cases(20, 5)
    assert rs.sample_cases(20, 5) != rs.sample_cases(20, 6)


def test_ops_draw_independently() -> None:
    both = rs.sample_cases(10, 3)
    assert [c for c in both if c["operator"] == "Convolve"] == rs.sample_cases(10, 3, ops=("Convolve",))


@pytest.mark.parametrize("cpu", ["cortex-m55", "cortex-m4"])
@pytest.mark.parametrize("seed", SEEDS)
def test_cases_fit_the_board(cpu: str, seed: int) -> None:
    workspace = rs.min_workspace(cpu)
    mve = cpu == "cortex-m55"
    for case in rs.sample_cases(50, seed, cpu):
        op, layer = case["operator"], _layer(case)
        assert min(layer.out_hw()) >= 1
        assert rs.footprint(op, layer) <= workspace
        assert rs.layer_macs(op, layer) <= rs.MAX_MACS
        assert rs.layer_route(op, layer, mve) == case["expected_route"]
        assert case["shape_seed"] == seed and f"rs{seed}_" in case["name"]
        assert len(case["name"] + HW_CASE_SUFFIX) < 96
        if op == "DepthwiseConv":
            assert 1 <= case["depth_multiplier"] <= 4


def test_routes_and_edges_covered() -> None:
    cases = rs.sample_cases(50, 11)
    routes = rs.route_counts(cases)
    assert set(routes["Convolve"]) == set(rs.CONV_ROUTES)
    assert {"arm_depthwise_conv_s8", "arm_depthwise_conv_s8_opt"} <= set(routes["DepthwiseConv"])
    assert set(routes["DepthwiseConv"]) & set(rs.CONV_ROUTES)
    layers = [(c, _layer(c)) for c in cases]
    assert {c["activation"] for c in cases} == {"NONE", "RELU", "RELU6"}
    assert {c["padding"] for c in cases} == {"SAME", "VALID"}
    assert {s for c in cases for s in c["strides"]} == {1, 2, 3}
    assert any(max(c["dilation"]) > 1 for c in cases)
    assert any(l.cin % 4 for _, l in layers) and any(l.cin in rs.PRIMES for _, l in layers)
    assert any("activation_min" in c for c in cases)
    assert any(c["input_range"] != c["calibration_range"] for c in cases)
    assert any(c["calibration_range"][0] != -c["calibration_range"][1] for c in cases)
    assert any(
        c["operator"] == "DepthwiseConv" and (l.kh, l.kw) == (3, 3) and l.w * l.cin > 1440 for c, l in layers
    )
    assert {c["depth_multiplier"] for c in cases if c["operator"] == "DepthwiseConv"} == {1, 2, 3, 4}


def test_plain_cpu_hits_3x3_route() -> None:
    routes = rs.route_counts(rs.sample_cases(20, 2, "cortex-m4"))
    assert "arm_depthwise_conv_3x3_s8" in routes["DepthwiseConv"]
    assert not set(routes["Convolve"]) & set(rs.MVE_CONV_ROUTES)


def test_written_descriptors_load(tmp_path: Path) -> None:
    descriptors = rs.prepare_shapes(tmp_path, 6, 9, "cortex-m55")
    loaded = load_all_descriptors(str(descriptors))
    assert [d["name"] for d in loaded] == [c["name"] for c in rs.sample_cases(6, 9)]
    summary = json.loads((descriptors.parent / "summary.json").read_text())
    assert summary["shape_seed"] == 9 and summary["cases"] == 12


def test_quant_knobs() -> None:
    out = np.array([-128, -20, 0, 90, 127], dtype=np.int8)
    assert clamp_golden({}, out) is out
    assert clamp_golden({"activation_min": -10, "activation_max": 80}, out).tolist() == [-10, -10, 0, 80, 80]
    assert value_range({}, "calibration_range", (-1, 1)) == (-1.0, 1.0)
    assert value_range({"calibration_range": [-3, 9]}, "calibration_range", (-1, 1)) == (-3.0, 9.0)


def test_step_passes_shape_flags(tmp_path: Path) -> None:
    config = Config(project_root=Path.cwd(), random_shapes=4, shape_seed=77)
    cmd = GenerateStep(config)._build_cmd("cortex-m55", "int")
    assert cmd[cmd.index("--random-shapes") + 1] == "4"
    assert cmd[cmd.index("--shape-seed") + 1] == "77"


def test_explicit_zero_seed_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from helia_core_tester.cli import get_config

    monkeypatch.setenv("HELIA_CORE_TESTER_SHAPE_SEED", "9")
    assert get_config(project_root=Path.cwd(), random_shapes=2, shape_seed=0).shape_seed == 0
    assert get_config(project_root=Path.cwd(), random_shapes=2).shape_seed == 9
    monkeypatch.delenv("HELIA_CORE_TESTER_SHAPE_SEED")
    assert get_config(project_root=Path.cwd(), random_shapes=2).shape_seed == 0


@pytest.mark.parametrize(("settings", "message"), [
    ({"random_shapes": 0}, "random_shapes must be >= 1"),
    ({"random_shapes": -3}, "random_shapes must be >= 1"),
    ({"random_shapes": 2, "shape_seed": -1}, "shape_seed must be in"),
    ({"random_shapes": 2, "shape_seed": 2**32}, "shape_seed must be in"),
    ({"random_shapes": 2, "suite": "float"}, "needs the int suite"),
])
def test_bad_random_settings_refused(settings: dict, message: str) -> None:
    from helia_core_tester.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match=message):
        Config(project_root=Path.cwd(), **settings)


def test_dry_run_and_plan_skip_float_targets() -> None:
    step = GenerateStep(Config(project_root=Path.cwd(), random_shapes=2, suite="both"))
    previews = step.dry_run().details["commands"]
    assert previews and all(cmd[cmd.index("--suite") + 1] == "int" for cmd in previews)
    assert len(previews) == len(step._targets())


def test_flat_random_draw_is_dropped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generated = tmp_path / "artifacts" / "generated_tests" / "int" / "cortex-m55"
    fixed = generated / "ConvolutionFunctions" / "fixed_case"
    old_draw = generated / "ConvolutionFunctions" / "rs2_conv_000"
    for case in (fixed, old_draw):
        case.mkdir(parents=True)
    base = {"operator": "Convolve", "activation_dtype": "S8", "weight_dtype": "S8",
            "_family": "ConvolutionFunctions", "_parity_kind": "cmsis"}
    monkeypatch.setattr(generation_module, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(rs, "prepare_shapes", lambda *_args: tmp_path)
    monkeypatch.setattr(
        generation_module, "load_all_descriptors",
        lambda _path: [{"name": "rs3_conv_000", **base}, {"name": "rs3_conv_001", **base}],
    )

    def _fake_generate_test(desc, out_dir, generation_failures=None, **_kwargs):
        test_dir = Path(out_dir) / desc["_family"] / desc["name"]
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / f"{desc['name']}.tflite").write_bytes(b"\x01")
        if desc["name"].endswith("001"):
            generation_failures.append({"name": desc["name"]})
            raise DegenerateGoldenError("flat")

    monkeypatch.setattr(generation_module, "generate_test", _fake_generate_test)
    filters = {"op": None, "dtype": None, "wtype": None, "name": None, "limit": None, "seed": 1,
               "cpu": "cortex-m55", "suite": "int", "float_precision": "both",
               "generated_tests_dir": str(generated), "random_shapes": 2, "shape_seed": 3, "keep_unselected": True}
    generation_module.test_generation(filters)

    report = tmp_path / "artifacts" / "reports" / "generation" / "int" / "cortex-m55"
    summary = json.loads((report / "generation_summary.json").read_text())
    assert summary["counts"]["skipped_degenerate"] == 1
    assert summary["filters"]["shape_seed"] == 3
    assert fixed.is_dir() and not old_draw.exists()
    assert not (generated / "ConvolutionFunctions" / "rs3_conv_001").exists()


def test_largest_seed_case_id_fits_firmware() -> None:
    from helia_core_tester.core.config import MAX_SHAPE_SEED
    from helia_core_tester.hardware.session import MAX_CASE_ID_BYTES

    longest = f"rs{MAX_SHAPE_SEED}_{max(rs.TAGS.values(), key=len)}_9999{HW_CASE_SUFFIX}"
    assert len(longest.encode()) <= MAX_CASE_ID_BYTES

