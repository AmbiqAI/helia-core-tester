"""Seeded random conv shapes."""

from __future__ import annotations

import json
import os
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
    ({"hidden_dir": "/tmp/hidden"}, "needs random_shapes"),
    ({"random_shapes": 2, "hidden_dir": "artifacts/hidden"}, "neither hold nor sit inside"),
    ({"random_shapes": 2, "hidden_dir": ".."}, "neither hold nor sit inside"),
    ({"random_shapes": 2, "hidden_dir": "/tmp/h", "hidden_seed_file": "/tmp/h/artifacts/seed.txt"}, "and hidden_dir"),
    ({"random_shapes": 2, "hidden_dir": "/tmp/h", "generated_tests_root": "/tmp/g"}, "no generated_tests_root"),
    ({"random_shapes": 2, "hidden_dir": "/tmp/h", "reports_root": "/tmp/r"}, "no reports_root"),
    ({"random_shapes": 2, "hidden_dir": "/tmp/h", "hidden_seed_file": "seed.txt"}, "outside the tester tree"),
    ({"random_shapes": 2, "hidden_seed_file": "/tmp/seed"}, "needs hidden_dir"),
    ({"hidden_seed_file": "/tmp/seed"}, "needs hidden_dir"),
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



SECRET = "s3cret-hidden-seed-0123"


def test_hidden_cases_are_opaque() -> None:
    import re

    cases = rs.hidden_cases(6, SECRET.encode(), "cortex-m55")
    assert cases == rs.hidden_cases(6, SECRET.encode(), "cortex-m55")
    assert cases != rs.hidden_cases(6, (SECRET + "x").encode(), "cortex-m55")
    for case in cases:
        assert re.fullmatch(r"h[0-9a-f]{12}", case["name"]) and "shape_seed" not in case
    assert len({c["name"] for c in cases}) == len(cases)


@pytest.mark.parametrize("raw", ["", "short", " " * 20])
def test_short_secret_refused(raw: str) -> None:
    with pytest.raises(ValueError, match="16"):
        rs.hidden_secret(raw)


def test_hidden_summary_keeps_only_commitment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(rs.SECRET_ENV, SECRET)
    descriptors = rs.prepare_hidden(tmp_path, 3, "cortex-m55")
    assert descriptors.parent == tmp_path / "artifacts" / "random_shapes" / "cortex-m55"
    text = (descriptors.parent / "summary.json").read_text()
    assert json.loads(text)["seed_commitment"] == rs.seed_commitment(SECRET.encode())
    written = text + "".join(p.read_text() for p in descriptors.rglob("*.yaml"))
    assert SECRET not in written and "shape_seed" not in written


def test_hidden_refuses_shape_seed(tmp_path: Path) -> None:
    from helia_core_tester.cli import get_config
    from helia_core_tester.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="not shape_seed"):
        get_config(project_root=Path.cwd(), random_shapes=2, shape_seed=4, hidden_dir=tmp_path)


def test_step_passes_hidden_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(rs.SECRET_ENV, raising=False)
    seed_file = tmp_path / "seed"
    config = Config(project_root=Path.cwd(), random_shapes=4, hidden_dir=tmp_path / "h", hidden_seed_file=seed_file)
    step = GenerateStep(config)
    cmd = step._build_cmd("cortex-m55", "int")
    assert "--shape-seed" not in cmd and cmd[cmd.index("--hidden-dir") + 1] == str(tmp_path / "h")
    assert "--generated-tests-dir" not in cmd
    assert step.validate()
    seed_file.write_text(SECRET + "\n")
    assert step.validate() is None
    assert step._hidden_env()[rs.SECRET_ENV] == SECRET
    assert SECRET not in json.dumps(config.to_dict())


def test_hidden_generation_stays_outside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = tmp_path / "hidden"
    generated = hidden / "artifacts" / "generated_tests" / "int" / "cortex-m55"
    old_draw = generated / "ConvolutionFunctions" / "h000000000000"
    old_draw.mkdir(parents=True)
    monkeypatch.setenv(rs.SECRET_ENV, SECRET)
    monkeypatch.setattr(generation_module, "find_repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(rs, "prepare_hidden", lambda *_args: tmp_path)
    base = {"operator": "Convolve", "activation_dtype": "S8", "weight_dtype": "S8",
            "_family": "ConvolutionFunctions", "_parity_kind": "cmsis"}
    monkeypatch.setattr(generation_module, "load_all_descriptors", lambda _path: [{"name": "habcdefabcdef", **base}])

    def _fake_generate_test(desc, out_dir, **_kwargs):
        test_dir = Path(out_dir) / desc["_family"] / desc["name"]
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / f"{desc['name']}.tflite").write_bytes(b"\x01")

    monkeypatch.setattr(generation_module, "generate_test", _fake_generate_test)
    filters = {"op": None, "dtype": None, "wtype": None, "name": None, "limit": None, "seed": 1,
               "cpu": "cortex-m55", "suite": "int", "float_precision": "both", "generated_tests_dir": str(generated),
               "random_shapes": 1, "shape_seed": 0, "hidden_dir": str(hidden), "keep_unselected": False}
    generation_module.test_generation(filters)

    report = hidden / "artifacts" / "reports" / "generation" / "int" / "cortex-m55"
    summary = json.loads((report / "generation_summary.json").read_text())
    assert summary["filters"]["seed_commitment"] == rs.seed_commitment(SECRET.encode())
    assert "shape_seed" not in summary["filters"]
    assert not old_draw.exists() and not (tmp_path / "repo").exists()


def test_hidden_dir_owns_its_tree(tmp_path: Path) -> None:
    from helia_core_tester.generation import conftest

    class _Options:
        def __init__(self, **values):
            self.values = {"--cpu": "m55", "--suite": "int", **values}

        def getoption(self, name):
            return self.values.get(name)

    hidden = _Options(**{"--hidden-dir": str(tmp_path), "--random-shapes": 2, "--keep-unselected": True})
    assert not conftest._keeps_unselected(hidden)
    assert conftest._generated_override(hidden) == str(tmp_path / "artifacts/generated_tests/int/cortex-m55")
    public = _Options(**{"--random-shapes": 2})
    assert conftest._keeps_unselected(public) and conftest._generated_override(public) is None


def _guard_options(**values):
    from types import SimpleNamespace

    options = SimpleNamespace(tbstyle="long", showlocals=True, fulltrace=True)
    values = {"--cpu": "m55", "--suite": "int", "--random-shapes": 2, **values}
    return SimpleNamespace(option=options, getoption=values.get)


def test_direct_pytest_hidden_guard(tmp_path: Path) -> None:
    from helia_core_tester.generation import conftest

    config = _guard_options(**{"--hidden-dir": str(tmp_path)})
    conftest._guard_hidden(config)
    assert (config.option.tbstyle, config.option.showlocals, config.option.fulltrace) == ("native", False, False)
    in_tree = _guard_options(**{"--hidden-dir": str(Path.cwd() / "artifacts" / "h")})
    with pytest.raises(pytest.UsageError, match="neither hold nor sit inside"):
        conftest._guard_hidden(in_tree)
    with pytest.raises(pytest.UsageError, match="neither hold nor sit inside"):
        conftest._guard_hidden(_guard_options(**{"--hidden-dir": str(Path.cwd().parent)}))
    escaped = _guard_options(**{"--hidden-dir": str(tmp_path), "--generated-tests-dir": str(tmp_path)})
    with pytest.raises(pytest.UsageError, match="takes no --generated-tests-dir"):
        conftest._guard_hidden(escaped)
    conftest._guard_hidden(_guard_options())


@pytest.mark.parametrize("leaf", ["random_shapes", "reports"])
def test_symlinked_hidden_dest_refused(tmp_path: Path, leaf: str) -> None:
    from helia_core_tester.core.errors import ConfigurationError
    from helia_core_tester.generation import conftest

    hidden = tmp_path / "hidden"
    (hidden / "artifacts").mkdir(parents=True)
    (hidden / "artifacts" / leaf).symlink_to(Path.cwd() / "artifacts", target_is_directory=True)
    with pytest.raises(pytest.UsageError, match="is a symlink"):
        conftest._guard_hidden(_guard_options(**{"--hidden-dir": str(hidden)}))
    with pytest.raises(ConfigurationError, match="is a symlink"):
        Config(project_root=Path.cwd(), random_shapes=2, hidden_dir=hidden)


@pytest.mark.parametrize("count", [None, 0, -1])
def test_direct_hidden_needs_a_count(tmp_path: Path, count) -> None:
    from helia_core_tester.generation import conftest

    with pytest.raises(pytest.UsageError, match="needs --random-shapes"):
        conftest._guard_hidden(_guard_options(**{"--hidden-dir": str(tmp_path), "--random-shapes": count}))


@pytest.mark.parametrize("kind", ["family", "file"])
def test_symlink_inside_hidden_tree_refused(tmp_path: Path, kind: str) -> None:
    from helia_core_tester.core.errors import ConfigurationError

    hidden = tmp_path / "hidden"
    family = hidden / "artifacts" / "generated_tests" / "int" / "cortex-m55" / "ConvolutionFunctions"
    if kind == "family":
        family.parent.mkdir(parents=True)
        family.symlink_to(Path.cwd() / "assets", target_is_directory=True)
    else:
        family.mkdir(parents=True)
        (family / "case.c").symlink_to(Path.cwd() / "README.md")
    with pytest.raises(ConfigurationError, match="is a symlink"):
        Config(project_root=Path.cwd(), random_shapes=2, hidden_dir=hidden)


def test_pruning_never_follows_links(tmp_path: Path) -> None:
    from helia_core_tester.generation.reuse import prune_unlisted_cases, reset_case_dir

    outside = tmp_path / "outside"
    (outside / "case").mkdir(parents=True)
    tree = tmp_path / "tree"
    (tree / "Fam").mkdir(parents=True)
    (tree / "Linked").symlink_to(outside, target_is_directory=True)
    (tree / "Fam" / "linked_case").symlink_to(outside / "case", target_is_directory=True)
    assert prune_unlisted_cases(tree, set()) == 2
    assert (outside / "case").is_dir() and not (tree / "Linked").is_symlink()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    reset_case_dir(link)
    assert not link.is_symlink() and (outside / "case").is_dir()


def test_hidden_outputs_report_hidden_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    import helia_core_tester.core.steps.generate as generate_module

    monkeypatch.setenv(rs.SECRET_ENV, SECRET)
    step = GenerateStep(Config(project_root=Path.cwd(), random_shapes=2, hidden_dir=tmp_path / "h"))
    expected = str(tmp_path / "h" / "artifacts" / "generated_tests")
    assert step.dry_run().outputs["generated_tests_root"] == expected
    assert step._plan_details().outputs["generated_tests_root"] == expected
    monkeypatch.setattr(generate_module, "run_command", lambda *a, **k: None)
    assert step._do_execute().outputs["generated_tests_root"] == expected

    def _fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "pytest")

    monkeypatch.setattr(generate_module, "run_command", _fail)
    assert step._do_execute().outputs["generated_tests_root"] == expected


def test_hard_linked_report_refused(tmp_path: Path) -> None:
    from helia_core_tester.core.errors import ConfigurationError

    public = tmp_path / "checkout_summary.json"
    public.write_text("public")
    hidden = tmp_path / "hidden"
    report = hidden / "artifacts" / "reports" / "generation" / "int" / "cortex-m55"
    report.mkdir(parents=True)
    os.link(public, report / "generation_summary.json")
    with pytest.raises(ConfigurationError, match="hard-linked"):
        Config(project_root=Path.cwd(), random_shapes=2, hidden_dir=hidden)
    assert public.read_text() == "public"


@pytest.mark.parametrize("seed", [0, 7])
def test_hidden_constructor_refuses_shape_seed(tmp_path: Path, seed: int) -> None:
    from helia_core_tester.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="not shape_seed"):
        Config(project_root=Path.cwd(), random_shapes=2, hidden_dir=tmp_path, shape_seed=seed)


@pytest.mark.parametrize("seed", [0, 4])
def test_direct_hidden_refuses_shape_seed(tmp_path: Path, seed: int) -> None:
    from helia_core_tester.generation import conftest

    with pytest.raises(pytest.UsageError, match="not --shape-seed"):
        conftest._guard_hidden(_guard_options(**{"--hidden-dir": str(tmp_path), "--shape-seed": seed}))
