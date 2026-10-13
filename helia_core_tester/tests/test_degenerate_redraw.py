"""A fresh run seed can draw data whose golden is degenerate; generation redraws such a case from
seeds derived from its own, at most DEGENERATE_DRAW_ATTEMPTS times, and records the seed it kept."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import helia_core_tester.generation.test_ops as generation_module
from helia_core_tester.generation.golden_check import DegenerateGoldenError
from helia_core_tester.generation.test_ops import DEGENERATE_DRAW_ATTEMPTS, default_seed_for_case, redraw_seed

NAME = "fc_redraw"
RUN_SEED = 123


def _filters(generated_tests_dir: Path) -> dict[str, object]:
    return {"op": None, "dtype": None, "wtype": None, "name": None, "limit": None, "seed": RUN_SEED,
            "cpu": "cortex-m55", "suite": "int", "float_precision": "both",
            "generated_tests_dir": str(generated_tests_dir), "force_generate": True}


def test_first_draw_is_the_case_seed():
    assert redraw_seed(1234, 0) == 1234


def test_redraw_seeds_are_deterministic_distinct_and_32_bit():
    seeds = [redraw_seed(1234, a) for a in range(DEGENERATE_DRAW_ATTEMPTS)]
    assert seeds == [redraw_seed(1234, a) for a in range(DEGENERATE_DRAW_ATTEMPTS)]
    assert len(set(seeds)) == DEGENERATE_DRAW_ATTEMPTS
    assert all(0 <= s < 2 ** 32 for s in seeds)
    assert redraw_seed(1234, 1) != redraw_seed(1235, 1)


@pytest.mark.parametrize("attempt", [-1, DEGENERATE_DRAW_ATTEMPTS])
def test_redraw_attempt_out_of_range_is_rejected(attempt):
    with pytest.raises(ValueError, match="redraw attempt"):
        redraw_seed(1, attempt)


def _run(tmp_path, monkeypatch, behaviour):
    """Generate one fake case; `behaviour(attempt)` returns None (success) or raises."""
    repo_root = tmp_path
    out = repo_root / "artifacts" / "generated_tests" / "int" / "cortex-m55"
    monkeypatch.setattr(generation_module, "find_repo_root", lambda: repo_root)
    monkeypatch.setattr(generation_module, "find_descriptors_dir", lambda: repo_root / "assets" / "descriptors")
    monkeypatch.setattr(generation_module, "load_all_descriptors", lambda _p: [{
        "name": NAME, "operator": "FullyConnected", "activation_dtype": "S8", "weight_dtype": "S8",
        "_family": "FullyConnectedFunctions", "_parity_kind": "cmsis", "_source_family": "FullyConnectedFunctions",
        "_source_stem": "fully_connected", "_source_relpath": "FullyConnectedFunctions/fully_connected.yaml",
    }])
    seeds = []

    def _fake_generate_test(desc, out_dir, seed=None, cpu="cortex-m55",
                            generation_failures=None, run_seed=0, kernel_numerics="shipped"):
        seeds.append(seed)
        test_dir = Path(out_dir) / desc["_family"] / desc["name"]
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / f"{desc['name']}_fully_connected.c").write_text("// fake\n")
        try:
            behaviour(len(seeds) - 1)
        except Exception as e:
            generation_failures.append({"name": desc["name"], "operator": desc["operator"], "stage": "c_files",
                                        "exception": repr(e), "case_seed": int(seed)})
            raise

    monkeypatch.setattr(generation_module, "generate_test", _fake_generate_test)
    return out, seeds


def _degenerate_until(n):
    def behaviour(attempt):
        if attempt < n:
            raise DegenerateGoldenError(f"{NAME}: degenerate golden (attempt {attempt})")
    return behaviour


@pytest.mark.parametrize("redraws", [0, 1, DEGENERATE_DRAW_ATTEMPTS - 1])
def test_a_degenerate_draw_is_redrawn_and_the_kept_seed_recorded(tmp_path, monkeypatch, capsys, redraws):
    out, seeds = _run(tmp_path, monkeypatch, _degenerate_until(redraws))
    generation_module.test_generation(_filters(out))
    base = default_seed_for_case(NAME, RUN_SEED)
    assert seeds == [redraw_seed(base, a) for a in range(redraws + 1)]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tests"][0]["case_seed"] == redraw_seed(base, redraws)
    report = tmp_path / "artifacts" / "reports" / "generation" / "int" / "cortex-m55"
    assert json.loads((report / "generation_failures.json").read_text()) == []
    printed = capsys.readouterr().out
    assert printed.count("Redrawing") == redraws
    assert "ERROR" not in printed


def test_a_case_degenerate_on_every_draw_fails_after_the_last(tmp_path, monkeypatch, capsys):
    out, seeds = _run(tmp_path, monkeypatch, _degenerate_until(DEGENERATE_DRAW_ATTEMPTS))
    with pytest.raises(AssertionError):  # the only case failed: nothing was generated
        generation_module.test_generation(_filters(out))
    assert len(seeds) == len(set(seeds)) == DEGENERATE_DRAW_ATTEMPTS
    report = tmp_path / "artifacts" / "reports" / "generation" / "int" / "cortex-m55"
    failures = json.loads((report / "generation_failures.json").read_text())
    assert [f["name"] for f in failures] == [NAME]
    assert failures[0]["case_seed"] == seeds[-1]
    printed = capsys.readouterr().out
    assert f"after {DEGENERATE_DRAW_ATTEMPTS} draws" in printed
    assert f"Reproduce with --seed {RUN_SEED} --name {NAME}" in printed


def test_other_generation_errors_are_not_redrawn(tmp_path, monkeypatch):
    def broken(_attempt):
        raise RuntimeError("template error")
    out, seeds = _run(tmp_path, monkeypatch, broken)
    with pytest.raises(AssertionError):
        generation_module.test_generation(_filters(out))
    assert len(seeds) == 1
    report = tmp_path / "artifacts" / "reports" / "generation" / "int" / "cortex-m55"
    assert [f["name"] for f in json.loads((report / "generation_failures.json").read_text())] == [NAME]
