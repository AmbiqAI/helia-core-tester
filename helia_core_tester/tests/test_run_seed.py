"""The run seed: one per run, every case drawn from it, recorded for replay, reuse only when chosen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from helia_core_tester.generation.artifact_identity import generated_case_artifact_sha256
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import RUN_SEED_ENV, default_seed_for_case, generate_test, resolve_run_seed

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CASE = "abs_default_s8"


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)


def _generate(tmp_path: Path, run_seed: int) -> Path:
    out = tmp_path / str(run_seed)
    generate_test(_descriptor(_CASE), str(out), run_seed=run_seed)
    return next(p.parent for p in out.rglob(f"{_CASE}.tflite"))


def test_case_seed_mixes_the_run_seed_with_the_name() -> None:
    # Run seed 0 is the historical per-name seed the pinned fixtures were drawn with.
    assert default_seed_for_case("a", 0) == default_seed_for_case("a")
    assert default_seed_for_case("a", 1) == default_seed_for_case("a", 1)
    assert default_seed_for_case("a", 0) != default_seed_for_case("a", 1)
    assert default_seed_for_case("a", 1) != default_seed_for_case("a", 2)
    assert default_seed_for_case("a", 1) != default_seed_for_case("b", 1)
    assert 0 <= default_seed_for_case("a", 1) < 2**32


def test_run_seed_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_SEED_ENV, "7")
    assert resolve_run_seed({"seed": 3}) == (3, True)
    # The pipeline's own draw is passed as --seed but is not a choice.
    assert resolve_run_seed({"seed": 3, "fresh_seed": True}) == (3, False)
    assert resolve_run_seed({"seed": None}) == (7, True)
    monkeypatch.delenv(RUN_SEED_ENV)
    first, chosen = resolve_run_seed({})
    second, _ = resolve_run_seed({})
    assert not chosen and 0 <= first < 2**32 and first != second


def test_generated_case_records_both_seeds(tmp_path: Path) -> None:
    case_dir = _generate(tmp_path, run_seed=11)
    sidecar = yaml.safe_load((case_dir / "descriptor.yaml").read_text())
    assert (sidecar["run_seed"], sidecar["case_seed"]) == (11, default_seed_for_case(_CASE, 11))
    generation = json.loads(next(case_dir.glob("*.sidecar.json")).read_text())
    assert (generation["run_seed"], generation["case_seed"]) == (11, sidecar["case_seed"])


def test_same_run_seed_reproduces_and_another_redraws(tmp_path: Path) -> None:
    first = generated_case_artifact_sha256(_generate(tmp_path / "a", run_seed=5))
    again = generated_case_artifact_sha256(_generate(tmp_path / "b", run_seed=5))
    other = generated_case_artifact_sha256(_generate(tmp_path / "c", run_seed=6))

    assert first == again
    assert first != other
