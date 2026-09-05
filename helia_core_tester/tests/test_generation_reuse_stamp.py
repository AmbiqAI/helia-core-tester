from __future__ import annotations

from pathlib import Path

from helia_core_tester.core.config import Config
from helia_core_tester.core.steps.generate import GenerateStep
from helia_core_tester.generation import reuse


def _stamp(descriptor: dict, **overrides) -> str:
    kwargs = {
        "case_name": "Add_s8_basic",
        "cpu": "cortex-m55",
        "suite": "int",
        "float_precision": "both",
        "seed": 500,
        "version_hash": "v1",
    }
    kwargs.update(overrides)
    return reuse.case_stamp(descriptor, **kwargs)


def _make_case(root: Path, family: str, name: str, stamp: str | None = None) -> Path:
    case_dir = root / family / name
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "descriptor.yaml").write_text(f"name: {name}\noperator: Add\n")
    (case_dir / f"{name}.tflite").write_bytes(b"\x00")
    if stamp is not None:
        reuse.write_stamp(case_dir, stamp)
    return case_dir


def test_stamp_tracks_every_generation_input() -> None:
    descriptor = {"name": "Add_s8_basic", "operator": "Add", "shape": [1, 4]}
    baseline = _stamp(descriptor)

    assert _stamp(dict(descriptor, shape=[1, 8])) != baseline
    assert _stamp(descriptor, cpu="cortex-m4") != baseline
    assert _stamp(descriptor, suite="float") != baseline
    assert _stamp(descriptor, float_precision="f32") != baseline
    assert _stamp(descriptor, seed=501) != baseline
    assert _stamp(descriptor, version_hash="v2") != baseline
    assert _stamp(dict(reversed(list(descriptor.items())))) == baseline


def test_generator_version_hash_covers_templates_and_sources(monkeypatch, tmp_path: Path) -> None:
    template = tmp_path / "templates" / "family" / "case.c.j2"
    template.parent.mkdir(parents=True)
    template.write_text("original")
    source = tmp_path / "gen.py"
    source.write_text("print()")

    monkeypatch.setattr(reuse, "_iter_generator_sources", lambda: iter([source, template]))
    monkeypatch.setattr(reuse, "_pinned_package_versions", lambda: {"tensorflow": "1.0"})

    monkeypatch.setattr(reuse, "_version_hash_cache", None, raising=False)
    baseline = reuse.generator_version_hash()

    template.write_text("edited")
    monkeypatch.setattr(reuse, "_version_hash_cache", None, raising=False)
    assert reuse.generator_version_hash() != baseline

    template.write_text("original")
    monkeypatch.setattr(reuse, "_version_hash_cache", None, raising=False)
    assert reuse.generator_version_hash() == baseline

    monkeypatch.setattr(reuse, "_pinned_package_versions", lambda: {"tensorflow": "2.0"})
    monkeypatch.setattr(reuse, "_version_hash_cache", None, raising=False)
    assert reuse.generator_version_hash() != baseline


def test_stamp_round_trip_and_invalidation(tmp_path: Path) -> None:
    case_dir = _make_case(tmp_path, "BasicMathFunctions", "Add_s8_basic", stamp="abc")

    assert reuse.read_stamp(case_dir) == "abc"
    assert reuse.case_artifacts_present(case_dir, "Add_s8_basic")

    reuse.clear_stamp(case_dir)
    assert reuse.read_stamp(case_dir) is None
    reuse.clear_stamp(case_dir)  # idempotent


def test_case_artifacts_present_rejects_a_gutted_case(tmp_path: Path) -> None:
    case_dir = _make_case(tmp_path, "BasicMathFunctions", "Add_s8_basic", stamp="abc")
    (case_dir / "Add_s8_basic.tflite").unlink()
    assert not reuse.case_artifacts_present(case_dir, "Add_s8_basic")

    (case_dir / "includes").mkdir()
    (case_dir / "includes" / "data.h").write_text("")
    assert reuse.case_artifacts_present(case_dir, "Add_s8_basic")

    (case_dir / "descriptor.yaml").unlink()
    assert not reuse.case_artifacts_present(case_dir, "Add_s8_basic")


def test_prune_drops_cases_outside_the_active_filter(tmp_path: Path) -> None:
    _make_case(tmp_path, "BasicMathFunctions", "Add_s8_basic")
    _make_case(tmp_path, "BasicMathFunctions", "Sub_s8_basic")
    _make_case(tmp_path, "PoolingFunctions", "MaxPool_s8_basic")
    (tmp_path / "manifest.json").write_text("{}")

    removed = reuse.prune_unlisted_cases(
        tmp_path, {"BasicMathFunctions/Add_s8_basic"}
    )

    assert removed == 2
    assert (tmp_path / "BasicMathFunctions" / "Add_s8_basic").is_dir()
    assert not (tmp_path / "BasicMathFunctions" / "Sub_s8_basic").exists()
    assert not (tmp_path / "PoolingFunctions").exists()
    assert (tmp_path / "manifest.json").is_file()


def _config_root(tmp_path: Path) -> Path:
    (tmp_path / "helia_core_tester" / "generation").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_generate_step_forwards_force_generate(tmp_path: Path) -> None:
    root = _config_root(tmp_path)

    default_cmd = GenerateStep(Config(project_root=root))._build_cmd(cpu="cortex-m55", suite="int")
    assert "--force-generate" not in default_cmd

    forced = Config(
        project_root=root,
        force_generate=True,
        _explicit_overrides={"project_root", "force_generate"},
    )
    assert "--force-generate" in GenerateStep(forced)._build_cmd(cpu="cortex-m55", suite="int")


def test_force_generate_honours_env_override(tmp_path: Path, monkeypatch) -> None:
    root = _config_root(tmp_path)
    monkeypatch.setenv("HELIA_CORE_TESTER_FORCE_GENERATE", "true")
    assert Config(project_root=root).force_generate is True
