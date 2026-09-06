from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from helia_core_tester.core.config import Config
from helia_core_tester.core.steps.generate import GenerateStep
from helia_core_tester.generation import reuse


def _stamp(descriptor: dict, **overrides) -> str:
    kwargs = {
        "case_name": "Add_s8_basic",
        "cpu": "cortex-m55",
        "suite": "int",
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
    (case_dir / f"{name}.c").write_text("void run(void) {}\n")
    (case_dir / "CMakeLists.txt").write_text(f"add_test({name})\n")
    (case_dir / "includes").mkdir(exist_ok=True)
    (case_dir / "includes" / "data.h").write_text("static const int x = 1;\n")
    if stamp is not None:
        reuse.write_stamp(case_dir, stamp)
    return case_dir


def _fake_checkout(root: Path) -> Path:
    (root / "Include").mkdir(parents=True, exist_ok=True)
    (root / "Include" / "arm_nnfunctions.h").write_text(
        "int32_t arm_lstm_unidirectional_s8_temp1_get_buffer_size(void);\n"
    )
    data_dir = root / "Tests" / "UnitTest" / "TestCases" / "TestData" / "lstm_1"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "output.h").write_text("const int8_t lstm_1_output[] = {1, 2, 3};\n")
    return root


def _fake_git(root: Path, *, status: str):
    """Stand in for git over a checkout that is its own working tree top."""

    def _git_output(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "--show-toplevel"):
            return f"{root}\n"
        if args[0] == "rev-parse":
            return "18a89ff\n"
        return status

    return _git_output


def test_stamp_tracks_every_generation_input() -> None:
    descriptor = {"name": "Add_s8_basic", "operator": "Add", "shape": [1, 4]}
    baseline = _stamp(descriptor)

    assert _stamp(dict(descriptor, shape=[1, 8])) != baseline
    assert _stamp(descriptor, cpu="cortex-m4") != baseline
    assert _stamp(descriptor, suite="float") != baseline
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
    monkeypatch.setattr(reuse, "_environment_identity", lambda: {"lock_sha256": "aa"})

    monkeypatch.setattr(reuse, "_version_hash_cache", None, raising=False)
    baseline = reuse.generator_version_hash()

    template.write_text("edited")
    monkeypatch.setattr(reuse, "_version_hash_cache", None, raising=False)
    assert reuse.generator_version_hash() != baseline

    template.write_text("original")
    monkeypatch.setattr(reuse, "_version_hash_cache", None, raising=False)
    assert reuse.generator_version_hash() == baseline

    monkeypatch.setattr(reuse, "_environment_identity", lambda: {"lock_sha256": "bb"})
    monkeypatch.setattr(reuse, "_version_hash_cache", None, raising=False)
    assert reuse.generator_version_hash() != baseline


def test_stamp_round_trip_and_invalidation(tmp_path: Path) -> None:
    case_dir = _make_case(tmp_path, "BasicMathFunctions", "Add_s8_basic", stamp="abc")

    assert reuse.read_stamp(case_dir)["stamp"] == "abc"
    assert reuse.case_reusable(case_dir, "abc")
    assert not reuse.case_reusable(case_dir, "a-different-stamp")

    reuse.reset_case_dir(case_dir)
    assert not case_dir.exists()
    assert reuse.read_stamp(case_dir) is None
    assert not reuse.case_reusable(case_dir, "abc")
    reuse.reset_case_dir(case_dir)  # idempotent


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda d: (d / "Add_s8_basic.tflite").unlink(), id="tflite-deleted"),
        pytest.param(lambda d: (d / "Add_s8_basic.c").unlink(), id="c-deleted"),
        pytest.param(
            lambda d: (d / "Add_s8_basic.c").write_text("void run(vo"), id="c-truncated"
        ),
        pytest.param(lambda d: (d / "includes" / "data.h").unlink(), id="header-deleted"),
        pytest.param(
            lambda d: (d / "includes" / "data.h").write_text("static const int"),
            id="header-truncated",
        ),
        pytest.param(lambda d: (d / "CMakeLists.txt").unlink(), id="cmakelists-deleted"),
        pytest.param(
            lambda d: (d / "CMakeLists.txt").write_text("add_te"), id="cmakelists-truncated"
        ),
        pytest.param(lambda d: (d / "descriptor.yaml").unlink(), id="descriptor-deleted"),
    ],
)
def test_a_damaged_artifact_stops_the_case_being_reusable(tmp_path: Path, corrupt) -> None:
    case_dir = _make_case(tmp_path, "BasicMathFunctions", "Add_s8_basic", stamp="abc")
    assert reuse.case_reusable(case_dir, "abc")

    corrupt(case_dir)

    assert not reuse.case_reusable(case_dir, "abc")


def test_an_emptied_case_directory_is_not_reusable(tmp_path: Path) -> None:
    case_dir = _make_case(tmp_path, "BasicMathFunctions", "Add_s8_basic", stamp="abc")
    for path in list(case_dir.rglob("*")):
        if path.is_file() and path.name != reuse.STAMP_FILENAME:
            path.unlink()

    assert not reuse.case_reusable(case_dir, "abc")


def test_checkout_content_is_the_identity_when_the_root_is_not_a_git_tree(
    monkeypatch, tmp_path: Path
) -> None:
    root = _fake_checkout(tmp_path / "ns-cmsis-nn")
    monkeypatch.setattr(reuse, "resolve_cmsis_nn_root", lambda: root)
    monkeypatch.setattr(reuse, "_git_output", lambda *args: None)

    monkeypatch.setattr(reuse, "_checkout_identity_cache", None)
    baseline = reuse.cmsis_nn_checkout_identity()
    assert baseline["state"] == "content"

    golden = root / "Tests" / "UnitTest" / "TestCases" / "TestData" / "lstm_1" / "output.h"
    golden.write_text(golden.read_text().replace("{1, 2, 3}", "{1, 2, 4}"))
    monkeypatch.setattr(reuse, "_checkout_identity_cache", None)
    perturbed = reuse.cmsis_nn_checkout_identity()
    assert perturbed != baseline

    header = root / "Include" / "arm_nnfunctions.h"
    header.write_text("/* sizers hidden */\n")
    monkeypatch.setattr(reuse, "_checkout_identity_cache", None)
    assert reuse.cmsis_nn_checkout_identity() != perturbed


def test_a_clean_git_checkout_is_identified_by_its_commit(monkeypatch, tmp_path: Path) -> None:
    root = _fake_checkout(tmp_path / "ns-cmsis-nn")
    monkeypatch.setattr(reuse, "resolve_cmsis_nn_root", lambda: root)
    monkeypatch.setattr(reuse, "_git_output", _fake_git(root, status=""))

    monkeypatch.setattr(reuse, "_checkout_identity_cache", None)
    assert reuse.cmsis_nn_checkout_identity() == {
        "state": "git-clean",
        "commit": "18a89ff",
    }


def test_a_dirty_git_checkout_falls_back_to_content(monkeypatch, tmp_path: Path) -> None:
    root = _fake_checkout(tmp_path / "ns-cmsis-nn")
    monkeypatch.setattr(reuse, "resolve_cmsis_nn_root", lambda: root)
    monkeypatch.setattr(reuse, "_git_output", _fake_git(root, status=" M Include/x.h\n"))

    monkeypatch.setattr(reuse, "_checkout_identity_cache", None)
    baseline = reuse.cmsis_nn_checkout_identity()
    assert baseline["state"] == "git-dirty"
    assert baseline["commit"] == "18a89ff"

    golden = root / "Tests" / "UnitTest" / "TestCases" / "TestData" / "lstm_1" / "output.h"
    golden.write_text(golden.read_text().replace("{1, 2, 3}", "{1, 2, 4}"))
    monkeypatch.setattr(reuse, "_checkout_identity_cache", None)
    assert reuse.cmsis_nn_checkout_identity() != baseline


def test_stamp_tracks_the_resolved_checkout(monkeypatch) -> None:
    descriptor = {"name": "lstm_1", "operator": "UnidirectionalSequenceLSTM"}

    monkeypatch.setattr(
        reuse, "cmsis_nn_checkout_identity", lambda: {"state": "git-clean", "commit": "aaa"}
    )
    baseline = _stamp(descriptor)

    monkeypatch.setattr(
        reuse, "cmsis_nn_checkout_identity", lambda: {"state": "git-clean", "commit": "bbb"}
    )
    assert _stamp(descriptor) != baseline

    monkeypatch.setattr(reuse, "cmsis_nn_checkout_identity", lambda: {"state": "absent"})
    assert _stamp(descriptor) != baseline


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


def test_an_empty_filter_prunes_every_case(tmp_path: Path) -> None:
    _make_case(tmp_path, "BasicMathFunctions", "Add_s8_basic")
    _make_case(tmp_path, "PoolingFunctions", "MaxPool_s8_basic")

    assert reuse.prune_unlisted_cases(tmp_path, set()) == 2
    assert not (tmp_path / "BasicMathFunctions").exists()
    assert not (tmp_path / "PoolingFunctions").exists()


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


def test_reset_case_dir_clears_output_from_a_previous_shape(tmp_path: Path) -> None:
    case_dir = _make_case(tmp_path, "BasicMathFunctions", "Add_s8_basic", stamp="abc")
    stale = case_dir / "Add_s8_basic_mul.c"
    stale.write_text("void run(void) {}\n")

    reuse.reset_case_dir(case_dir)

    assert not case_dir.exists()
    assert not stale.exists()
    reuse.reset_case_dir(case_dir)  # idempotent


def test_a_non_git_checkout_inside_another_repo_is_identified_by_content(
    monkeypatch, tmp_path: Path
) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    subprocess.run(["git", "-C", str(outer), "commit", "-q", "--allow-empty", "-m", "seed"],
                   check=True,
                   env={**os.environ,
                        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
    root = _fake_checkout(outer / "scratch" / "ns-cmsis-nn")
    monkeypatch.setattr(reuse, "resolve_cmsis_nn_root", lambda: root)

    monkeypatch.setattr(reuse, "_checkout_identity_cache", None)
    baseline = reuse.cmsis_nn_checkout_identity()
    assert baseline["state"] == "content"

    header = root / "Include" / "arm_nnfunctions.h"
    header.write_text("/* sizers hidden */\n")
    monkeypatch.setattr(reuse, "_checkout_identity_cache", None)
    assert reuse.cmsis_nn_checkout_identity() != baseline
