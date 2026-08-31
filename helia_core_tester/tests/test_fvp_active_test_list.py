"""Regression coverage for helia-core-tester issue #66.

Build and run steps key their working trees on cpu+suite only
(``core/path_layout.py``), not on the generation filter (``--op``, ``--name``,
``--float-precision``, ...). So a filtered run in a tree a wider run already
built can still see that run's ``.elf`` binaries -- CMake never deletes the
output of a target it no longer defines -- and a ``--no-build`` run step then
discovers and executes them even though the current filter excludes them.

``manifest.json`` is rewritten from scratch by every generation run to hold
exactly the active filter's cases; it is the one artifact that survives across
the separate generate/build/run subprocess invocations, so it is what carries
the active test list between them. These tests pin two invariants built on
that list:

1. ``cmake_configure`` prunes any build-tree ``.elf`` that fell out of the
   active list before a (re)configure, so a filtered rebuild never leaves an
   excluded case's binary behind;
2. ``find_elves`` never returns a ``.elf`` outside the active list, even when
   nothing pruned it first (the ``--no-build`` run-step path).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helia_core_tester.fvp.cmake import active_test_list, cmake_configure, find_elves
from helia_core_tester.fvp.errors import FvpScriptError


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF")
    return path


def _write_manifest(generated_tests_dir: Path, names: list[str]) -> None:
    generated_tests_dir.mkdir(parents=True, exist_ok=True)
    (generated_tests_dir / "manifest.json").write_text(
        json.dumps({"tests": [{"name": name} for name in names]})
    )


def _stub_configure_subprocess(monkeypatch) -> None:
    monkeypatch.setattr(
        "helia_core_tester.fvp.cmake.subprocess.call",
        lambda *args, **kwargs: 0,
    )


# --------------------------------------------------------------------------- #
# active_test_list
# --------------------------------------------------------------------------- #


def test_active_test_list_reads_manifest_names(tmp_path: Path) -> None:
    generated_tests_dir = tmp_path / "generated_tests" / "int" / "cortex-m55"
    _write_manifest(generated_tests_dir, ["conv_s8_1x1", "add_s8_broadcast"])

    assert active_test_list(generated_tests_dir) == {"conv_s8_1x1", "add_s8_broadcast"}


def test_active_test_list_none_without_manifest(tmp_path: Path) -> None:
    assert active_test_list(tmp_path / "nope") is None
    assert active_test_list(None) is None


def test_active_test_list_empty_set_when_manifest_admits_nothing(tmp_path: Path) -> None:
    """A manifest that exists but lists zero tests (every descriptor
    capability-skipped for this cpu, e.g. the float suite on cortex-m0) is a
    real, empty active list -- distinct from "no manifest" -- so it must NOT
    be folded into None (which would fall back to unfiltered discovery)."""
    generated_tests_dir = tmp_path / "generated_tests" / "float" / "cortex-m0"
    _write_manifest(generated_tests_dir, [])

    result = active_test_list(generated_tests_dir)

    assert result == set()
    assert result is not None


def test_active_test_list_fails_closed_on_invalid_json(tmp_path: Path) -> None:
    """A corrupt manifest (crash mid-write, disk issue, ...) must not fall
    back to None -- silently disabling pruning/filtering at the exact moment
    something has already gone wrong is worse than raising loudly."""
    generated_tests_dir = tmp_path / "generated_tests" / "int" / "cortex-m55"
    generated_tests_dir.mkdir(parents=True, exist_ok=True)
    (generated_tests_dir / "manifest.json").write_text("{not valid json")

    with pytest.raises(FvpScriptError):
        active_test_list(generated_tests_dir)


def test_active_test_list_fails_closed_on_non_object_manifest(tmp_path: Path) -> None:
    generated_tests_dir = tmp_path / "generated_tests" / "int" / "cortex-m55"
    generated_tests_dir.mkdir(parents=True, exist_ok=True)
    (generated_tests_dir / "manifest.json").write_text(json.dumps(["not", "an", "object"]))

    with pytest.raises(FvpScriptError):
        active_test_list(generated_tests_dir)


def test_active_test_list_fails_closed_on_non_list_tests_field(tmp_path: Path) -> None:
    generated_tests_dir = tmp_path / "generated_tests" / "int" / "cortex-m55"
    generated_tests_dir.mkdir(parents=True, exist_ok=True)
    (generated_tests_dir / "manifest.json").write_text(json.dumps({"tests": "not-a-list"}))

    with pytest.raises(FvpScriptError):
        active_test_list(generated_tests_dir)


def test_active_test_list_fails_closed_on_non_object_test_entry(tmp_path: Path) -> None:
    generated_tests_dir = tmp_path / "generated_tests" / "int" / "cortex-m55"
    generated_tests_dir.mkdir(parents=True, exist_ok=True)
    (generated_tests_dir / "manifest.json").write_text(json.dumps({"tests": ["not-an-object"]}))

    with pytest.raises(FvpScriptError):
        active_test_list(generated_tests_dir)


def test_active_test_list_fails_closed_on_missing_tests_field(tmp_path: Path) -> None:
    """A manifest object with no 'tests' field at all is schema drift/
    corruption, not "zero tests admitted" -- must raise, not silently
    default to an empty active list (which would prune/refuse to run
    everything as if the filter legitimately admitted nothing)."""
    generated_tests_dir = tmp_path / "generated_tests" / "int" / "cortex-m55"
    generated_tests_dir.mkdir(parents=True, exist_ok=True)
    (generated_tests_dir / "manifest.json").write_text(json.dumps({"skipped": []}))

    with pytest.raises(FvpScriptError):
        active_test_list(generated_tests_dir)


def test_active_test_list_fails_closed_on_entry_with_no_name(tmp_path: Path) -> None:
    """An entry present in 'tests' but missing (or with an empty) 'name'
    must raise rather than being silently dropped from the active list --
    dropping it would mean its still-on-disk .elf gets pruned/never run as
    if the filter had excluded it, when it was actually admitted."""
    generated_tests_dir = tmp_path / "generated_tests" / "int" / "cortex-m55"
    generated_tests_dir.mkdir(parents=True, exist_ok=True)
    (generated_tests_dir / "manifest.json").write_text(
        json.dumps({"tests": [{"name": "add_s8_broadcast"}, {"operator": "Conv"}]})
    )

    with pytest.raises(FvpScriptError):
        active_test_list(generated_tests_dir)


# --------------------------------------------------------------------------- #
# cmake_configure pruning
# --------------------------------------------------------------------------- #


def test_cmake_configure_prunes_elf_outside_active_test_list(tmp_path: Path, monkeypatch) -> None:
    _stub_configure_subprocess(monkeypatch)

    build_dir = tmp_path / "build-int-cortex-m55-gcc"
    keep = _touch(build_dir / "tests" / "BasicMathFunctions" / "add_s8_broadcast.elf")
    stale = _touch(build_dir / "tests" / "ConvolutionFunctions" / "conv_s8_1x1.elf")

    generated_tests_dir = tmp_path / "generated_tests" / "int" / "cortex-m55"
    _write_manifest(generated_tests_dir, ["add_s8_broadcast"])

    cmake_configure(
        source_dir=tmp_path / "src",
        build_dir=build_dir,
        toolchain_file=tmp_path / "toolchain.cmake",
        cpu="cortex-m55",
        cmsis5=tmp_path / "CMSIS_5",
        optimization="-Ofast",
        extra_defs=[],
        generator=None,
        generated_tests_dir=generated_tests_dir,
        enable_coverage=False,
        verbosity=0,
        env={},
    )

    assert keep.exists()
    assert not stale.exists()


def test_cmake_configure_prunes_everything_when_manifest_admits_nothing(tmp_path: Path, monkeypatch) -> None:
    """A manifest with zero admitted tests must prune ALL build-tree ELFs, not
    fall back to keeping everything on disk (which is what happens if the
    empty active list were folded into None)."""
    _stub_configure_subprocess(monkeypatch)

    build_dir = tmp_path / "build-float-cortex-m0-gcc"
    stale = _touch(build_dir / "tests" / "ActivationFunctions" / "nn_activation_float_tanh_f16.elf")

    generated_tests_dir = tmp_path / "generated_tests" / "float" / "cortex-m0"
    _write_manifest(generated_tests_dir, [])

    cmake_configure(
        source_dir=tmp_path / "src",
        build_dir=build_dir,
        toolchain_file=tmp_path / "toolchain.cmake",
        cpu="cortex-m0",
        cmsis5=tmp_path / "CMSIS_5",
        optimization="-Ofast",
        extra_defs=[],
        generator=None,
        generated_tests_dir=generated_tests_dir,
        enable_coverage=False,
        verbosity=0,
        env={},
    )

    assert not stale.exists()


def test_cmake_configure_keeps_all_elves_without_manifest(tmp_path: Path, monkeypatch) -> None:
    _stub_configure_subprocess(monkeypatch)

    build_dir = tmp_path / "build-int-cortex-m55-gcc"
    elf = _touch(build_dir / "tests" / "ConvolutionFunctions" / "conv_s8_1x1.elf")

    cmake_configure(
        source_dir=tmp_path / "src",
        build_dir=build_dir,
        toolchain_file=tmp_path / "toolchain.cmake",
        cpu="cortex-m55",
        cmsis5=tmp_path / "CMSIS_5",
        optimization="-Ofast",
        extra_defs=[],
        generator=None,
        generated_tests_dir=None,
        enable_coverage=False,
        verbosity=0,
        env={},
    )

    assert elf.exists()


# --------------------------------------------------------------------------- #
# find_elves discovery filter
# --------------------------------------------------------------------------- #


def test_find_elves_ignores_elf_outside_active_test_list(tmp_path: Path) -> None:
    build_dir = tmp_path / "build-int-cortex-m55-gcc"
    _touch(build_dir / "tests" / "BasicMathFunctions" / "add_s8_broadcast.elf")
    stale = _touch(build_dir / "tests" / "ConvolutionFunctions" / "conv_s8_1x1.elf")

    generated_tests_dir = tmp_path / "generated_tests" / "int" / "cortex-m55"
    _write_manifest(generated_tests_dir, ["add_s8_broadcast"])

    found = {p.stem for p in find_elves(build_dir, active_test_list(generated_tests_dir))}

    assert found == {"add_s8_broadcast"}
    # Discovery only filters; a --no-build run step that never pruned still
    # leaves the stale binary physically present.
    assert stale.exists()


def test_find_elves_unfiltered_without_active_test_list(tmp_path: Path) -> None:
    build_dir = tmp_path / "build-int-cortex-m55-gcc"
    _touch(build_dir / "tests" / "A" / "one.elf")
    _touch(build_dir / "tests" / "B" / "two.elf")

    assert active_test_list(tmp_path / "nope") is None
    assert {p.stem for p in find_elves(build_dir, None)} == {"one", "two"}


def test_find_elves_empty_when_manifest_admits_nothing(tmp_path: Path) -> None:
    """Must find nothing, not fall back to unfiltered discovery, when the
    manifest exists but legitimately admitted zero cases."""
    build_dir = tmp_path / "build-float-cortex-m0-gcc"
    _touch(build_dir / "tests" / "ActivationFunctions" / "nn_activation_float_tanh_f16.elf")

    generated_tests_dir = tmp_path / "generated_tests" / "float" / "cortex-m0"
    _write_manifest(generated_tests_dir, [])

    assert find_elves(build_dir, active_test_list(generated_tests_dir)) == []
