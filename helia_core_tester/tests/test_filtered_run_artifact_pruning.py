"""Regression coverage for helia-core-tester issue #66.

A filtered run (``--float-precision f16``, ``--name ...``, ``--limit``, ...)
rewrites ``manifest.json`` / ``tests.cmake`` from the current run's entries
only, but historically left on disk every per-case directory a *prior*, wider
run generated in the same working tree. Those orphans were then re-counted by
the descriptor-aware reporting pass (34 reported cases where the filter admits
17) and could be compiled against a newer kernel.

These tests pin two invariants:

1. after a filtered generate, the case directories on disk under
   ``generated_tests/<suite>/<cpu>/`` are exactly the set the active filter
   admits -- no matter what an earlier run put there;
2. ELF discovery for a run ignores build-tree ``*.elf`` files that are not in
   the active run's manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import helia_core_tester.generation.test_ops as generation_module
from helia_core_tester.fvp.cmake import _prune_stale_test_elves, find_elves, manifest_test_names


# --------------------------------------------------------------------------- #
# generation-side pruning
# --------------------------------------------------------------------------- #

_F32_CASE = {
    "name": "nn_activation_float_tanh_f32",
    "operator": "NNActivationFloat",
    "tensor_dtypes": {"input": "FP32", "output": "FP32"},
    "resolved_tensor_dtypes": {"input": "FP32", "output": "FP32"},
    "_family": "ActivationFunctions",
    "_parity_kind": "cmsis",
    "_source_family": "ActivationFunctions",
    "_source_stem": "nn_activation_float",
    "_source_relpath": "ActivationFunctions/nn_activation_float.yaml",
    "_descriptor_suite": "float",
    "input_shape": [1, 7],
}
_F16_CASE = {
    **_F32_CASE,
    "name": "nn_activation_float_tanh_f16",
    "tensor_dtypes": {"input": "FP16", "output": "FP16"},
    "resolved_tensor_dtypes": {"input": "FP16", "output": "FP16"},
}


def _filters(generated_tests_dir: Path, *, float_precision: str, name: str | None = None) -> dict[str, object]:
    return {
        "op": None,
        "dtype": None,
        "wtype": None,
        "name": name,
        "limit": None,
        "seed": 123,
        "cpu": "cortex-m55",
        "suite": "float",
        "float_precision": float_precision,
        "generated_tests_dir": str(generated_tests_dir),
    }


def _fake_generate_test(desc, out_dir, seed=None, cpu="cortex-m55", conversion_failures=None, generation_failures=None):
    """Mimic ``generate_test``'s on-disk footprint: a case dir with a
    ``descriptor.yaml`` marker, a ``.tflite`` blob, and one ``*_*.c`` source so
    the entry is 'runnable' and reaches ``tests.cmake``."""
    test_dir = Path(out_dir) / desc["_family"] / desc["name"]
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "descriptor.yaml").write_text("name: %s\n" % desc["name"])
    (test_dir / f"{desc['name']}.tflite").write_bytes(b"\x01")
    (test_dir / f"{desc['name']}_test.c").write_text("int main(void){return 0;}\n")


def _generate(monkeypatch: pytest.MonkeyPatch, repo_root: Path, descriptors: list[dict], filters: dict) -> None:
    monkeypatch.setattr(generation_module, "find_repo_root", lambda: repo_root)
    monkeypatch.setattr(generation_module, "find_descriptors_dir", lambda: repo_root / "assets" / "descriptors")
    monkeypatch.setattr(generation_module, "load_all_descriptors", lambda _path: [dict(d) for d in descriptors])
    monkeypatch.setattr(generation_module, "generate_test", _fake_generate_test)
    generation_module.test_generation(filters)


def _case_dirs_on_disk(generated_tests_dir: Path) -> set[str]:
    return {p.parent.name for p in generated_tests_dir.rglob("descriptor.yaml")}


def _manifest_case_names(generated_tests_dir: Path) -> set[str]:
    manifest = json.loads((generated_tests_dir / "manifest.json").read_text())
    return {entry["name"] for entry in manifest["tests"]}


def test_precision_filtered_regen_prunes_stale_case_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    generated_tests_dir = repo_root / "artifacts" / "generated_tests" / "float" / "cortex-m55"

    # Run 1: the wide f32 pass populates the tree.
    _generate(monkeypatch, repo_root, [_F32_CASE], _filters(generated_tests_dir, float_precision="f32"))
    assert _case_dirs_on_disk(generated_tests_dir) == {"nn_activation_float_tanh_f32"}

    # Run 2: a narrower f16 pass in the SAME tree. Only the f16 case is in this
    # run's filter; the f32 case dir from run 1 must not survive.
    _generate(
        monkeypatch,
        repo_root,
        [_F32_CASE, _F16_CASE],
        _filters(generated_tests_dir, float_precision="f16"),
    )

    assert _manifest_case_names(generated_tests_dir) == {"nn_activation_float_tanh_f16"}
    assert _case_dirs_on_disk(generated_tests_dir) == {"nn_activation_float_tanh_f16"}
    # The invariant the reporting pass depends on: tree == manifest.
    assert _case_dirs_on_disk(generated_tests_dir) == _manifest_case_names(generated_tests_dir)


def test_name_filtered_regen_prunes_previously_generated_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    generated_tests_dir = repo_root / "artifacts" / "generated_tests" / "float" / "cortex-m55"
    both = [_F32_CASE, _F16_CASE]

    _generate(
        monkeypatch, repo_root, both,
        _filters(generated_tests_dir, float_precision="both", name="nn_activation_float_tanh_f32"),
    )
    assert _case_dirs_on_disk(generated_tests_dir) == {"nn_activation_float_tanh_f32"}

    _generate(
        monkeypatch, repo_root, both,
        _filters(generated_tests_dir, float_precision="both", name="nn_activation_float_tanh_f16"),
    )
    assert _case_dirs_on_disk(generated_tests_dir) == {"nn_activation_float_tanh_f16"}


# --------------------------------------------------------------------------- #
# ELF-discovery side
# --------------------------------------------------------------------------- #

def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF")
    return path


def test_find_elves_ignores_elf_absent_from_manifest(tmp_path: Path) -> None:
    build_dir = tmp_path / "artifacts" / "build-float-cortex-m55-gcc"
    _touch(build_dir / "tests" / "ActivationFunctions" / "nn_activation_float_tanh_f16.elf")
    stale = _touch(build_dir / "tests" / "ActivationFunctions" / "nn_activation_float_tanh_f32.elf")

    generated_tests_dir = tmp_path / "artifacts" / "generated_tests" / "float" / "cortex-m55"
    generated_tests_dir.mkdir(parents=True, exist_ok=True)
    (generated_tests_dir / "manifest.json").write_text(
        json.dumps({"tests": [{"name": "nn_activation_float_tanh_f16"}]})
    )

    allowed = manifest_test_names(generated_tests_dir)
    found = {p.stem for p in find_elves(build_dir, allowed)}

    assert found == {"nn_activation_float_tanh_f16"}
    assert stale.exists()  # discovery is filtered; it does not delete


def test_find_elves_unfiltered_without_manifest(tmp_path: Path) -> None:
    build_dir = tmp_path / "artifacts" / "build-int-cortex-m55-gcc"
    _touch(build_dir / "tests" / "A" / "one.elf")
    _touch(build_dir / "tests" / "B" / "two.elf")

    assert manifest_test_names(tmp_path / "nope") is None
    assert {p.stem for p in find_elves(build_dir, None)} == {"one", "two"}


def test_prune_stale_test_elves_deletes_only_out_of_filter_binaries(tmp_path: Path) -> None:
    build_dir = tmp_path / "artifacts" / "build-float-cortex-m55-gcc"
    keep = _touch(build_dir / "tests" / "ActivationFunctions" / "nn_activation_float_tanh_f16.elf")
    stale = _touch(build_dir / "tests" / "ActivationFunctions" / "nn_activation_float_tanh_f32.elf")

    removed = _prune_stale_test_elves(build_dir, {"nn_activation_float_tanh_f16"})

    assert removed == [stale]
    assert keep.exists()
    assert not stale.exists()


def test_prune_stale_test_elves_noop_without_manifest(tmp_path: Path) -> None:
    build_dir = tmp_path / "artifacts" / "build-int-cortex-m55-gcc"
    elf = _touch(build_dir / "tests" / "A" / "one.elf")

    assert _prune_stale_test_elves(build_dir, None) == []
    assert elf.exists()
