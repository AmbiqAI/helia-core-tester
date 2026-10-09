"""Registry and descriptor parse caches."""

from __future__ import annotations

import os

from helia_core_tester.generation.io import descriptors
from helia_core_tester.hardware import kernel_registry

_ABS = (
    "operator: Abs\n"
    "name: {name}\n"
    "activation_dtype: S8\n"
    "weight_dtype: S8\n"
    "input_shape: [1, 4]\n"
)

_REGISTRY = (
    "kernels:\n"
    "  - kernel_id: {kid}\n"
    "    family: BasicMathFunctions\n"
    "    operator: Abs\n"
    "    dtype: S8\n"
    "    cmsis_function: arm_abs_s8\n"
)


def _registry(tmp_path, kid):
    path = tmp_path / "assets" / "kernel_registry.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(_REGISTRY.format(kid=kid))
    return path


def _rewrite_same_stat(path, text):
    stat = path.stat()
    path.write_text(text)
    assert path.stat().st_size == stat.st_size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_registry_cache_hit(tmp_path):
    _registry(tmp_path, 7)
    first = kernel_registry.load_kernel_registry(tmp_path)
    hits = kernel_registry._parse_registry.cache_info().hits
    second = kernel_registry.load_kernel_registry(tmp_path)
    assert kernel_registry._parse_registry.cache_info().hits == hits + 1
    assert first == second
    assert first is not second


def test_registry_edit_invalidates(tmp_path):
    path = _registry(tmp_path, 7)
    assert kernel_registry.load_kernel_registry(tmp_path)[0].kernel_id == 7
    _rewrite_same_stat(path, _REGISTRY.format(kid=8))
    assert kernel_registry.load_kernel_registry(tmp_path)[0].kernel_id == 8


def test_registry_list_isolated(tmp_path):
    _registry(tmp_path, 7)
    kernel_registry.load_kernel_registry(tmp_path).clear()
    assert len(kernel_registry.load_kernel_registry(tmp_path)) == 1


def test_descriptor_cache_hit(tmp_path):
    path = tmp_path / "abs.yaml"
    path.write_text(_ABS.format(name="abs_a_s8"))
    first = descriptors.load_descriptor(str(path))
    hits = descriptors._parse_yaml_docs.cache_info().hits
    second = descriptors.load_descriptor(str(path))
    assert descriptors._parse_yaml_docs.cache_info().hits == hits + 1
    assert first == second


def test_descriptor_edit_invalidates(tmp_path):
    path = tmp_path / "abs.yaml"
    path.write_text(_ABS.format(name="abs_a_s8"))
    assert descriptors.load_descriptor(str(path))[0]["name"] == "abs_a_s8"
    # Same size and mtime, new bytes.
    _rewrite_same_stat(path, _ABS.format(name="abs_b_s8"))
    assert descriptors.load_descriptor(str(path))[0]["name"] == "abs_b_s8"


def test_descriptor_mutation_isolated(tmp_path):
    path = tmp_path / "abs.yaml"
    path.write_text(_ABS.format(name="abs_a_s8"))
    first = descriptors.load_descriptor(str(path))
    first[0]["name"] = "mutated"
    first[0]["input_shape"].append(99)
    first[0]["hint"]["kernel"] = "x"
    second = descriptors.load_descriptor(str(path))
    assert second[0]["name"] == "abs_a_s8"
    assert second[0]["input_shape"] == [1, 4]
    assert "kernel" not in second[0]["hint"]


def test_load_all_isolated(tmp_path):
    (tmp_path / "BasicMathFunctions").mkdir()
    path = tmp_path / "BasicMathFunctions" / "abs.yaml"
    path.write_text(_ABS.format(name="abs_a_s8"))
    first = descriptors.load_all_descriptors(str(tmp_path))
    first[0]["input_shape"][0] = 5
    second = descriptors.load_all_descriptors(str(tmp_path))
    assert second[0]["input_shape"] == [1, 4]
