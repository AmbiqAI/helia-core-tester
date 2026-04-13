from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import helia_core_tester.generation.test_ops as generation_module
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops import get_op_map


_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*_(s8|s16|s4|s32)$")
_FLOAT_REFERENCE_NAMES = {
    "quantize_fp32_to_s8_basic",
    "dequantize_s8_to_fp32_basic",
}
_BANNED_NAME_PATTERNS = (
    re.compile(r"^avgpooling"),
    re.compile(r"^maxpooling"),
    re.compile(r"(^|_)conv2d($|_)"),
    re.compile(r"(^|_)depthwise_conv2d($|_)"),
    re.compile(r"(^|_)matmul_batch($|_)"),
    re.compile(r"(^|_)stridedslice($|_)"),
    re.compile(r"(^|_)leakyrelu($|_)"),
    re.compile(r"(^|_)reducemax($|_)"),
    re.compile(r"(^|_)reducemin($|_)"),
    re.compile(r"(^|_)crs($|_)"),
)
_OLD_OPERATOR_NAMES = (
    "Conv2D",
    "DepthwiseConv2D",
    "MatMul",
    "Pooling",
    "HardSwish",
    "Elementwise",
    "Equal",
    "NotEqual",
    "Greater",
    "GreaterEqual",
    "Less",
    "LessEqual",
    "LSTM",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _descriptor_dir() -> Path:
    return _repo_root() / "assets" / "descriptors"


def test_descriptor_names_follow_canonical_contract() -> None:
    descriptors = load_all_descriptors(str(_descriptor_dir()))
    assert descriptors

    names = [desc["name"] for desc in descriptors]
    assert len(names) == len(set(names))

    for desc in descriptors:
        name = desc["name"]
        source_stem = desc["_source_stem"]
        source_relpath = desc["_source_relpath"]
        source_family = desc["_source_family"]
        family = desc["_family"]

        assert _NAME_PATTERN.match(name) or name in _FLOAT_REFERENCE_NAMES, name
        assert not re.search(r"_(s8|s16|s4|s32)_case_", name), name
        assert not re.search(r"_basic_(s8|s16|s4|s32)$", name), name
        assert not any(pattern.search(name) for pattern in _BANNED_NAME_PATTERNS), name
        assert "/" in source_relpath, source_relpath
        assert source_family == family

        assert name.startswith(f"{source_stem}_"), (source_stem, name)


def test_schema_is_valid_and_excludes_old_operator_names() -> None:
    schema_path = _repo_root() / "helia_core_tester" / "generation" / "descriptors" / "schema.json"
    schema = json.loads(schema_path.read_text())
    operators = schema["properties"]["operator"]["enum"]

    for operator_name in _OLD_OPERATOR_NAMES:
        assert operator_name not in operators

    for operator_name in ("Convolve", "DepthwiseConv", "BatchMatMul", "AvgPool", "MaxPool", "Comparison"):
        assert operator_name in operators


def test_dead_old_op_modules_removed_from_registry_and_tree() -> None:
    op_map = get_op_map()
    for old_name in ("Equal", "NotEqual", "Greater", "GreaterEqual", "Less", "LessEqual", "Elementwise"):
        assert old_name not in op_map

    ops_dir = _repo_root() / "helia_core_tester" / "generation" / "ops"
    for basename in ("equal.py", "not_equal.py", "greater.py", "greater_equal.py", "less.py", "less_equal.py", "elementwise.py"):
        assert not any(ops_dir.rglob(basename))


def test_generation_name_filter_uses_renamed_descriptor_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    generated_tests_dir = repo_root / "artifacts" / "generated_tests" / "cortex-m55"
    renamed_name = "avg_pool_same_pool5x6_stride5x9_s8"
    descriptors = [
        {
            "name": renamed_name,
            "operator": "AvgPool",
            "activation_dtype": "S8",
            "weight_dtype": "S8",
            "_source_stem": "avg_pool",
            "_source_family": "PoolingFunctions",
            "_source_relpath": "PoolingFunctions/avg_pool.yaml",
            "_family": "PoolingFunctions",
            "_parity_kind": "cmsis",
        }
    ]
    generated: list[str] = []

    monkeypatch.setattr(generation_module, "find_repo_root", lambda: repo_root)
    monkeypatch.setattr(generation_module, "find_descriptors_dir", lambda: repo_root / "assets" / "descriptors")
    monkeypatch.setattr(generation_module, "load_all_descriptors", lambda _path: descriptors)

    def _fake_generate_test(desc, out_dir, seed=None, cpu="cortex-m55", conversion_failures=None, generation_failures=None):
        generated.append(desc["name"])
        test_dir = Path(out_dir) / desc["_family"] / desc["name"]
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / f"{desc['name']}.tflite").write_bytes(b"\x01")

    monkeypatch.setattr(generation_module, "generate_test", _fake_generate_test)

    generation_module.test_generation(
        {
            "op": None,
            "dtype": None,
            "wtype": None,
            "name": renamed_name,
            "limit": None,
            "seed": 123,
            "cpu": "cortex-m55",
            "generated_tests_dir": str(generated_tests_dir),
        }
    )

    assert generated == [renamed_name]
    manifest = json.loads((generated_tests_dir / "manifest.json").read_text())
    assert manifest["tests"][0]["family"] == "PoolingFunctions"
    assert manifest["tests"][0]["relative_test_dir"] == "PoolingFunctions/avg_pool_same_pool5x6_stride5x9_s8"

    generated.clear()
    with pytest.raises(AssertionError, match="No TFLite models were generated"):
        generation_module.test_generation(
            {
                "op": None,
                "dtype": None,
                "wtype": None,
                "name": "avgpooling_s8",
                "limit": None,
                "seed": 123,
                "cpu": "cortex-m55",
                "generated_tests_dir": str(generated_tests_dir),
            }
        )
    assert generated == []
