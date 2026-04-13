from __future__ import annotations

import json
from pathlib import Path
import re
import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.catalog import get_operator_spec, iter_operator_specs
from helia_core_tester.scripts.scaffold_operator import create_operator_skeleton, operator_to_module_basename
from helia_core_tester.scripts.validate_operator_catalog import validate_operator_contracts


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _seed_scaffold_repo(tmp_path: Path) -> None:
    repo_root = _repo_root()
    for relpath in (
        "helia_core_tester/generation/ops/catalog.py",
        "helia_core_tester/generation/io/descriptors.py",
        "helia_core_tester/generation/descriptors/schema.json",
    ):
        src = repo_root / relpath
        dst = tmp_path / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())


def test_operator_catalog_contracts_validate() -> None:
    assert validate_operator_contracts(_repo_root()) == []


def test_loaded_descriptors_carry_grouped_catalog_metadata() -> None:
    descriptors = load_all_descriptors(str(_repo_root() / "assets" / "descriptors"))
    assert descriptors

    for desc in descriptors:
        spec = get_operator_spec(desc["operator"])
        assert desc["_source_relpath"] == spec.descriptor_relpath
        assert desc["_source_family"] == spec.family
        assert desc["_family"] == spec.family
        assert desc["_parity_kind"] == spec.parity_kind
        assert desc["_source_relpath"].endswith(f"{desc['_source_stem']}.yaml")


def test_extension_ops_are_isolated_in_tester_extensions() -> None:
    extension_operators = {
        spec.operator
        for spec in iter_operator_specs()
        if spec.parity_kind == "extension"
    }
    assert extension_operators == {"Fill", "Squeeze", "VariableUpdate"}
    for operator in extension_operators:
        spec = get_operator_spec(operator)
        assert spec.family == "TesterExtensions"


def test_scaffold_operator_creates_grouped_skeleton(tmp_path: Path) -> None:
    _seed_scaffold_repo(tmp_path)
    created = create_operator_skeleton(
        tmp_path,
        operator="ExampleOp",
        family="BasicMathFunctions",
        parity_kind="cmsis",
        descriptor_profile="single_input_unary",
    )

    assert created["module"] == tmp_path / "helia_core_tester" / "generation" / "ops" / "BasicMathFunctions" / "example_op.py"
    assert created["descriptor"] == tmp_path / "assets" / "descriptors" / "BasicMathFunctions" / "example_op.yaml"
    assert created["header_template"] == tmp_path / "assets" / "templates" / "BasicMathFunctions" / "example_op" / "example_op.h.j2"
    assert created["source_template"] == tmp_path / "assets" / "templates" / "BasicMathFunctions" / "example_op" / "example_op.c.j2"
    assert created["module"].exists()
    assert created["descriptor"].exists()
    assert created["header_template"].exists()
    assert created["source_template"].exists()
    assert created["catalog"].exists()
    assert created["descriptor_metadata"].exists()
    assert created["schema"].exists()

    module_text = created["module"].read_text()
    assert "def build_example_op(" in module_text
    assert "build_unary_same_shape_op" in module_text
    assert 'self.tensor_litert_dtype("input")' in module_text
    assert 'output_dtype=self.tensor_litert_dtype("output")' in module_text

    catalog_text = created["catalog"].read_text()
    assert '"ExampleOp": _spec("ExampleOp", "BasicMathFunctions", "example_op", "OpExampleOp"' in catalog_text

    descriptors_text = created["descriptor_metadata"].read_text()
    assert "'ExampleOp': 'single_input_unary'" in descriptors_text

    schema = json.loads(created["schema"].read_text())
    assert "ExampleOp" in schema["properties"]["operator"]["enum"]


def test_scaffold_operator_rejects_duplicates_without_overwrite(tmp_path: Path) -> None:
    _seed_scaffold_repo(tmp_path)
    create_operator_skeleton(
        tmp_path,
        operator="ExampleOp",
        family="BasicMathFunctions",
        parity_kind="cmsis",
        descriptor_profile="single_input_unary",
    )

    with pytest.raises(FileExistsError):
        create_operator_skeleton(
            tmp_path,
            operator="ExampleOp",
            family="BasicMathFunctions",
            parity_kind="cmsis",
            descriptor_profile="single_input_unary",
        )


def test_operator_to_module_basename_handles_acronyms() -> None:
    assert operator_to_module_basename("LSTMUnidirectional") == "lstm_unidirectional"
    assert operator_to_module_basename("NNActivationS16") == "nn_activation_s16"


def test_ops_use_grouped_template_paths_directly() -> None:
    ops_root = _repo_root() / "helia_core_tester" / "generation" / "ops"
    pattern = re.compile(r'"([A-Za-z0-9_]+/[A-Za-z0-9_./-]+\.j2)"')

    for path in ops_root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        for literal in pattern.findall(path.read_text()):
            first = literal.split("/", 1)[0]
            if first == "common":
                continue
            assert first.endswith("Functions") or first == "TesterExtensions", (path, literal)


def test_selected_ops_define_local_litert_wrappers() -> None:
    ops_root = _repo_root() / "helia_core_tester" / "generation" / "ops" / "BasicMathFunctions"
    expected_wrappers = {
        "abs.py": "build_abs_op",
        "add.py": "build_add_op",
        "argmax.py": "build_argmax_op",
        "argmin.py": "build_argmin_op",
        "sqrt.py": "build_sqrt_op",
    }

    for filename, wrapper_name in expected_wrappers.items():
        text = (ops_root / filename).read_text()
        assert f"def {wrapper_name}(" in text
        assert f"from helia_core_tester.generation.utils.litert_builder import {wrapper_name}" not in text

    litert_builder_text = (_repo_root() / "helia_core_tester" / "generation" / "utils" / "litert_builder.py").read_text()
    assert "def build_abs_op(" not in litert_builder_text
    assert "def build_add_op(" not in litert_builder_text
    assert "def build_arg_op(" not in litert_builder_text
    assert "def build_sqrt_op(" not in litert_builder_text


def test_root_readme_documents_add_op_workflow() -> None:
    readme_candidates = [
        _repo_root().parents[1] / "README.md",
        _repo_root() / "README.md",
    ]

    for readme in readme_candidates:
        content = readme.read_text()
        if "scaffold_operator.py" in content:
            assert "build_<op>_op()" in content or "tensor_dtypes" in content
            return

    raise AssertionError("No README documents the Helia-Core Tester op workflow")
