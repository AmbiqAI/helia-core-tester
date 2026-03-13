from __future__ import annotations

from pathlib import Path
import re

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.catalog import get_operator_spec, iter_operator_specs
from helia_core_tester.scripts.scaffold_operator import create_operator_skeleton, operator_to_module_basename
from helia_core_tester.scripts.validate_operator_catalog import validate_operator_contracts


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    created = create_operator_skeleton(
        tmp_path,
        operator="ExampleOp",
        family="BasicMathFunctions",
        parity_kind="cmsis",
    )

    assert created["module"] == tmp_path / "helia_core_tester" / "generation" / "ops" / "BasicMathFunctions" / "example_op.py"
    assert created["descriptor"] == tmp_path / "assets" / "descriptors" / "BasicMathFunctions" / "example_op.yaml"
    assert created["header_template"] == tmp_path / "assets" / "templates" / "BasicMathFunctions" / "example_op" / "example_op.h.j2"
    assert created["source_template"] == tmp_path / "assets" / "templates" / "BasicMathFunctions" / "example_op" / "example_op.c.j2"
    assert created["module"].exists()
    assert created["descriptor"].exists()
    assert created["header_template"].exists()
    assert created["source_template"].exists()


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
