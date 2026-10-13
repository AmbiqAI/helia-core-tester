"""Issue #100: descriptors/schema.json is enforced by the descriptor loader.

Covers the whole shipped tree, the enums that must track their code sources of
truth, each class of rejected document, variations, and a schema that cannot be
loaded (which must fail closed rather than skip validation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from helia_core_tester.core.cpu_targets import known_capabilities
from helia_core_tester.generation.io import descriptors as desc_io
from helia_core_tester.generation.io.dtypes import (
    ALLOWED_TENSOR_DTYPES,
    FLOAT_DTYPES,
    LEGACY_ACTIVATION_DTYPES,
    LEGACY_WEIGHT_DTYPES,
)
from helia_core_tester.generation.ops.catalog import iter_operator_specs


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema() -> dict:
    return json.loads(desc_io.SCHEMA_PATH.read_text())


def _clamp(**overrides) -> dict:
    doc = {
        "operator": "Clamp",
        "name": "clamp_case_s8",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "hint": {"call_style": "per_tensor"},
        "input_shape": [1, 4, 4, 4],
        "act_min": -64,
        "act_max": 63,
    }
    doc.update(overrides)
    return {k: v for k, v in doc.items() if v is not None}


def _write(tmp_path: Path, docs: list, stem: str = "case") -> Path:
    path = tmp_path / f"{stem}.yaml"
    path.write_text(yaml.safe_dump_all(docs, sort_keys=False))
    return path


def _load(tmp_path: Path, doc, stem: str = "case") -> list:
    return desc_io.load_descriptor(str(_write(tmp_path, [doc], stem)))


def _rejected(tmp_path: Path, doc, stem: str = "case") -> str:
    with pytest.raises(ValueError) as excinfo:
        _load(tmp_path, doc, stem)
    message = str(excinfo.value)
    assert "violates schema.json" in message or "must be a mapping" in message
    assert f"{stem}.yaml" in message
    return message


# --- the shipped tree -------------------------------------------------------


def test_schema_is_a_valid_draft_2020_12_schema() -> None:
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(_schema())


def test_every_shipped_descriptor_passes_the_schema() -> None:
    descriptors = desc_io.load_all_descriptors(str(_repo_root() / "assets" / "descriptors"))
    assert len(descriptors) > 2000


# --- enums pinned to their sources of truth ----------------------------------


def test_operator_enum_matches_the_catalog() -> None:
    assert set(_schema()["properties"]["operator"]["enum"]) == {
        spec.operator for spec in iter_operator_specs()
    }


def test_dtype_enums_match_the_dtype_module() -> None:
    schema = _schema()
    assert set(schema["$defs"]["tensorDType"]["enum"]) == set(ALLOWED_TENSOR_DTYPES)
    assert set(schema["properties"]["activation_dtype"]["enum"]) == set(LEGACY_ACTIVATION_DTYPES) | FLOAT_DTYPES
    assert set(schema["properties"]["weight_dtype"]["enum"]) == set(LEGACY_WEIGHT_DTYPES) | FLOAT_DTYPES


def test_required_capabilities_enum_matches_the_cpu_profiles() -> None:
    items = _schema()["properties"]["required_capabilities"]["items"]
    assert set(items["enum"]) == set(known_capabilities())


# --- accepted boundaries ----------------------------------------------------


def test_minimal_valid_document_loads(tmp_path: Path) -> None:
    (loaded,) = _load(tmp_path, _clamp())
    assert loaded["name"] == "clamp_case_s8"


@pytest.mark.parametrize(
    ("doc", "stem"),
    [
        (_clamp(act_min=-64.0), "case"),  # integral float is still an integer code
        (_clamp(act_min=-64.5, suite="float"), "case"),
        (_clamp(act_min=-64.5), "clamp_float"),  # suite defaults to float from the stem
    ],
    ids=["integral-float", "explicit-float-suite", "float-stem"],
)
def test_activation_bound_boundaries_accepted(tmp_path: Path, doc, stem) -> None:
    _load(tmp_path, doc, stem)


@pytest.mark.parametrize("caps", [["mve"], ["soft_float"], ["dsp", "fp16_execution"]])
def test_known_required_capabilities_accepted(tmp_path: Path, caps) -> None:
    _load(tmp_path, _clamp(required_capabilities=caps))


# --- rejected documents -------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "path_fragment"),
    [
        ({"act_min": -64.5}, "$.act_min"),
        ({"act_max": 63.25, "suite": "default"}, "$.act_max"),
        ({"suite": "integer"}, "$.suite"),
        ({"required_capabilities": ["soft-float"]}, "$.required_capabilities[0]"),
        ({"required_capabilities": []}, "$.required_capabilities"),
        ({"required_capabilities": ["mve", "mve"]}, "$.required_capabilities"),
        ({"required_capabilities": "mve"}, "$.required_capabilities"),
        ({"input_mode": "sweep"}, "$.input_mode"),
        ({"nonfinite_tokens": ["NaN"]}, "$.nonfinite_tokens[0]"),
        ({"nonfinite_policy": "lenient"}, "$.nonfinite_policy"),
        ({"activation_dtype": "INT8"}, "$.activation_dtype"),
        ({"name": "Clamp-Case"}, "$.name"),
        ({"name": None}, "$"),
        ({"operator": "Clampp"}, "$.operator"),
    ],
)
def test_schema_violations_fail_loading(tmp_path: Path, overrides, path_fragment) -> None:
    message = _rejected(tmp_path, _clamp(**overrides))
    assert f"{path_fragment}:" in message


def test_nonfinite_sweep_without_a_policy_is_rejected(tmp_path: Path) -> None:
    doc = _clamp(suite="float", input_mode="nonfinite_sweep", nonfinite_tokens=["nan"])
    message = _rejected(tmp_path, doc)
    assert "'nonfinite_policy' is a required property" in message


def test_every_violation_is_reported_with_the_descriptor_name(tmp_path: Path) -> None:
    message = _rejected(tmp_path, _clamp(act_min=-64.5, input_mode="sweep"))
    assert "descriptor 'clamp_case_s8'" in message
    assert "$.act_min:" in message
    assert "$.input_mode:" in message


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    _rejected(tmp_path, ["operator", "Clamp"])


@pytest.mark.parametrize(
    ("block_size", "ok"),
    [(2, True), (1, True), (0, False), ([2], False), ("2", False)],
)
def test_block_size_is_a_positive_integer(tmp_path: Path, block_size, ok) -> None:
    doc = {
        "operator": "DepthToSpace",
        "name": "depth_to_space_case_s8",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "input_shape": [1, 2, 2, 8],
        "block_size": block_size,
    }
    if ok:
        _load(tmp_path, doc)
    else:
        assert "$.block_size:" in _rejected(tmp_path, doc)


@pytest.mark.parametrize(
    ("crops", "ok"),
    [([[0, 0], [1, 1]], True), ([[]], False), ([0, 0], False), ([[0, 0, 0]], False)],
)
def test_crops_are_begin_end_pairs(tmp_path: Path, crops, ok) -> None:
    doc = {
        "operator": "BatchToSpaceND",
        "name": "batch_to_space_nd_case_s8",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "input_shape": [4, 2, 2, 1],
        "block_shape": [2, 2],
        "crops": crops,
    }
    if ok:
        _load(tmp_path, doc)
    else:
        assert "$.crops[0]" in _rejected(tmp_path, doc)


def _prelu_scalar(**overrides) -> dict:
    doc = {
        "operator": "PReLUScalar",
        "name": "prelu_scalar_case_s8",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "input_shape": [1, 1, 1, 2],
        "alpha_shape": [1, 1, 1, 2],
    }
    doc.update(overrides)
    return doc


@pytest.mark.parametrize(
    "overrides",
    [
        {"scalar_input_value": 0.5},
        {"hint": {"extras": {"input_values": [0.8, -0.8]}}},
    ],
    ids=["scalar", "per-pixel"],
)
def test_prelu_scalar_accepts_either_input_source(tmp_path: Path, overrides) -> None:
    _load(tmp_path, _prelu_scalar(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [{}, {"hint": {"extras": {"alpha_values": [0.1]}}}, {"hint": {}}],
    ids=["neither", "extras-without-inputs", "empty-hint"],
)
def test_prelu_scalar_without_an_input_source_is_rejected(tmp_path: Path, overrides) -> None:
    _rejected(tmp_path, _prelu_scalar(**overrides))


# --- variations -------------------------------------------------------------


def test_invalid_field_in_a_variation_is_rejected() -> None:
    base = _clamp(variations=[{"name": "clamp_case_ok_s8"}, {"name": "clamp_case_bad_s8", "act_min": 0.5}])
    (normalized,) = [desc_io._validate_and_normalize_descriptor(base)]
    normalized["_source_stem"] = "clamp"
    with pytest.raises(ValueError, match=r"clamp_case_bad_s8.*\$\.act_min"):
        desc_io.expand_descriptor_variations(normalized)


def test_float_stem_carries_into_variations() -> None:
    base = _clamp(variations=[{"name": "clamp_case_half_s8", "act_min": 0.5}])
    normalized = desc_io._validate_and_normalize_descriptor(base)
    normalized["_source_stem"] = "clamp_float"
    (expanded,) = desc_io.expand_descriptor_variations(normalized)
    assert expanded["act_min"] == 0.5


# --- schema load faults fail closed -----------------------------------------


@pytest.mark.parametrize(
    "content",
    [None, "{ not json", json.dumps({"type": "not-a-type"})],
    ids=["missing", "corrupt-json", "invalid-schema"],
)
def test_unusable_schema_fails_closed(tmp_path: Path, monkeypatch, content) -> None:
    schema_path = tmp_path / "schema.json"
    if content is not None:
        schema_path.write_text(content)
    monkeypatch.setattr(desc_io, "SCHEMA_PATH", schema_path)
    with pytest.raises(ValueError, match="Descriptor schema .* is unusable"):
        _load(tmp_path, _clamp())
