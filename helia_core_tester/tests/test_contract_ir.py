"""helia_core_tester.contract.ir: loading the ns-cmsis-nn kernel contract export."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from helia_core_tester import __version__
from helia_core_tester.contract.ir import (
    CONTRACT_RELPATH,
    STATUS_ABSENT,
    STATUS_PRESENT,
    ContractError,
    ContractSchemaError,
    load_contract_set,
)

FIXTURE = Path(__file__).parent / "fixtures" / "contract" / "kernel_contracts.json"

# Declarations for every fixture symbol, in the header the fixture names for it.
HEADERS = {
    "Include/arm_nnfunctions.h": """\
/** doc */
arm_cmsis_nn_status arm_elementwise_add_s8(const int8_t *input_1_vect, const int8_t *input_2_vect,
                                           int8_t *output, const int32_t block_size);
arm_cmsis_nn_status arm_fx_pool_s8(const cmsis_nn_context *ctx, const cmsis_nn_dims *input_dims,
                                   const int8_t *input_data, int8_t *output_data);
int32_t arm_fx_pool_s8_get_buffer_size(const cmsis_nn_dims *input_dims);
int32_t arm_fx_pool_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims);
""",
    "Include/arm_nnfunctions_flt.h": """\
#if ARM_NN_ENABLE_F16
arm_cmsis_nn_status arm_fx_pool_f16(const int32_t dims[4], arm_nn_tensor_layout layout,
                                    void (*hook)(int32_t), float16_t *output);
#endif
""",
    "Include/arm_nnsupportfunctions.h": "void arm_nn_fx_helper(void);\n",
}


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "ns-cmsis-nn"
    for relative, text in HEADERS.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    target = root / CONTRACT_RELPATH
    target.parent.mkdir(parents=True)
    shutil.copy(FIXTURE, target)
    return root


def _rewrite(root: Path, mutate) -> None:
    path = root / CONTRACT_RELPATH
    document = json.loads(path.read_text())
    mutate(document)
    path.write_text(json.dumps(document))


def test_fixture_loads_with_every_field(checkout: Path) -> None:
    contracts = load_contract_set(checkout)
    assert contracts.status == STATUS_PRESENT and contracts.present
    assert contracts.path == checkout / CONTRACT_RELPATH
    add = contracts.require("arm_elementwise_add_s8")
    assert add.header == "Include/arm_nnfunctions.h" and add.line == 12 and add.guards == ()
    assert add.returns == "arm_cmsis_nn_status" and add.kind == "kernel"
    assert [p.name for p in add.params] == ["input_1_vect", "input_2_vect", "output", "block_size"]
    assert add.param("output").direction == "in,out" and add.param("output").is_pointer
    assert not add.param("block_size").is_pointer
    f16 = contracts.require("arm_fx_pool_f16")
    assert f16.guards == ("ARM_NN_ENABLE_F16",)
    assert f16.param("dims").extent == "[4]" and f16.param("dims").is_pointer
    assert f16.param("hook").is_function_pointer
    assert contracts.require("arm_fx_pool_s8_get_buffer_size").kind == "sizer"
    assert contracts.require("arm_fx_pool_s8_get_buffer_size_mve").kind == "sizer"
    assert contracts.require("arm_nn_fx_helper").kind == "support"
    assert contracts.require("arm_nn_fx_helper").params == ()
    assert [d.name for d in contracts.kernels()] == ["arm_elementwise_add_s8", "arm_fx_pool_s8", "arm_fx_pool_f16"]
    with pytest.raises(KeyError):
        add.param("nope")


def test_sizer_resolution_follows_cpu_profile(checkout: Path) -> None:
    contracts = load_contract_set(checkout)
    assert contracts.sizer_for("arm_fx_pool_s8", "cortex-m55") == "arm_fx_pool_s8_get_buffer_size_mve"
    # No _dsp variant declared: cortex-m4 falls back to the plain query.
    assert contracts.sizer_for("arm_fx_pool_s8", "cortex-m4") == "arm_fx_pool_s8_get_buffer_size"
    assert contracts.sizer_for("arm_fx_pool_s8", "cortex-m0") == "arm_fx_pool_s8_get_buffer_size"
    assert contracts.sizer_for("arm_elementwise_add_s8", "cortex-m55") is None


def test_no_root_and_no_file_are_absent_not_errors(tmp_path: Path, checkout: Path) -> None:
    assert load_contract_set(None).status == STATUS_ABSENT
    (checkout / CONTRACT_RELPATH).unlink()
    contracts = load_contract_set(checkout)
    assert contracts.status == STATUS_ABSENT and not contracts.present
    assert contracts.find("arm_elementwise_add_s8") is None
    with pytest.raises(ContractError, match="no kernel contract"):
        contracts.require("arm_elementwise_add_s8")


def test_unsupported_schema_names_the_tester_version(checkout: Path) -> None:
    _rewrite(checkout, lambda d: d.update(schema="ns-cmsis-nn/kernel-contracts/2"))
    with pytest.raises(ContractSchemaError, match=rf"kernel-contracts/2.*{__version__}"):
        load_contract_set(checkout)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["functions"][0]["params"][0].update(type="mystery_t *"), "outside the known vocabulary"),
        (lambda d: d["functions"][0]["params"][0].update(direction="inout"), "direction 'inout'"),
        (lambda d: d["functions"][0]["params"][0].update(extent="4"), "not an array extent"),
        (lambda d: d["functions"][0]["params"][0].update(name="1st"), "not an identifier"),
        (lambda d: d["functions"][0]["params"][0].update(bogus=1), "unknown keys"),
        (lambda d: d["functions"][0]["params"].append(dict(d["functions"][0]["params"][0])), "duplicate parameter"),
        (lambda d: d["functions"][0].pop("returns"), "lacks \\['returns'\\]"),
        (lambda d: d["functions"][0].update(returns="what"), "return type 'what'"),
        (lambda d: d["functions"][0].update(header="Include/arm_nn_types.h"), "not a public functions header"),
        (lambda d: d["functions"][0].update(line=0), "not a positive integer"),
        (lambda d: d["functions"][0].update(line=True), "not a positive integer"),
        (lambda d: d["functions"][0].update(guards="ARM_NN_ENABLE_F16"), "not a list of conditions"),
        (lambda d: d["functions"].append(dict(d["functions"][1])), "appears twice"),
        (lambda d: d.update(functions=[]), "has no functions"),
        (lambda d: d.update(functions="x"), "has no functions"),
    ],
)
def test_malformed_records_fail_closed(checkout: Path, mutate, message: str) -> None:
    _rewrite(checkout, mutate)
    with pytest.raises(ContractError, match=message):
        load_contract_set(checkout)


def test_top_level_shape_and_corrupt_json(checkout: Path) -> None:
    path = checkout / CONTRACT_RELPATH
    path.write_text("[]")
    with pytest.raises(ContractError, match="top level is not an object"):
        load_contract_set(checkout)
    path.write_text(path.read_text()[:1] + "{")
    with pytest.raises(ContractError, match="not valid JSON"):
        load_contract_set(checkout)


def test_symbol_the_header_does_not_declare_is_a_contradiction(checkout: Path) -> None:
    header = checkout / "Include" / "arm_nnfunctions.h"
    header.write_text(header.read_text().replace("arm_fx_pool_s8(", "arm_fx_pool_renamed_s8("))
    with pytest.raises(ContractError, match=r"1 symbol\(s\) not declared in Include/arm_nnfunctions.h: arm_fx_pool_s8"):
        load_contract_set(checkout)


def test_doc_mention_is_not_a_declaration(checkout: Path) -> None:
    header = checkout / "Include" / "arm_nnfunctions.h"
    text = header.read_text().replace(
        "int32_t arm_fx_pool_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims);\n",
        "/* see arm_fx_pool_s8_get_buffer_size_mve(dims) */\n")
    header.write_text(text)
    with pytest.raises(ContractError, match="arm_fx_pool_s8_get_buffer_size_mve"):
        load_contract_set(checkout)


def test_named_header_missing_from_checkout(checkout: Path) -> None:
    (checkout / "Include" / "arm_nnsupportfunctions.h").unlink()
    with pytest.raises(ContractError, match="names Include/arm_nnsupportfunctions.h, which the checkout"):
        load_contract_set(checkout)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_export_is_an_error_not_absent(checkout: Path) -> None:
    path = checkout / CONTRACT_RELPATH
    path.chmod(0)
    try:
        with pytest.raises(ContractError, match="cannot be read"):
            load_contract_set(checkout)
    finally:
        path.chmod(0o644)


def test_fixture_is_canonical_against_the_exporter_shape() -> None:
    """The fixture mirrors the ns-cmsis-nn exporter's record shape so the two repos
    cannot drift silently: fixed key order at the top, extent only when non-empty."""
    document = json.loads(FIXTURE.read_text())
    assert list(document) == ["schema", "functions"]
    for record in document["functions"]:
        assert list(record) == ["name", "header", "line", "guards", "returns", "params"]
        for param in record["params"]:
            assert "extent" not in param or param["extent"]
