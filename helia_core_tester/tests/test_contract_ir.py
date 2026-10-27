"""helia_core_tester.contract.ir: loading the ns-cmsis-nn kernel contract export."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from helia_core_tester import __version__
from helia_core_tester.contract.ir import (
    CONTRACT_RELPATH,
    MANIFEST_RELPATH,
    PACKAGE_RELPATH,
    STATUS_ABSENT,
    STATUS_PRESENT,
    ContractError,
    ContractSchemaError,
    clear_contract_cache,
    load_contract_set,
    load_contracts_package,
)
from helia_core_tester.tests.contract_checkout import implementation, install_reader, write_manifest

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
    return install_reader(root)


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
        (lambda d: d["functions"][0].pop("returns"), "are not \\['guards', 'header'"),
        (lambda d: d["functions"][0].update(extra=1), "are not \\['guards', 'header'"),
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
    with pytest.raises(ContractError, match="top level must be an object"):
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
    with pytest.raises(ContractError, match="names Include/arm_nnsupportfunctions.h, which cannot be read"):
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


def test_a_present_contract_is_loaded_once_per_checkout(checkout: Path) -> None:
    clear_contract_cache()
    first = load_contract_set(checkout)
    assert first.status == STATUS_PRESENT
    assert load_contract_set(checkout) is first
    assert load_contract_set(Path(str(checkout))) is first


def test_the_cache_reloads_when_the_export_changes(checkout: Path) -> None:
    clear_contract_cache()
    first = load_contract_set(checkout)
    _rewrite(checkout, lambda doc: doc["functions"].pop())
    path = checkout / CONTRACT_RELPATH
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    second = load_contract_set(checkout)
    assert second is not first and len(second.functions) == len(first.functions) - 1


def test_the_cache_reloads_when_a_public_header_changes(checkout: Path) -> None:
    clear_contract_cache()
    first = load_contract_set(checkout)
    header = checkout / "Include" / "arm_nnfunctions.h"
    header.write_text(header.read_text().replace("arm_fx_pool_s8_get_buffer_size_mve(", "arm_fx_pool_s8_gone_mve("))
    os.utime(header, ns=(header.stat().st_atime_ns, header.stat().st_mtime_ns + 1_000_000))
    with pytest.raises(ContractError, match="not declared in Include/arm_nnfunctions.h"):
        load_contract_set(checkout)


def test_a_failed_load_is_not_cached(checkout: Path) -> None:
    clear_contract_cache()
    _rewrite(checkout, lambda doc: doc.update(schema="ns-cmsis-nn/kernel-contracts/0"))
    with pytest.raises(ContractSchemaError):
        load_contract_set(checkout)
    _rewrite(checkout, lambda doc: doc.update(schema="ns-cmsis-nn/kernel-contracts/1"))
    path = checkout / CONTRACT_RELPATH
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 2_000_000))
    assert load_contract_set(checkout).status == STATUS_PRESENT


def test_absent_and_other_checkouts_do_not_share_the_cache(checkout: Path, tmp_path: Path) -> None:
    clear_contract_cache()
    present = load_contract_set(checkout)
    assert load_contract_set(tmp_path / "empty").status == STATUS_ABSENT
    assert load_contract_set(None).status == STATUS_ABSENT
    other = tmp_path / "other"
    shutil.copytree(checkout, other)
    assert load_contract_set(other) is not present
    assert load_contract_set(other).functions.keys() == present.functions.keys()


def test_the_reader_is_the_checkouts_own(checkout: Path) -> None:
    package = load_contracts_package(checkout)
    assert Path(package.__file__).parent == checkout / PACKAGE_RELPATH
    assert load_contracts_package(checkout) is package


def test_an_export_without_a_reader_fails_closed(checkout: Path) -> None:
    shutil.rmtree(checkout / "python")
    with pytest.raises(ContractError, match="no python/ns_cmsis_nn_contracts to read it.*#889"):
        load_contract_set(checkout)


def test_a_reader_without_an_export_is_absent(checkout: Path) -> None:
    (checkout / CONTRACT_RELPATH).unlink()
    assert load_contract_set(checkout).status == STATUS_ABSENT


def test_an_older_python_dir_without_the_reader_is_absent(tmp_path: Path) -> None:
    root = tmp_path / "v7.35.1"
    (root / "python" / "cmsis_nn").mkdir(parents=True)
    (root / "python" / "cmsis_nn" / "__init__.py").write_text("")
    (root / "Include").mkdir()
    assert load_contracts_package(root) is None
    assert load_contracts_package(None) is None
    assert load_contract_set(root).status == STATUS_ABSENT


def test_a_reader_that_cannot_be_imported_fails_closed(checkout: Path) -> None:
    (checkout / PACKAGE_RELPATH / "__init__.py").write_text("def broken(:\n")
    before = set(sys.modules)
    with pytest.raises(ContractError, match="cannot be imported \\(SyntaxError"):
        load_contract_set(checkout)
    assert not [name for name in set(sys.modules) - before if name.startswith("_hct_ns_cmsis_nn_contracts_")]


def test_a_reader_without_the_api_fails_closed(checkout: Path) -> None:
    (checkout / PACKAGE_RELPATH / "__init__.py").write_text("from .errors import ContractError\n")
    with pytest.raises(ContractError, match="does not export \\['ContractSchemaError', 'load_signatures'"):
        load_contract_set(checkout)


def test_a_reader_of_a_schema_the_tester_does_not_read(checkout: Path) -> None:
    signatures = checkout / PACKAGE_RELPATH / "signatures.py"
    signatures.write_text(signatures.read_text().replace('kernel-contracts/1"', 'kernel-contracts/2"'))
    _rewrite(checkout, lambda d: d.update(schema="ns-cmsis-nn/kernel-contracts/2"))
    with pytest.raises(ContractSchemaError, match=rf"reads schema 'ns-cmsis-nn/kernel-contracts/2'.*{__version__}"):
        load_contract_set(checkout)


def test_each_checkout_reads_with_its_own_reader(checkout: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    shutil.copytree(checkout, other)
    marker = other / PACKAGE_RELPATH / "__init__.py"
    marker.write_text(marker.read_text() + "\nHCT_MARKER = 'other'\n")
    mine, theirs = load_contracts_package(checkout), load_contracts_package(other)
    assert mine is not theirs and theirs.HCT_MARKER == "other" and not hasattr(mine, "HCT_MARKER")


def test_an_edited_reader_is_reimported(checkout: Path) -> None:
    clear_contract_cache()
    first = load_contract_set(checkout)
    init = checkout / PACKAGE_RELPATH / "__init__.py"
    init.write_text(init.read_text() + "\nHCT_MARKER = 1\n")
    os.utime(init, ns=(init.stat().st_atime_ns, init.stat().st_mtime_ns + 1_000_000))
    assert load_contract_set(checkout) is not first
    assert load_contracts_package(checkout).HCT_MARKER == 1


def test_an_unexpected_reader_failure_is_a_contract_error(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_contract_cache()
    package = load_contracts_package(checkout)

    def explode(root):
        raise RuntimeError("injected")

    monkeypatch.setattr(package, "load_signatures", explode)
    with pytest.raises(ContractError, match="failed reading .*RuntimeError: injected"):
        load_contract_set(checkout)


def test_a_checkout_without_a_manifest_has_no_records(checkout: Path) -> None:
    contracts = load_contract_set(checkout)
    assert contracts.implementations == {} and contracts.library_version is None and contracts.manifest_path is None
    assert contracts.implementations_of("arm_fx_pool_s8") == []


def test_manifest_records_are_read_and_route_records_are_not_direct_calls(checkout: Path) -> None:
    write_manifest(checkout, [
        implementation("fx_pool_s8", "arm_fx_pool_s8", op="avg_pool", requires=["mvei", "!autovectorize"],
                       since="7.40.0", declines=["layer", "build"]),
        implementation("fx_pool_via_itself", "arm_fx_pool_s8", op="avg_pool", wrapper="arm_fx_pool_s8",
                       route="fx_pool_s8:self"),
    ], library_version="7.41.0")
    contracts = load_contract_set(checkout)
    assert contracts.library_version == "7.41.0" and contracts.manifest_path == checkout / MANIFEST_RELPATH
    assert set(contracts.implementations) == {"fx_pool_s8", "fx_pool_via_itself"}
    direct = contracts.implementations_of("arm_fx_pool_s8")
    assert [impl.id for impl in direct] == ["fx_pool_s8"]
    assert direct[0].requires == frozenset({"mvei", "!autovectorize"}) and direct[0].since == "7.40.0"
    assert direct[0].declines_on_build and not contracts.implementations["fx_pool_via_itself"].declines_on_build
    assert contracts.implementations_of("arm_elementwise_add_s8") == []


@pytest.mark.parametrize(
    "mutate, error, message",
    [
        (lambda d: d.update(signature_sha256="0" * 64), ContractError, "the manifest is stale"),
        (lambda d: d.update(schema="ns-cmsis-nn/kernel-implementations/9"), ContractSchemaError,
         "kernel-implementations/9"),
        (lambda d: d["implementations"][0].update(entry="arm_not_exported"), ContractError,
         "'arm_not_exported' is not a public function"),
        (lambda d: d["implementations"][0].update(requires=["neon"]), ContractError, "unknown feature token"),
        (lambda d: d.update(implementations=[]), ContractError, "has no implementations"),
        (lambda d: d["implementations"][0].update(declines="build"), ContractError, "declines 'build' is not a list"),
        (lambda d: d["implementations"][0].update(declines=[{"cause": "weather", "when": "x"}]), ContractError,
         "with a cause in \\['buffer', 'build', 'layer'\\]"),
        (lambda d: d["implementations"][0].update(declines=["build"]), ContractError, "is not a list of"),
    ],
)
def test_a_bad_manifest_fails_closed(checkout: Path, mutate, error, message: str) -> None:
    path = write_manifest(checkout, [implementation("fx_pool_s8", "arm_fx_pool_s8", op="avg_pool")])
    document = json.loads(path.read_text())
    mutate(document)
    path.write_text(json.dumps(document))
    with pytest.raises(error, match=message):
        load_contract_set(checkout)


def test_the_cache_reloads_when_the_manifest_appears_or_changes(checkout: Path) -> None:
    clear_contract_cache()
    assert load_contract_set(checkout).implementations == {}
    path = write_manifest(checkout, [implementation("fx_pool_s8", "arm_fx_pool_s8", op="avg_pool")])
    assert set(load_contract_set(checkout).implementations) == {"fx_pool_s8"}
    write_manifest(checkout, [implementation("fx_pool_s8_b", "arm_fx_pool_s8", op="avg_pool")])
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    assert set(load_contract_set(checkout).implementations) == {"fx_pool_s8_b"}


def test_a_reader_without_decline_causes_fails_closed(checkout: Path) -> None:
    vocabulary = checkout / PACKAGE_RELPATH / "vocabulary.py"
    vocabulary.write_text(vocabulary.read_text().replace("DECLINE_CAUSES = ", "_DECLINE_CAUSES = "))
    with pytest.raises(ContractError, match="DECLINE_CAUSES None has no 'build' cause"):
        load_contract_set(checkout)


def test_a_reader_returning_an_unexpected_record_is_a_contract_error(checkout: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    clear_contract_cache()
    write_manifest(checkout, [implementation("fx_pool_s8", "arm_fx_pool_s8", op="avg_pool")])
    package = load_contracts_package(checkout)
    real = package.load_implementations

    def without_records(root, signatures):
        loaded = real(root, signatures)
        return type(loaded)(path=loaded.path, library_version=loaded.library_version, planner_abi=None,
                            implementations={"fx_pool_s8": object()})

    monkeypatch.setattr(package, "load_implementations", without_records)
    with pytest.raises(ContractError, match="failed reading .*AttributeError"):
        load_contract_set(checkout)
