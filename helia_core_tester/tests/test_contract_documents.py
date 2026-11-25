"""contract/documents.py and `hct contract validate`: hct's own contract schemas (plan E2)."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import jsonschema
import pytest
from typer.testing import CliRunner

from helia_core_tester import cli as cli_module
from helia_core_tester.contract import cli as contract_cli
from helia_core_tester.contract.documents import _SCHEMA_DIR, ContractError, ContractSchemaError
from helia_core_tester.contract.ir import clear_contract_cache, load_contract_set
from helia_core_tester.core.library import LibraryProfileError, load_profile, require_supported
from helia_core_tester.tests.contract_checkout import function, implementation, write_checkout

DSP_HEADER = "Include/dsp/basic_math_functions.h"
ADD = {"name": "arm_add_f32", "header": DSP_HEADER, "line": 40, "guards": [], "returns": "void",
       "params": [{"name": "pSrcA", "type": "const float32_t *", "direction": "in"},
                  {"name": "pSrcB", "type": "const float32_t *", "direction": "in"},
                  {"name": "pDst", "type": "float32_t *", "direction": "out"},
                  {"name": "blockSize", "type": "uint32_t", "direction": "in"}]}
INIT = {"name": "arm_fir_init_f32", "header": "Include/dsp/filtering_functions.h", "line": 9, "guards": [],
        "returns": "arm_status",
        "params": [{"name": "S", "type": "arm_fir_instance_f32 *", "direction": "out"},
                   {"name": "pState", "type": "float32_t *", "direction": "in,out"}]}


def _record(impl_id: str = "add_f32", entry: str = "arm_add_f32", **changes) -> dict:
    record = implementation(impl_id, entry, op="add")
    record.update(family="basic_math", quantization="none",
                  numerics={"reference": "host_ref:arm_add_f32", "claim": "bounded", "holds_with": ["finite inputs"]})
    record.update(changes)
    return record


def _dsp(root: Path, functions=(ADD, INIT), records: Optional[list] = None, *, library: str = "cmsis-dsp",
         manifest_library: Optional[str] = None) -> Path:
    """A CMSIS-DSP-shaped checkout with hct/* documents under Testing/hct and a profile naming them."""
    for record in functions:
        header = root / record["header"]
        header.parent.mkdir(parents=True, exist_ok=True)
        params = ", ".join(f"{p['type']} {p['name']}" for p in record["params"])
        with header.open("a") as handle:
            handle.write(f"{record['returns']} {record['name']}({params});\n")
    pack = root / "Testing" / "hct"
    pack.mkdir(parents=True, exist_ok=True)
    export = json.dumps({"schema": "hct/kernel-contracts/1", "library": library, "functions": list(functions)})
    (pack / "kernel_contracts.json").write_text(export)
    profile = f"schema: hct/library-profile/1\nlibrary: {library}\nsource_root: ../..\ncontracts:\n" \
              "  signatures: kernel_contracts.json\n"
    if records is not None:
        manifest = {"schema": "hct/kernel-implementations/1", "library": manifest_library or library,
                    "library_version": "1.16.2", "signature_schema": "hct/kernel-contracts/1",
                    "signature_sha256": hashlib.sha256(export.encode()).hexdigest(), "planner_abi": None,
                    "implementations": records}
        (pack / "kernel_implementations.json").write_text(json.dumps(manifest))
        profile += "  implementations: kernel_implementations.json\n"
    (pack / "hct-library.yaml").write_text(profile)
    clear_contract_cache()
    return root


def _load(root: Path):
    profile = load_profile(root / "Testing" / "hct" / "hct-library.yaml")
    return load_contract_set(profile.source_root, signatures=profile.signatures,
                             implementations=profile.implementations)


def _edit(path: Path, mutate) -> None:
    document = json.loads(path.read_text())
    mutate(document)
    path.write_text(json.dumps(document))


def _resign(root: Path) -> None:
    """Rebind the manifest to the export after the export was edited."""
    pack = root / "Testing" / "hct"
    sha = hashlib.sha256((pack / "kernel_contracts.json").read_bytes()).hexdigest()
    _edit(pack / "kernel_implementations.json", lambda d: d.update(signature_sha256=sha))


def test_the_schemas_are_valid_json_schema() -> None:
    for name in ("kernel-contracts-1.json", "kernel-implementations-1.json", "library-profile-1.json"):
        jsonschema.Draft202012Validator.check_schema(json.loads((_SCHEMA_DIR / name).read_text()))


def test_a_generic_contract_with_library_vocabulary_loads(tmp_path: Path) -> None:
    contracts = _load(_dsp(tmp_path, records=[_record()]))
    assert contracts.present and contracts.schema == "hct/kernel-contracts/1" and contracts.library == "cmsis-dsp"
    assert set(contracts.functions) == {"arm_add_f32", "arm_fir_init_f32"}
    assert contracts.functions["arm_fir_init_f32"].param("S").c_type == "arm_fir_instance_f32 *"
    assert contracts.library_version == "1.16.2"
    numerics = contracts.implementations["add_f32"].numerics
    assert (numerics.reference, numerics.claim, numerics.holds_with) == \
        ("host_ref:arm_add_f32", "bounded", ("finite inputs",))


def test_reading_a_generic_contract_imports_no_library_python(tmp_path: Path) -> None:
    root = _dsp(tmp_path, records=[_record()])
    before = set(sys.modules)
    _load(root)
    assert not [name for name in set(sys.modules) - before if not name.startswith(("helia_core_tester", "jsonschema",
                                                                                     "referencing", "rpds", "attr"))]


def test_legacy_vocabulary_is_refused_under_the_legacy_id(tmp_path: Path) -> None:
    nn_header = "Include/arm_nnfunctions.h"
    assert load_contract_set(write_checkout(tmp_path / "nn", [dict(ADD, header=nn_header)])).present
    with pytest.raises(ContractError, match="return type 'arm_status' is outside the known vocabulary"):
        load_contract_set(write_checkout(tmp_path / "nn2", [dict(INIT, header=nn_header)]))
    with pytest.raises(ContractError, match="header 'Include/dsp/basic_math_functions.h' is not a public functions"):
        load_contract_set(write_checkout(tmp_path / "nn3", [ADD]))


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.pop("library"), "'library' is a required property"),
    (lambda d: d.update(library="CMSIS DSP"), "library: 'CMSIS DSP' does not match"),
    (lambda d: d.update(schema="hct/kernel-contracts/2"), "schema 'hct/kernel-contracts/2' is not one"),
    (lambda d: d["functions"][0].update(header="../outside.h"), "header: '../outside.h' does not match"),
    (lambda d: d["functions"][0].update(header="/abs.h"), "header: '/abs.h' does not match"),
    (lambda d: d["functions"][0].update(header="Include\\\\dsp.h"), "does not match"),
    (lambda d: d["functions"][0].update(returns="unsigned long"), "return type 'unsigned long' is not a C type"),
    (lambda d: d["functions"][0]["params"][0].update(type="float32_t[4]"), "type 'float32_t\\[4\\]' is not a C type"),
    (lambda d: d["functions"][0]["params"][0].update(type=""), "should be non-empty"),
    (lambda d: d["functions"].append(copy.deepcopy(d["functions"][0])), "arm_add_f32 appears twice"),
])
def test_bad_generic_exports_are_refused(tmp_path: Path, mutate, message: str) -> None:
    root = _dsp(tmp_path)
    _edit(root / "Testing" / "hct" / "kernel_contracts.json", mutate)
    with pytest.raises(ContractError, match=message):
        _load(root)


def test_an_unknown_schema_is_a_schema_error(tmp_path: Path) -> None:
    root = _dsp(tmp_path)
    _edit(root / "Testing" / "hct" / "kernel_contracts.json", lambda d: d.update(schema="hct/kernel-contracts/2"))
    with pytest.raises(ContractSchemaError):
        _load(root)


def test_the_legacy_id_takes_no_library_field(tmp_path: Path) -> None:
    root = write_checkout(tmp_path / "nn", [function("arm_fx_s8")])
    _edit(root / "Tests" / "KernelContracts" / "kernel_contracts.json", lambda d: d.update(library="ns-cmsis-nn"))
    with pytest.raises(ContractError, match="ns-cmsis-nn/kernel-contracts/1 documents take no library field; .* is "
                                            "hct/kernel-contracts/1"):
        load_contract_set(root)


def test_a_header_missing_its_declaration_is_stale(tmp_path: Path) -> None:
    root = _dsp(tmp_path)
    header = root / DSP_HEADER
    header.write_text("/* arm_add_f32(a, b) is documented here only */\n")
    with pytest.raises(ContractError, match="1 symbol\\(s\\) not declared in Include/dsp/basic_math_functions.h"):
        _load(root)


@pytest.mark.parametrize("change, message", [
    ({"op": "Add"}, "op: 'Add' does not match"),
    ({"requires": ["MVE"]}, "requires/0: 'MVE' does not match"),
    ({"requires": ["helium", "helium"]}, "has non-unique elements"),
    ({"entry": "arm_sub_f32"}, "add_f32.entry: 'arm_sub_f32' is not a public function"),
    ({"since": "1.16"}, "since: '1.16' does not match"),
    ({"numerics": {"reference": "host_ref:x", "claim": "close", "holds_with": []}}, "claim: 'close' is not one of"),
    ({"numerics": {"reference": "nowhere", "claim": "bit_exact", "holds_with": []}}, "numerics/reference"),
    ({"numerics": {"reference": None, "claim": "bit_exact"}}, "'holds_with' is a required property"),
    ({"buffers": [{"name": "scratch", "param": "pScratch", "size_query": None, "units": "bytes", "align": 4,
                   "lifetime": "call", "optional": False, "format": None, "note": None}]},
     "arm_add_f32 has no parameter 'pScratch'"),
    ({"buffers": [{"name": "scratch", "param": None, "size_query": {"symbol": "arm_nope", "args": {}},
                   "units": "bytes", "align": 4, "lifetime": "call", "optional": False, "format": None,
                   "note": None}]}, "buffers\\[0\\].size_query: 'arm_nope' is not a public function"),
    ({"buffers": [{"name": "scratch", "param": None, "size_query": None, "units": "bytes", "align": 0,
                   "lifetime": "call", "optional": False, "format": None, "note": None}]},
     "align: 0 is less than the minimum of 1"),
    ({"buffers": [{"name": "s", "param": None, "size_query": None, "units": "bytes", "align": 1, "lifetime": "call",
                   "optional": False, "format": None, "note": None}] * 2}, "duplicate buffer names"),
    ({"prepare": [{"symbol": "arm_fir_init_f32", "args": {}, "consumes": [], "fills": "state", "note": None}]},
     "fills 'state', which is not one of its buffers"),
    ({"prepare": [{"symbol": "arm_fir_init_f32", "args": {"S": ""}, "consumes": [], "fills": None, "note": None}]},
     "prepare/0/args/S"),
    ({"eligibility": {"when": "aligned", "predicate": "arm_is_aligned", "predicate_static": False, "args": {}}},
     "add_f32.eligibility: 'arm_is_aligned' is not a public function"),
    ({"routed_by": "wrapper"}, "routed_by"),
])
def test_bad_generic_manifests_are_refused(tmp_path: Path, change: dict, message: str) -> None:
    root = _dsp(tmp_path, records=[_record(**change)])
    with pytest.raises(ContractError, match=message):
        _load(root)


def test_a_static_eligibility_predicate_need_not_be_exported(tmp_path: Path) -> None:
    eligibility = {"when": "aligned", "predicate": "arm_is_aligned", "predicate_static": True, "args": None}
    assert _load(_dsp(tmp_path, records=[_record(eligibility=eligibility)])).present


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(library="cmsis-nn"), "is for library 'cmsis-nn', but .* is for 'cmsis-dsp'"),
    (lambda d: d.update(signature_schema="ns-cmsis-nn/kernel-contracts/1"), "signature_schema"),
    (lambda d: d.update(signature_sha256="0" * 64), "the manifest is stale"),
    (lambda d: d.update(signature_sha256="ABC"), "signature_sha256: 'ABC' does not match"),
    (lambda d: d.update(planner_abi=0), "planner_abi"),
    (lambda d: d.update(planner_abi=True), "planner_abi"),
    (lambda d: d.update(library_version="v1"), "library_version: 'v1' does not match"),
    (lambda d: d.update(implementations=[]), "implementations: \\[\\] should be non-empty"),
    (lambda d: d["implementations"].append(copy.deepcopy(d["implementations"][0])),
     "implementation add_f32 appears twice"),
    (lambda d: d.update(extra=1), "'extra' was unexpected"),
])
def test_bad_generic_manifest_envelopes_are_refused(tmp_path: Path, mutate, message: str) -> None:
    root = _dsp(tmp_path, records=[_record()])
    _edit(root / "Testing" / "hct" / "kernel_implementations.json", mutate)
    with pytest.raises(ContractError, match=message):
        _load(root)


def test_two_records_on_one_route_are_refused(tmp_path: Path) -> None:
    route = {"wrapper": "arm_add_f32", "route": "add:fast", "requires": [], "when": "always"}
    root = _dsp(tmp_path, records=[_record("a", routed_by=route), _record("b", routed_by=route)])
    with pytest.raises(ContractError, match="two implementations claim the wrapper route arm_add_f32 add:fast"):
        _load(root)


def test_a_manifest_follows_its_export(tmp_path: Path) -> None:
    root = _dsp(tmp_path, records=[_record()])
    _edit(root / "Testing" / "hct" / "kernel_contracts.json", lambda d: d["functions"].pop())
    with pytest.raises(ContractError, match="the manifest is stale"):
        _load(root)
    _resign(root)
    assert set(_load(root).functions) == {"arm_add_f32"}


# The profile's contracts field.

def test_a_profile_names_its_contracts(tmp_path: Path) -> None:
    root = _dsp(tmp_path, records=[_record()])
    profile = load_profile(root / "Testing" / "hct" / "hct-library.yaml")
    assert profile.signatures == (root / "Testing" / "hct" / "kernel_contracts.json").resolve()
    assert profile.implementations == (root / "Testing" / "hct" / "kernel_implementations.json").resolve()


@pytest.mark.parametrize("contracts, message", [
    ("  signatures: ../../../elsewhere.json\n", "is outside source_root"),
    ("  signatures: /abs/kernel_contracts.json\n", "must be a relative POSIX path"),
    ("  signatures: a\\\\b.json\n", "must be a relative POSIX path"),
    ("  implementations: kernel_implementations.json\n", "'signatures' is a required property"),
    ("  signatures: kernel_contracts.json\n  extra: x\n", "'extra' was unexpected"),
    ("  signatures: ''\n", "should be non-empty"),
])
def test_bad_profile_contracts_are_refused(tmp_path: Path, contracts: str, message: str) -> None:
    pack = tmp_path / "lib" / "Testing" / "hct"
    pack.mkdir(parents=True)
    (pack / "hct-library.yaml").write_text(
        f"schema: hct/library-profile/1\nlibrary: cmsis-dsp\nsource_root: ../..\ncontracts:\n{contracts}")
    with pytest.raises(LibraryProfileError, match=message):
        load_profile(pack / "hct-library.yaml")


def test_the_engine_refuses_ns_cmsis_nn_contracts_it_would_not_read(tmp_path: Path) -> None:
    checkout = write_checkout(tmp_path / "nn", [function("arm_fx_s8")])
    (checkout / "CMakeLists.txt").write_text("")
    pack = checkout / "Tests" / "hct"
    pack.mkdir(parents=True)
    (pack / "hct-library.yaml").write_text("schema: hct/library-profile/1\nlibrary: ns-cmsis-nn\nsource_root: ../..\n"
                                           "contracts:\n  signatures: ../KernelContracts/kernel_contracts.json\n")
    assert require_supported(load_profile(pack / "hct-library.yaml"))
    (pack / "hct-library.yaml").write_text("schema: hct/library-profile/1\nlibrary: ns-cmsis-nn\nsource_root: ../..\n"
                                           "contracts:\n  signatures: sigs.json\n")
    with pytest.raises(LibraryProfileError, match="this hct runs ns-cmsis-nn from .*kernel_contracts.json"):
        require_supported(load_profile(pack / "hct-library.yaml"))


# hct contract validate.

def _validate(*args: str):
    result = CliRunner().invoke(cli_module.app, ["contract", "validate", *args])
    return result, (result.stdout or "") + (result.stderr or "")


def test_validate_a_generic_library(tmp_path: Path) -> None:
    root = _dsp(tmp_path, records=[_record(), _record("init", "arm_fir_init_f32", numerics={
        "reference": None, "claim": "bit_exact", "holds_with": []})])
    result, text = _validate("--library", str(root / "Testing" / "hct"))
    assert result.exit_code == 0, text
    assert "✓ Library: cmsis-dsp at" in text and "hct/kernel-contracts/1, 2 functions in 2 header(s)" in text
    assert "2 implementations (cmsis-dsp 1.16.2); numerics: 1 bit_exact, 1 bounded; 1 name a reference" in text


def test_validate_without_a_manifest_warns(tmp_path: Path) -> None:
    result, text = _validate("--library", str(_dsp(tmp_path) / "Testing" / "hct"))
    assert result.exit_code == 0 and "⚠ implementation manifest: none" in text


def test_validate_an_ns_cmsis_nn_checkout(tmp_path: Path) -> None:
    checkout = write_checkout(tmp_path / "nn", [function("arm_fx_s8")], implementations=[
        implementation("fx_s8", "arm_fx_s8", op="avg_pool")])
    (checkout / "CMakeLists.txt").write_text("")
    (checkout / "Include" / "arm_nnfunctions.h").write_text("arm_cmsis_nn_status arm_fx_s8(void);\n")
    for flag in ("--library", "--cmsis-nn-root"):
        clear_contract_cache()
        result, text = _validate(flag, str(checkout))
        assert result.exit_code == 0, text
        assert "ns-cmsis-nn/kernel-contracts/1, 1 functions" in text and "1 implementations (ns-cmsis-nn 7.47.0)" in text


def test_validate_refuses_a_bad_document(tmp_path: Path) -> None:
    root = _dsp(tmp_path, records=[_record(op="Add")])
    result, text = _validate("--library", str(root / "Testing" / "hct"))
    assert result.exit_code == contract_cli.EXIT_INVALID and "op: 'Add' does not match" in text


def test_validate_refuses_a_contract_for_another_library(tmp_path: Path) -> None:
    root = _dsp(tmp_path)
    _edit(root / "Testing" / "hct" / "kernel_contracts.json", lambda d: d.update(library="helia-dsp"))
    result, text = _validate("--library", str(root / "Testing" / "hct"))
    assert result.exit_code == contract_cli.EXIT_INVALID
    assert "is for library 'helia-dsp', but the profile is for 'cmsis-dsp'" in text


def test_validate_with_nothing_to_check(tmp_path: Path) -> None:
    pack = tmp_path / "lib"
    pack.mkdir()
    (pack / "hct-library.yaml").write_text("schema: hct/library-profile/1\nlibrary: cmsis-dsp\n")
    result, text = _validate("--library", str(pack))
    assert result.exit_code == contract_cli.EXIT_UNAVAILABLE and "names no contracts" in text
    old = tmp_path / "v7.35.1"
    (old / "Include").mkdir(parents=True)
    result, text = _validate("--cmsis-nn-root", str(old))
    assert result.exit_code == contract_cli.EXIT_UNAVAILABLE and "has no signature export" in text


def test_validate_a_named_export_that_is_missing(tmp_path: Path) -> None:
    root = _dsp(tmp_path)
    (root / "Testing" / "hct" / "kernel_contracts.json").unlink()
    result, text = _validate("--library", str(root / "Testing" / "hct"))
    assert result.exit_code == contract_cli.EXIT_INVALID and "the signature export the library names does not exist" \
        in text


def test_validate_refuses_a_bad_library(tmp_path: Path) -> None:
    result, text = _validate("--library", str(tmp_path / "missing"))
    assert result.exit_code == contract_cli.EXIT_UNAVAILABLE and "library directory not found" in text
