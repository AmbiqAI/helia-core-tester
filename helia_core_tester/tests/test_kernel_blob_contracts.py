"""Per-kernel CASE_META blob contracts (issue #145).

Adapters cast blob pointers by kernel id, so the registry's `blobs:` map is the dtype
each role must carry. The catalog generator renders it into firmware, the bridge
checks it on the host, and the firmware rejects a CASE_META that breaks it (the
negative firmware paths run in benchmark_server_session_host_main.c).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

from helia_core_tester.hardware import kernel_registry
from helia_core_tester.hardware.generated_test_bridge import build_case_bundle_from_generated_test
from helia_core_tester.hardware.kernel_registry import (
    BlobContractError,
    blob_contract,
    check_blob_contract,
    load_kernel_registry,
    lookup_kernel_id,
)
from helia_core_tester.tests.test_generated_test_bridge_recurrent import _case, _svdf_header

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "assets" / "kernel_registry.yaml"
CATALOG_C = PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_catalog.c"
SESSION_C = PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_session.c"
BLOB_TYPES_H = PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_blob_types.h"

ABS_S8 = 1
CONVOLVE_S8 = 2
CONVOLVE_S16 = 14
DEPTHWISE_CONV_S16 = 15


@pytest.fixture(scope="module")
def catalog_generator():
    spec = importlib.util.spec_from_file_location("generate_kernel_catalog", PROJECT_ROOT / "scripts" / "generate_kernel_catalog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_registry_entry_declares_a_valid_contract(catalog_generator) -> None:
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))["kernels"]
    for entry in raw:
        catalog_generator._validate_blob_contract(entry)
    assert all(kernel.blobs for kernel in load_kernel_registry(PROJECT_ROOT))
    assert len(load_kernel_registry(PROJECT_ROOT)) == len(raw)


@pytest.mark.parametrize(
    ("blobs", "message"),
    [
        (None, "needs a non-empty `blobs:` map"),
        ({}, "needs a non-empty `blobs:` map"),
        (["input_0"], "needs a non-empty `blobs:` map"),
        ({"output": "S8"}, "unknown role 'output'"),
        ({"expected_output": "S8"}, "unknown role 'expected_output'"),
        ({"input_0": "S9"}, "unknown dtype 'S9'"),
        ({"input_0": "F32"}, "unknown dtype 'F32'"),
        ({"bias": []}, "role 'bias' has an empty dtype list"),
        ({"bias": ["S32", "S9"]}, "unknown dtype 'S9'"),
        ({"bias": ["S32", "S32"]}, "lists a dtype more than once"),
    ],
)
def test_catalog_generator_rejects_bad_contracts(catalog_generator, blobs, message) -> None:
    entry = {"kernel_id": 999}
    if blobs is not None:
        entry["blobs"] = blobs
    with pytest.raises(ValueError, match=re.escape(message)):
        catalog_generator._validate_blob_contract(entry)


def test_catalog_table_matches_registry(catalog_generator) -> None:
    text = CATALOG_C.read_text(encoding="utf-8")
    rows = dict(re.findall(r"^    \{(\d+)u, \{(.*)\}\},$", text, re.M))
    kernels = load_kernel_registry(PROJECT_ROOT)
    assert sorted(int(kernel_id) for kernel_id in rows) == sorted(kernel.kernel_id for kernel in kernels)
    for kernel in kernels:
        fields = {
            role: set(re.findall(r"HCT_DTYPE_BIT\((HCT_DTYPE_[A-Z0-9]+)\)", bits))
            for role, bits in re.findall(r"\[HCT_BLOB_ROLE_([A-Z0-9_]+)\] = ([^\[\]]+?)(?=, \[|$)", rows[str(kernel.kernel_id)])
        }
        expected = {
            role.upper(): {catalog_generator._BLOB_DTYPE_MACROS[dtype] for dtype in dtypes} for role, dtypes in kernel.blobs
        }
        assert fields == expected, kernel.kernel_id
    # Only the wrapper that takes cmsis_nn_bias_data accepts two bias widths.
    multi = {kernel.cmsis_function for kernel in kernels if any(len(dtypes) > 1 for _, dtypes in kernel.blobs)}
    assert multi == {"arm_convolve_wrapper_s16"}
    assert dict(blob_contract(PROJECT_ROOT, CONVOLVE_S16))["bias"] == ("S32", "S64")
    assert dict(blob_contract(PROJECT_ROOT, DEPTHWISE_CONV_S16))["bias"] == ("S64",)


def test_generator_names_match_firmware_wire_codes(catalog_generator) -> None:
    """Roles and dtypes the generator accepts are exactly the ones CASE_META can decode."""
    session = SESSION_C.read_text(encoding="utf-8")
    roles = dict(re.findall(r'strcmp\(name, "([a-z0-9_]+)"\) == 0\) return (HCT_BLOB_ROLE_[A-Z0-9_]+);', session))
    dtypes = dict(re.findall(r'strcmp\(name, "([A-Z0-9]+)"\) == 0\) return (HCT_DTYPE_[A-Z0-9]+);', session))
    assert roles == {role: f"HCT_BLOB_ROLE_{role.upper()}" for role in catalog_generator._BLOB_ROLES}
    assert dtypes == catalog_generator._BLOB_DTYPE_MACROS
    header = BLOB_TYPES_H.read_text(encoding="utf-8")
    # Designated initializers leave unlisted roles at 0, which must mean "not accepted".
    assert re.search(r"#define HCT_DTYPE_UNKNOWN 0u\b", header)
    role_codes = [int(code) for code in re.findall(r"#define HCT_BLOB_ROLE_[A-Z0-9_]+ (\d+)u", header)]
    count = int(re.search(r"#define HCT_BLOB_ROLE_COUNT (\d+)u", header).group(1))
    assert max(role_codes[:-1]) == count - 1 == len(catalog_generator._BLOB_ROLES)


@pytest.mark.parametrize(
    ("operator", "dtype", "role", "expected"),
    [
        ("StridedSlice", "S8", "input_0", "S8"),
        ("StridedSlice", "S16", "input_0", "S16"),
        ("StridedSlice", "S32", "input_0", "S32"),
        ("StridedSlice", "S16", "meta_0", "S32"),
    ],
)
def test_data_movement_operand_dtype_follows_kernel(operator, dtype, role, expected) -> None:
    kernel_id = next(
        kernel.kernel_id
        for kernel in load_kernel_registry(PROJECT_ROOT)
        if kernel.operator == operator and kernel.dtype == dtype
    )
    assert blob_contract(PROJECT_ROOT, kernel_id)[role] == (expected,)


def test_reverse_sequence_lengths_are_s32() -> None:
    for kernel in load_kernel_registry(PROJECT_ROOT):
        if kernel.operator == "ReverseSequence":
            assert dict(kernel.blobs)["input_1"] == ("S32",)


def test_check_blob_contract_accepts_valid_and_partial_sets() -> None:
    check_blob_contract(PROJECT_ROOT, ABS_S8, [("input_0", "S8")])
    # Optional roles (no bias, no per-channel quant) may be absent.
    check_blob_contract(PROJECT_ROOT, CONVOLVE_S8, [("input_0", "S8"), ("weights", "S8")])
    for bias_dtype in ("S32", "S64"):
        check_blob_contract(PROJECT_ROOT, CONVOLVE_S16, [("input_0", "S16"), ("weights", "S8"), ("bias", bias_dtype)])


@pytest.mark.parametrize(
    ("kernel_id", "blobs", "message"),
    [
        (ABS_S8, [("input_0", "S16")], "role 'input_0' is S16, contract requires S8"),
        (ABS_S8, [("input_0", "S8"), ("weights", "S8")], "role 'weights' is not accepted"),
        (ABS_S8, [("input_0", "S8"), ("input_0", "S8")], "role 'input_0' appears more than once"),
        (CONVOLVE_S8, [("input_0", "S8"), ("bias", "S8")], "role 'bias' is S8, contract requires S32"),
        (CONVOLVE_S16, [("input_0", "S16"), ("bias", "S16")], "role 'bias' is S16, contract requires S32 or S64"),
        (DEPTHWISE_CONV_S16, [("input_0", "S16"), ("bias", "S32")], "role 'bias' is S32, contract requires S64"),
        (0, [("input_0", "S8")], "kernel_id=0 has no `blobs:` contract"),
        (10_000, [("input_0", "S8")], "kernel_id=10000 has no `blobs:` contract"),
    ],
)
def test_check_blob_contract_rejects(kernel_id, blobs, message) -> None:
    with pytest.raises(BlobContractError, match=re.escape(message)):
        check_blob_contract(PROJECT_ROOT, kernel_id, blobs)


def test_check_blob_contract_reports_every_problem() -> None:
    with pytest.raises(BlobContractError) as excinfo:
        check_blob_contract(PROJECT_ROOT, ABS_S8, [("input_0", "S16"), ("bias", "S32")])
    assert "input_0" in str(excinfo.value) and "bias" in str(excinfo.value)


def test_bridge_fails_closed_on_contract_mismatch(tmp_path: Path, monkeypatch) -> None:
    """A bundle that breaks its contract is a bridge/registry bug, not an unsupported case."""
    name = "svdf_contract_s8"
    case = _case(tmp_path, "SVDFunctions", {"operator": "SVDF", "name": name, "activation_dtype": "S8"},
                 _svdf_header(name, state_type="int8_t", bias=True))
    svdf_id = lookup_kernel_id(PROJECT_ROOT, family="SVDFunctions", operator="SVDF")
    build_case_bundle_from_generated_test(PROJECT_ROOT, case, output_root=tmp_path / "ok", require_fvp_pass=False)

    real = load_kernel_registry(PROJECT_ROOT)
    narrowed = tuple(
        dataclasses.replace(kernel, blobs=tuple(pair for pair in kernel.blobs if pair[0] != "bias"))
        if kernel.kernel_id == svdf_id
        else kernel
        for kernel in real
    )
    monkeypatch.setattr(kernel_registry, "load_kernel_registry", lambda _root: narrowed)
    with pytest.raises(BlobContractError, match=re.escape("role 'bias' is not accepted")):
        build_case_bundle_from_generated_test(PROJECT_ROOT, case, output_root=tmp_path / "bad", require_fvp_pass=False)
