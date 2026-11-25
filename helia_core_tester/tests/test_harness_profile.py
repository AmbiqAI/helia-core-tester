"""The profile's harness block (core/harness.py): parsing, defaults, hct_library.h and its consumers."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import pytest

from helia_core_tester.contract import legacy_nn
from helia_core_tester.contract.bind import ContractBindError, bind, check_types, takes
from helia_core_tester.contract.ir import FunctionDecl, ParamDecl
from helia_core_tester.contract.render import load_current_contracts
from helia_core_tester.core import harness as harness_mod
from helia_core_tester.core.harness import (
    HarnessProfileError,
    current_harness,
    header_dir,
    legacy_harness,
    parse_harness,
    render_header,
    write_header,
)
from helia_core_tester.core.library import LibraryProfileError, load_profile, resolve_library
from helia_core_tester.generation.ops.ConvolutionFunctions.convolve import convolve_argument_pool
from helia_core_tester.reporting.parser import TestResultParser as ResultParser
from helia_core_tester.tests.harness_render import CONVOLVE_VALIDATION_KEY, convolve_context, render_pool

DSP = {
    "includes": ["arm_math.h"],
    "header_includes": ["arm_math.h", "dsp/basic_math_functions.h"],
    "status": {
        "type": "arm_status",
        "success": "ARM_MATH_SUCCESS",
        "argument_error": "ARM_MATH_ARGUMENT_ERROR",
        "codes": {"ARM_MATH_SUCCESS": 0, "ARM_MATH_ARGUMENT_ERROR": -1, "ARM_MATH_LENGTH_ERROR": -2,
                  "ARM_MATH_SIZE_MISMATCH": -3},
    },
    "alias_groups": {"input": ["pSrc", "pSrcA", "pIn"], "output": ["pDst", "pOut"]},
}


def _dsp(**changes) -> dict:
    data = copy.deepcopy(DSP)
    for path, value in changes.items():
        *parents, leaf = path.split("__")
        node = data
        for key in parents:
            node = node[key]
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return data


def _profile(directory: Path, body: str) -> Path:
    path = directory / "hct-library.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _fresh_cache():
    harness_mod._harness_of_root.cache_clear()
    yield
    harness_mod._harness_of_root.cache_clear()


def test_dsp_block_parses() -> None:
    parsed = parse_harness("cmsis-dsp", DSP)
    assert parsed.status.success == "ARM_MATH_SUCCESS"
    assert parsed.status.not_implemented is None
    assert parsed.status.names()[-3] == "ARM_MATH_SIZE_MISMATCH"
    assert parsed.group_of()["pSrcA"] == ("pSrc", "pSrcA", "pIn")
    assert parsed.context is None and parsed.dims is None


@pytest.mark.parametrize("changes, message", [
    ({"includes": []}, "non-empty list"),
    ({"includes": ["../arm_math.h"]}, "not a relative header name"),
    ({"includes": ["/usr/include/arm_math.h"]}, "not a relative header name"),
    ({"includes": ["arm_math.c"]}, "not a relative header name"),
    ({"includes": ['arm"math.h']}, "not a relative header name"),
    ({"includes": ["arm_math.h\n"]}, "not a relative header name"),
    ({"includes": ["arm..math.h"]}, "not a relative header name"),
    ({"includes": ["dsp/..h"]}, "not a relative header name"),
    ({"header_includes": ["arm_math.h", "arm_math.h"]}, "lists a header twice"),
    ({"status__success": "ARM_MATH_OK"}, "not one of its codes"),
    ({"status__argument_error": "ARM_MATH_SUCCESS"}, "name the same code"),
    ({"status__not_implemented": "ARM_MATH_NOPE"}, "not one of its codes"),
    ({"status__type": "arm status"}, "not a C identifier"),
    ({"status__type": "arm_status\n"}, "not a C identifier"),
    ({"context": "lib_context\n"}, "not a C identifier"),
    ({"alias_groups": {"input": ["pSrc", "pSrcA\n"]}}, "not a C identifier"),
    ({"status__type": None}, "not a C identifier"),
    ({"status__success": None}, "missing success"),
    ({"status__codes": {}}, "at least one"),
    ({"status__codes": {"ARM_MATH_SUCCESS": 0, "ARM_MATH_ARGUMENT_ERROR": 0}}, "name more than one code"),
    ({"status__codes": {"ARM_MATH_SUCCESS": 0, "ARM_MATH_ARGUMENT_ERROR": True}}, "not a 32-bit integer"),
    ({"status__codes": {"ARM_MATH_SUCCESS": 0, "ARM_MATH_ARGUMENT_ERROR": 2**31}}, "not a 32-bit integer"),
    ({"status__codes": {"ARM_MATH_SUCCESS": 0, "ARM_MATH_ARGUMENT_ERROR": -(2**31) - 1}}, "not a 32-bit integer"),
    ({"status__codes": {"ARM_MATH_SUCCESS": 0, "1BAD": -1}}, "not a C identifier"),
    ({"alias_groups": {"input": ["pSrc"]}}, "at least two spellings"),
    ({"alias_groups": {"input": ["pSrc", "pDst"], "output": ["pDst", "pOut"]}}, "in both input and output"),
    ({"alias_groups": {"input": ["pSrc", "p-Src"]}}, "not a C identifier"),
    ({"context": "struct ctx"}, "not a C identifier"),
    ({"status": None}, "missing status"),
    ({"extra": 1}, "unknown field"),
])
def test_block_refusals(changes, message) -> None:
    with pytest.raises(HarnessProfileError, match=message):
        parse_harness("cmsis-dsp", _dsp(**changes))


def test_boundary_codes_are_accepted() -> None:
    parsed = parse_harness("cmsis-dsp", _dsp(status__codes={"ARM_MATH_SUCCESS": 2**31 - 1,
                                                           "ARM_MATH_ARGUMENT_ERROR": -(2**31)}))
    assert parsed.status.value("ARM_MATH_ARGUMENT_ERROR") == -(2**31)


def test_require_names_the_missing_field() -> None:
    parsed = parse_harness("cmsis-dsp", DSP)
    for field in ("context", "dims", "not_implemented"):
        with pytest.raises(HarnessProfileError, match=field):
            parsed.require(field)
    assert legacy_harness().require("context") == "cmsis_nn_context"


def test_legacy_block_matches_ns_cmsis_nn_headers() -> None:
    """The built-in codes are ns-cmsis-nn's own; hct_library.h also asserts them at compile time."""
    root = os.environ.get("CMSIS_NN_ROOT")
    header = Path(root) / "Include" / "arm_nn_types.h" if root else None
    if header is None or not header.is_file():
        pytest.skip("CMSIS_NN_ROOT with ns-cmsis-nn headers required")
    defined = dict((name, int(value)) for name, value in
                   re.findall(r"^\s*(ARM_CMSIS_NN_\w+)\s*=\s*(-?\d+)", header.read_text(), re.M))
    assert dict(legacy_harness().status.codes) == defined


def test_profile_loads_its_harness(tmp_path: Path) -> None:
    body = (
        "schema: hct/library-profile/1\nlibrary: cmsis-dsp\nharness:\n"
        "  includes: [arm_math.h]\n  header_includes: [arm_math.h]\n"
        "  status:\n    type: arm_status\n    success: ARM_MATH_SUCCESS\n    argument_error: ARM_MATH_ARGUMENT_ERROR\n"
        "    codes: {ARM_MATH_SUCCESS: 0, ARM_MATH_ARGUMENT_ERROR: -1}\n"
    )
    profile = load_profile(_profile(tmp_path, body))
    assert profile.harness.library == "cmsis-dsp"
    assert profile.harness.status.type == "arm_status"


def test_other_library_without_a_block_has_none(tmp_path: Path) -> None:
    profile = load_profile(_profile(tmp_path, "schema: hct/library-profile/1\nlibrary: cmsis-dsp\n"))
    assert profile.harness is None
    with pytest.raises(HarnessProfileError, match="no harness block"):
        harness_mod._harness_of_root(str(tmp_path))


def test_ns_cmsis_nn_profile_without_a_block_takes_the_built_in_one(tmp_path: Path) -> None:
    profile = load_profile(_profile(tmp_path, "schema: hct/library-profile/1\nlibrary: ns-cmsis-nn\n"))
    assert profile.harness == legacy_harness()


@pytest.mark.parametrize("block, message", [
    ("harness:\n  includes: [arm_math.h]\n  header_includes: [arm_math.h]\n  bogus: 1\n  status: {}\n",
     "harness"),
    ("harness:\n  includes: [arm_math.h]\n  header_includes: [arm_math.h]\n"
     "  status: {type: arm_status, success: OK, argument_error: BAD, codes: {OK: 0}}\n", "not one of its codes"),
    ("harness:\n  includes: [arm_math.h]\n  header_includes: [arm_math.h]\n"
     "  status: {type: 'arm status', success: OK, argument_error: BAD, codes: {OK: 0, BAD: -1}}\n", "harness"),
])
def test_profile_refuses_a_bad_block(tmp_path: Path, block: str, message: str) -> None:
    path = _profile(tmp_path, "schema: hct/library-profile/1\nlibrary: cmsis-dsp\n" + block)
    with pytest.raises(LibraryProfileError, match=message):
        load_profile(path)


def test_current_harness_follows_the_resolved_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import helia_core_tester.generation.utils.temp_sizer_probe as probe

    monkeypatch.delenv("CMSIS_NN_ROOT", raising=False)
    monkeypatch.setattr(probe, "resolve_cmsis_nn_root", lambda: None)
    assert current_harness() == legacy_harness()
    _profile(tmp_path, "schema: hct/library-profile/1\nlibrary: cmsis-dsp\n")
    monkeypatch.setattr(probe, "resolve_cmsis_nn_root", lambda: tmp_path)
    with pytest.raises(HarnessProfileError, match="no harness block"):
        current_harness()
    assert resolve_library(tmp_path).harness is None


def test_header_carries_includes_macros_and_asserts() -> None:
    text = render_header(parse_harness("cmsis-dsp", DSP))
    assert '#include "arm_math.h"' in text
    assert "dsp/basic_math_functions.h" not in text, "header_includes belong to the case header only"
    assert "#define HCT_STATUS_SUCCESS ARM_MATH_SUCCESS" in text
    assert "#define HCT_STATUS_ARGUMENT_ERROR ARM_MATH_ARGUMENT_ERROR" in text
    assert text.count("_Static_assert(") == 4
    assert "_Static_assert((int)ARM_MATH_SIZE_MISMATCH == -3," in text


def test_write_header_keeps_a_current_file(tmp_path: Path) -> None:
    harness = parse_harness("cmsis-dsp", DSP)
    path = write_header(tmp_path, harness)
    os.utime(path, (1, 1))
    write_header(tmp_path, harness)
    assert path.stat().st_mtime == 1, "an unchanged header must not trigger a rebuild"
    write_header(tmp_path, legacy_harness())
    assert "ARM_CMSIS_NN_SUCCESS" in path.read_text()
    assert [p.name for p in tmp_path.iterdir()] == ["hct_library.h"], "no temporary file is left behind"


def test_header_dir_is_keyed_by_content(tmp_path: Path) -> None:
    dsp = header_dir(parse_harness("cmsis-dsp", DSP), tmp_path)
    nn = header_dir(legacy_harness(), tmp_path)
    assert dsp != nn
    assert header_dir(parse_harness("cmsis-dsp", DSP), tmp_path) == dsp
    assert (dsp / "hct_library.h").read_text() == render_header(parse_harness("cmsis-dsp", DSP))


def _decl(*names: str) -> FunctionDecl:
    return FunctionDecl(name="arm_add_q15", header="Include/dsp/basic_math_functions.h", line=1, guards=(),
                        returns="void", params=tuple(ParamDecl(n, "const q15_t *", "in") for n in names))


def test_bind_uses_the_profile_groups() -> None:
    dsp = parse_harness("cmsis-dsp", DSP)
    decl = _decl("pSrcA", "pDst")
    assert bind(decl, {"pSrc": "a", "pOut": "o"}, harness=dsp) == {"pSrcA": "a", "pDst": "o"}
    assert takes(decl, "pSrc", harness=dsp)
    assert not takes(decl, "input_data", harness=dsp)
    with pytest.raises(ContractBindError, match="several spellings"):
        bind(decl, {"pSrc": "a", "pIn": "b", "pDst": "o"}, harness=dsp)
    with pytest.raises(ContractBindError, match="cannot supply"):
        bind(decl, {"input_data": "a", "pDst": "o"}, harness=dsp)


def test_check_types_refuses_a_role_the_profile_does_not_group() -> None:
    dsp = parse_harness("cmsis-dsp", DSP)
    with pytest.raises(ContractBindError, match="unknown tensor role 'filter'"):
        check_types(_decl("pSrcA", "pDst"), {"filter": "S8"}, harness=dsp)


def test_parser_names_the_returned_status() -> None:
    names = parse_harness("cmsis-dsp", DSP).status.names()
    parser = ResultParser(status_names=names)
    out = "Running test: x\nadd failed with status -2\n1 Failures\n"
    result = parser.parse_fvp_output(out, Path("x.elf"), "cortex-m55", 0.0)
    assert result.failure_reason == "add API error (status -2, ARM_MATH_LENGTH_ERROR)"
    unknown = parser.parse_fvp_output(out.replace("-2", "-9"), Path("x.elf"), "cortex-m55", 0.0)
    assert unknown.failure_reason == "add API error (status -9)"


def _renamed_nn(**changes) -> dict:
    """ns-cmsis-nn's block with its status names changed, so a rendered name can only come from it."""
    data = copy.deepcopy(legacy_nn.HARNESS)
    data["status"] = {"type": "lib_status", "success": "LIB_OK", "argument_error": "LIB_BAD_ARG",
                      "not_implemented": "LIB_NO_IMPL", "codes": {"LIB_OK": 0, "LIB_BAD_ARG": -1, "LIB_NO_IMPL": -2}}
    data["context"] = "lib_context"
    data.update(changes)
    return data


def _render_convolve(harness_data: dict, **overrides) -> tuple[str, str]:
    context = convolve_context("arm_convolve_wrapper_s8", **overrides)
    pool = convolve_argument_pool(context, has_biases=True, bias_is_struct=False)
    return render_pool(context, pool, stem="convolve", validation_key=CONVOLVE_VALIDATION_KEY, label="Convolution",
                       contracts=load_current_contracts(), harness=parse_harness("lib", harness_data))


def test_harness_renders_the_profile_names() -> None:
    header, source = _render_convolve(_renamed_nn(), expected_status="LIB_BAD_ARG")
    # The parity assert spells the kernel's own prototype from the contract; every status the
    # harness itself names comes from the profile.
    assert "ARM_CMSIS_NN" not in source
    assert "static lib_context conv_case_ctx;" in source
    assert "return LIB_OK;" in source
    assert "        LIB_BAD_ARG\n" in source, "a non-success expectation validates against the named code"
    assert "defined by lib" not in source and "the lib export records" in source
    assert header.count('#include "arm_nnfunctions.h"') == 1 and '#include "arm_nn_types.h"' in header


def test_harness_refuses_a_context_the_profile_lacks() -> None:
    with pytest.raises(HarnessProfileError, match="context"):
        _render_convolve(_renamed_nn(context=None))


DSP_PROFILE = (
    "schema: hct/library-profile/1\nlibrary: cmsis-dsp\nharness:\n"
    "  includes: [arm_math.h]\n  header_includes: [arm_math.h]\n"
    "  status:\n    type: arm_status\n    success: ARM_MATH_SUCCESS\n    argument_error: ARM_MATH_ARGUMENT_ERROR\n"
    "    codes: {ARM_MATH_SUCCESS: 0, ARM_MATH_ARGUMENT_ERROR: -1}\n"
)


def test_cmsis_nn_root_is_read_as_given_whatever_its_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A root without Include/ is not the ns-cmsis-nn layout, but its profile still decides."""
    _profile(tmp_path, DSP_PROFILE)
    monkeypatch.setenv("CMSIS_NN_ROOT", str(tmp_path))
    assert current_harness().status.success == "ARM_MATH_SUCCESS"
    monkeypatch.setenv("CMSIS_NN_ROOT", str(tmp_path / "missing"))
    with pytest.raises(HarnessProfileError, match="is not a directory"):
        current_harness()


def test_host_builds_take_the_harness_of_the_tree_they_compile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Host check, record and mutation compile in hct's own process, where CMSIS_NN_ROOT need not be set."""
    from helia_core_tester.mutation.host_build import library_header_flags

    tree = tmp_path / "dsp"
    tree.mkdir()
    _profile(tree, DSP_PROFILE)
    monkeypatch.delenv("CMSIS_NN_ROOT", raising=False)
    monkeypatch.setenv(harness_mod.HEADER_ENV, str(tmp_path / "headers"))
    flag, directory = library_header_flags(tree)
    assert flag == "-I"
    assert "#define HCT_STATUS_SUCCESS ARM_MATH_SUCCESS" in (Path(directory) / "hct_library.h").read_text()
    _, nn = library_header_flags(tmp_path)
    assert "ARM_CMSIS_NN_SUCCESS" in (Path(nn) / "hct_library.h").read_text()


def test_host_runtime_cache_key_follows_the_tree_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from helia_core_tester.generation.reference import host_kernels

    dsp, nn = tmp_path / "dsp", tmp_path / "nn"
    for tree in (dsp, nn):
        (tree / "Include").mkdir(parents=True)
        (tree / "Source").mkdir()
    _profile(dsp, DSP_PROFILE)
    monkeypatch.setattr(host_kernels, "compiler_identity", lambda cc: {"cc": "fixed"})
    assert host_kernels.host_kernels_key(dsp, "m0", "cc") != host_kernels.host_kernels_key(nn, "m0", "cc")


def test_expected_status_is_checked_against_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from helia_core_tester.generation.ops._shared.base import OperationBase

    dsp = parse_harness("cmsis-dsp", DSP)
    monkeypatch.setattr(harness_mod, "current_harness", lambda: dsp)

    def status(desc):
        return OperationBase.expected_status(SimpleNamespace(desc=desc))

    assert status({"name": "c"}) == "ARM_MATH_SUCCESS"
    assert status({"name": "c", "expected_status": "ARM_MATH_LENGTH_ERROR"}) == "ARM_MATH_LENGTH_ERROR"
    with pytest.raises(ValueError, match="known values are .*ARM_MATH_SIZE_MISMATCH"):
        status({"name": "c", "expected_status": "ARM_CMSIS_NN_ARG_ERROR"})
    with pytest.raises(ValueError, match="autovectorize_declines"):
        status({"name": "c", "expected_status": "ARM_MATH_ARGUMENT_ERROR", "autovectorize_declines": True})


def test_run_step_passes_the_configured_checkout_to_the_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The FVP runner names status codes from the profile, so it must see the checkout --library chose."""
    import subprocess

    from helia_core_tester.core.config import Config
    from helia_core_tester.core.steps.run import RunStep
    from helia_core_tester.tests.test_fvp_parallel_control import _make_config_root, _runtime_env

    checkout = tmp_path / "ns"
    (checkout / "Include").mkdir(parents=True)
    (checkout / "Include" / "arm_nnfunctions.h").write_text("")
    (checkout / "CMakeLists.txt").write_text("")
    seen = []
    monkeypatch.setattr("helia_core_tester.core.steps.run.subprocess.run",
                        lambda cmd, cwd, check, text, bufsize, env=None: seen.append(env)
                        or subprocess.CompletedProcess(cmd, 0))
    root = _make_config_root(tmp_path / "tester")
    cfg = Config(project_root=root, cmsis_nn_root=checkout, enable_reporting=False)
    assert RunStep(cfg, runtime_env=_runtime_env(root))._do_execute().success
    assert seen and all(env["CMSIS_NN_ROOT"] == str(checkout.resolve()) for env in seen)
