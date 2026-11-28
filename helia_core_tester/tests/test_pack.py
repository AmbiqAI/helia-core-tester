"""A library's own pack: the profile block, its modules, its descriptors and the cases they render."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from typer.testing import CliRunner

from helia_core_tester.contract import ir as contract_ir
from helia_core_tester.core import harness as harness_mod
from helia_core_tester.core.library import LibraryProfileError, load_profile, require_supported
from helia_core_tester.generation.reference import host_kernels as hk
from helia_core_tester.pack import API_VERSION, CaseSpec, PackError, Registry, api_version_problem
from helia_core_tester.pack import cli as pack_cli
from helia_core_tester.pack import loader as pack_loader
from helia_core_tester.pack.descriptors import load_case_descriptors, load_pack_descriptors

_HARNESS = (
    "harness:\n  includes: [dsp.h]\n  header_includes: [dsp.h]\n"
    "  status:\n    type: dsp_status\n    success: DSP_OK\n    argument_error: DSP_ARG\n"
    "    codes: {DSP_OK: 0, DSP_ARG: -1, DSP_LEN: -2}\n"
)
_FUNCTIONS = [
    ("arm_add_q15", [("in", "pSrcA", "const q15_t *"), ("in", "pSrcB", "const q15_t *"),
                     ("out", "pDst", "q15_t *"), ("in", "blockSize", "uint32_t")]),
    ("arm_add_f32", [("in", "pSrcA", "const float32_t *"), ("in", "pSrcB", "const float32_t *"),
                     ("out", "pDst", "float32_t *"), ("in", "blockSize", "uint32_t")]),
    ("arm_offset_q15", [("in", "pSrc", "const q15_t *"), ("in", "offset", "q15_t"),
                        ("out", "pDst", "q15_t *"), ("in", "blockSize", "uint32_t")]),
    ("arm_max_q15", [("in", "pSrc", "const q15_t *"), ("in", "blockSize", "uint32_t"),
                     ("out", "pResult", "q15_t *"), ("out", "pIndex", "uint32_t *")]),
    ("arm_add_f64", [("in", "pSrcA", "const float64_t *"), ("in", "pSrcB", "const float64_t *"),
                     ("out", "pDst", "float64_t *"), ("in", "blockSize", "uint32_t")]),
    ("arm_state_q15", [("in", "pSrc", "const q15_t *"), ("in,out", "pState", "q15_t *"),
                       ("out", "pDst", "q15_t *"), ("in", "blockSize", "uint32_t")]),
    ("arm_mix_f32", [("in", "pSrc", "const float64_t *"), ("out", "pDst", "float32_t *"),
                     ("in", "blockSize", "uint32_t")]),
    ("arm_sized_q15", [("in", "pSrc", "const q15_t *"), ("out", "pDst", "q15_t *"), ("in", "blockSize", "size_t")]),
    ("arm_gather_q15", [("in", "pSrc", "const q15_t *"), ("in", "pIndex", "const int32_t *"),
                        ("out", "pDst", "q15_t *"), ("in", "blockSize", "uint32_t")]),
    ("arm_rows_q15", [("in", "pRows", "const q15_t * const *"), ("out", "pDst", "q15_t *"), ("in", "blockSize", "uint32_t")]),
]

_BINARY = '''
import numpy as np
from helia_core_tester.pack import CaseSpec

HCT_PACK_API = "1.0"


def binary(case):
    n = int(case.values.get("n", 9))
    dtype = np.int16 if case.param("pSrcA").element == "q15_t" else np.float32
    a = np.arange(n, dtype=dtype)
    b = (np.arange(n, dtype=dtype) * 3)
    golden = case.values.get("golden")
    return CaseSpec(arrays={"pSrcA": a, "pSrcB": b}, output="pDst", output_count=n,
                    scalars={"blockSize": n}, golden=None if golden is None else (a + b).astype(dtype))


def register(pack):
    pack.generator("basic.binary", binary, entries=["arm_add_q15", "arm_add_f32", "arm_add_f64", "arm_max_q15"])
'''


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch):
    for key in (harness_mod.PROFILE_ENV, "CMSIS_NN_ROOT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(harness_mod, "_REGISTERED", {})

    def clear() -> None:
        harness_mod._library_of_root.cache_clear()
        harness_mod._profile_file.cache_clear()
        pack_loader._CACHE.clear()
        contract_ir._CACHE.clear()

    clear()
    yield
    clear()


def _library(root: Path, *, pack: Optional[str] = None, modules: dict | None = None,
             descriptors: dict | None = None, library: str = "cmsis-dsp") -> Path:
    """A small DSP-like library: a header, its contract export and a profile below Testing/hct."""
    (root / "Include").mkdir(parents=True, exist_ok=True)
    header, records = [], []
    for name, params in _FUNCTIONS:
        header.append(f"void {name}({', '.join(f'{t} {n}' for _, n, t in params)});")
        records.append({"name": name, "header": "Include/dsp.h", "line": len(header), "guards": [],
                        "returns": "void",
                        "params": [{"direction": d, "name": n, "type": t} for d, n, t in params]})
    (root / "Include" / "dsp.h").write_text("\n".join(header) + "\n")
    pack_dir = root / "Testing" / "hct"
    (pack_dir / "descriptors").mkdir(parents=True, exist_ok=True)
    (pack_dir / "kernel_contracts.json").write_text(json.dumps(
        {"schema": "hct/kernel-contracts/1", "library": library, "functions": records}))
    for name, text in (modules if modules is not None else {"basic": _BINARY}).items():
        (pack_dir / f"{name}.py").write_text(text)
    for rel, text in (descriptors or {}).items():
        target = pack_dir / "descriptors" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    block = pack if pack is not None else (
        "pack:\n  api: \"1.0\"\n  descriptors: descriptors\n  modules: ["
        + ", ".join(modules if modules is not None else {"basic": _BINARY}) + "]\n")
    (pack_dir / "hct-library.yaml").write_text(
        f"schema: hct/library-profile/1\nlibrary: {library}\nsource_root: ../..\n"
        "contracts:\n  signatures: kernel_contracts.json\n" + _HARNESS + block)
    return pack_dir / "hct-library.yaml"


def _use(monkeypatch: pytest.MonkeyPatch, profile: Path) -> None:
    monkeypatch.setenv(harness_mod.PROFILE_ENV, str(profile))
    monkeypatch.setenv("CMSIS_NN_ROOT", str(load_profile(profile).source_root))


def _doc(name: str, entry: str = "arm_add_q15", **fields) -> str:
    lines = [f"schema: hct/descriptor/1", f"name: {name}", f"entry: {entry}"]
    lines += [f"{key}: {json.dumps(value)}" for key, value in fields.items()]
    return "\n".join(lines) + "\n"


def _cases(monkeypatch, tmp_path, descriptors: dict, **kwargs):
    profile = _library(tmp_path, descriptors=descriptors, **kwargs)
    _use(monkeypatch, profile)
    from helia_core_tester.generation.io.descriptors import load_all_descriptors

    return load_case_descriptors(load_all_descriptors)


# --- the API version ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, ok", [("1.0", True), (f"1.{API_VERSION[1]}", True), ("1.9", False),
                                       ("2.0", False), ("0.9", False), ("1", False), (1.0, False), (None, False)])
def test_api_version(value, ok) -> None:
    assert (api_version_problem(value) is None) == ok


# --- the profile block -------------------------------------------------------------------------

def test_the_profile_names_the_pack(tmp_path) -> None:
    profile = load_profile(_library(tmp_path))
    assert profile.pack.api == "1.0"
    assert profile.pack.root == (tmp_path / "Testing" / "hct").resolve()
    assert profile.pack.descriptors == profile.pack.root / "descriptors"
    assert profile.pack.modules == ("basic",)
    assert require_supported(profile) is profile


@pytest.mark.parametrize("block, said", [
    ('pack:\n  api: "2.0"\n  descriptors: descriptors\n', "not supported"),
    ('pack:\n  api: "1.9"\n  descriptors: descriptors\n', "not supported"),
    ('pack:\n  api: "one"\n  descriptors: descriptors\n', "does not match"),
    ('pack:\n  api: "1.0"\n', "'descriptors' is a required property"),
    ('pack:\n  api: "1.0"\n  descriptors: missing\n', "is not a directory"),
    ('pack:\n  api: "1.0"\n  descriptors: /abs\n', "relative POSIX path"),
    ('pack:\n  api: "1.0"\n  descriptors: ../../../outside\n', "must not contain '..'"),
    ('pack:\n  api: "1.0"\n  path: nowhere\n  descriptors: descriptors\n', "is not a directory"),
    ('pack:\n  api: "1.0"\n  descriptors: descriptors\n  modules: [absent]\n', "'absent' has no"),
    ('pack:\n  api: "1.0"\n  descriptors: descriptors\n  modules: ["bad-name"]\n', "does not match"),
    ('pack:\n  api: "1.0"\n  descriptors: descriptors\n  extra: 1\n', "Additional properties"),
])
def test_the_profile_refuses_a_bad_pack_block(tmp_path, block, said) -> None:
    with pytest.raises(LibraryProfileError, match=said):
        load_profile(_library(tmp_path, pack=block))


def test_a_pack_path_that_escapes_through_a_symlink_is_refused(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    lib = tmp_path / "lib"
    profile = _library(lib, pack='pack:\n  api: "1.0"\n  descriptors: link\n')
    (profile.parent / "link").symlink_to(outside)
    with pytest.raises(LibraryProfileError, match="outside source_root"):
        load_profile(profile)


@pytest.mark.parametrize("drop", ["pack", "harness", "contracts"])
def test_another_library_runs_only_from_a_whole_pack(tmp_path, drop) -> None:
    profile = _library(tmp_path)
    text = profile.read_text()
    if drop == "pack":
        text = text[:text.index("pack:")]
    elif drop == "harness":
        text = text[:text.index("harness:")] + text[text.index("pack:"):]
    else:
        text = text.replace("contracts:\n  signatures: kernel_contracts.json\n", "")
    profile.write_text(text)
    with pytest.raises(LibraryProfileError, match=f"must name {drop}"):
        require_supported(load_profile(profile))


# --- modules -------------------------------------------------------------------------------------

def _load(tmp_path, modules: dict) -> pack_loader.LoadedPack:
    return pack_loader.load_pack(load_profile(_library(tmp_path, modules=modules)))


@pytest.mark.parametrize("text, said", [
    ("raise RuntimeError('boom')\n", "import failed: RuntimeError: boom"),
    ("HCT_PACK_API = '1.0'\n", "defines no register"),
    ("def register(pack):\n    pass\n", "HCT_PACK_API: API version None"),
    ("HCT_PACK_API = '2.0'\ndef register(pack):\n    pass\n", "HCT_PACK_API: API version 2.0 is not supported"),
    ("HCT_PACK_API = '1.0'\ndef register(pack):\n    raise ValueError('nope')\n", "register\\(\\) failed: ValueError"),
    ("HCT_PACK_API = '1.0'\ndef register(pack):\n    pack.generator('a', len, entries=[])\n", "claims no entries"),
    ("HCT_PACK_API = '1.0'\ndef register(pack):\n    pack.generator('a', len, entries='arm_*')\n",
     "must be a list"),
    ("HCT_PACK_API = '1.0'\ndef register(pack):\n    pack.generator('a b', len, entries=['x'])\n",
     "not a dotted identifier"),
    ("HCT_PACK_API = '1.0'\ndef register(pack):\n    pack.generator('a', 3, entries=['x'])\n", "not callable"),
    ("HCT_PACK_API = '1.0'\ndef register(pack):\n    pack.generator('a', len, entries=['x;y'])\n",
     "not a C identifier"),
    ("HCT_PACK_API = '1.0'\ndef register(pack):\n    pack.generator('a', len, entries=['x'])\n"
     "    pack.generator('a', len, entries=['y'])\n", "already registered"),
])
def test_a_bad_module_is_refused(tmp_path, text, said) -> None:
    with pytest.raises(PackError, match=said):
        _load(tmp_path, {"m": text})


def test_the_registry_is_frozen_once_loaded(tmp_path) -> None:
    pack = _load(tmp_path, {"basic": _BINARY})
    with pytest.raises(PackError, match="frozen"):
        pack.registry.generator("late", len, entries=["x"])


def test_modules_import_each_other_and_cannot_shadow_the_standard_library(tmp_path) -> None:
    helper = "VALUE = 7\n"
    shadow = "raise RuntimeError('the pack shadowed json')\n"
    main = ("import json\nfrom . import helper\nHCT_PACK_API = '1.0'\n"
            "def register(pack):\n    pack.generator(f'g{helper.VALUE}', len, entries=['arm_*'])\n")
    profile = _library(tmp_path, modules={"main": main})
    (profile.parent / "helper.py").write_text(helper)
    (profile.parent / "json.py").write_text(shadow)
    pack = pack_loader.load_pack(load_profile(profile))
    assert "g7" in pack.registry.generators
    assert profile.parent / "helper.py" in pack.files
    import json as stdlib_json

    assert stdlib_json.dumps([1]) == "[1]"


def test_generator_lookup_needs_one_claim(tmp_path) -> None:
    registry = Registry()
    registry.generator("a", len, entries=["arm_add_*"])
    registry.generator("b", len, entries=["arm_add_q15"])
    with pytest.raises(PackError, match="claimed by a, b"):
        registry.generator_for("arm_add_q15", None, "case")
    assert registry.generator_for("arm_add_q15", "b", "case").key == "b"
    assert registry.generator_for("arm_add_f32", None, "case").key == "a"
    with pytest.raises(PackError, match="no generator claims arm_mult_q15"):
        registry.generator_for("arm_mult_q15", None, "case")
    with pytest.raises(PackError, match="does not claim arm_mult_q15"):
        registry.generator_for("arm_mult_q15", "a", "case")
    with pytest.raises(PackError, match="no generator 'c'"):
        registry.generator_for("arm_add_q15", "c", "case")


def test_a_pack_file_change_changes_its_digest(tmp_path) -> None:
    profile = load_profile(_library(tmp_path))
    before = pack_loader.files_digest(pack_loader.load_pack(profile))
    (tmp_path / "Testing" / "hct" / "rules.py").write_text("X = 1\n")
    after = pack_loader.files_digest(pack_loader.load_pack(profile))
    assert before != after
    assert pack_loader.files_digest(None) == ""


# --- descriptors ---------------------------------------------------------------------------------

def test_descriptors_load_as_cases(monkeypatch, tmp_path) -> None:
    cases, pack = _cases(monkeypatch, tmp_path, {
        "BasicMath/add.yaml": _doc("add_q15", params={"n": 9}) + "---\n" + _doc("add_f32", entry="arm_add_f32")})
    assert pack is not None
    by_name = {c["name"]: c for c in cases}
    assert by_name["add_q15"]["operator"] == "basic.binary"
    assert by_name["add_q15"]["suite"] == "int"
    assert by_name["add_q15"]["resolved_tensor_dtypes"] == {"input": "S16", "output": "S16"}
    assert by_name["add_q15"]["resolved_comparison"] == {"mode": "exact_int"}
    assert by_name["add_q15"]["_family"] == "BasicMath" and by_name["add_q15"]["_pack"]
    assert by_name["add_f32"]["suite"] == "float"
    assert by_name["add_f32"]["resolved_comparison"]["mode"] == "float"


def test_an_empty_pack_has_no_cases(monkeypatch, tmp_path) -> None:
    cases, pack = _cases(monkeypatch, tmp_path, {})
    assert cases == [] and pack is not None


@pytest.mark.parametrize("files, said", [
    ({"BasicMath/a.yaml": "schema: hct/descriptor/2\nname: a\nentry: arm_add_q15\n"}, "is not hct/descriptor/1"),
    ({"BasicMath/a.yaml": _doc("a") + "bogus: 1\n"}, "Additional properties"),
    ({"BasicMath/a.yaml": _doc("Bad-Name")}, "does not match"),
    ({"BasicMath/a.yaml": _doc("a", entry="arm_nope_q15")}, "arm_nope_q15 is not in the library's contract"),
    ({"BasicMath/a.yaml": _doc("a", entry="arm_offset_q15")}, "no generator claims arm_offset_q15"),
    ({"BasicMath/a.yaml": _doc("a", generator="other")}, "no generator 'other'"),
    ({"BasicMath/a.yaml": _doc("a"), "Other/b.yaml": _doc("a")}, "case a is also defined in"),
    ({"a.yaml": _doc("a")}, "must be <Family>/<stem>.yaml"),
    ({"BasicMath/a.yaml": _doc("a", entry="arm_max_q15")}, "has 2 output pointers"),
    ({"BasicMath/a.yaml": _doc("a", entry="arm_add_f64")}, r"pSrcA is const float64_t \*, which pack cases do not support"),
    ({"BasicMath/a.yaml": _doc("a", comparison={"tolerance": 2})}, "needs a comparison reason"),
    ({"BasicMath/a.yaml": _doc("a", comparison={"atol": 0.1})}, "apply to float outputs"),
    ({"BasicMath/a.yaml": _doc("a", entry="arm_add_f32", comparison={"atol": 0.1})}, "wider than the FP32 default"),
    ({"BasicMath/a.yaml": _doc("a", entry="arm_add_f32", comparison={"tolerance": 1, "reason": "x"})},
     "applies to integer outputs"),
    ({"BasicMath/a.yaml": _doc("a", expected_status="DSP_NOPE")}, "DSP_NOPE is not one of the profile's codes"),
    ({"BasicMath/a.yaml": "name: a\noperator: Add\n"}, "operator descriptors are ns-cmsis-nn's"),
    ({"BasicMath/a.yaml": _doc("b") + "---\nname: a\noperator: Add\n"}, "document 2 is not hct/descriptor/1"),
    ({"BasicMath/a.yaml": "a: [\n"}, "not valid YAML"),
    ({"BasicMath/a.yaml": "name: a\nname: b\n"}, "not valid YAML"),
])
def test_a_bad_descriptor_is_refused(monkeypatch, tmp_path, files, said) -> None:
    with pytest.raises(PackError, match=said):
        _cases(monkeypatch, tmp_path, files)


def test_every_problem_is_reported_together(monkeypatch, tmp_path) -> None:
    with pytest.raises(PackError, match="2 problem") as caught:
        _cases(monkeypatch, tmp_path, {"A/a.yaml": _doc("a", entry="arm_nope"),
                                       "B/b.yaml": _doc("b", expected_status="DSP_NOPE")})
    assert "arm_nope" in str(caught.value) and "DSP_NOPE" in str(caught.value)


def test_a_tightened_float_bound_and_a_reasoned_integer_tolerance_are_kept(monkeypatch, tmp_path) -> None:
    cases, _ = _cases(monkeypatch, tmp_path, {"B/a.yaml": _doc("a", comparison={"tolerance": 1, "reason": "rounding"})
                                              + "---\n" + _doc("f", entry="arm_add_f32") + "comparison: {atol: 1.0e-7}\n"})
    by_name = {c["name"]: c for c in cases}
    assert by_name["a"]["resolved_comparison"] == {"mode": "tolerant_int", "tolerance": 1}
    assert by_name["f"]["resolved_comparison"]["atol"] == 1e-7


def test_another_library_cannot_use_hcts_operator_descriptors(monkeypatch, tmp_path) -> None:
    with pytest.raises(PackError, match="operator descriptors are ns-cmsis-nn's"):
        _cases(monkeypatch, tmp_path, {"BasicMath/a.yaml": "name: a\noperator: Add\n"}, modules={})


def test_another_library_without_a_pack_has_no_tests(monkeypatch, tmp_path) -> None:
    profile = _library(tmp_path)
    profile.write_text(profile.read_text()[:profile.read_text().index("pack:")])
    _use(monkeypatch, profile)
    from helia_core_tester.generation.io.descriptors import load_all_descriptors

    with pytest.raises(PackError, match="names no pack"):
        load_case_descriptors(load_all_descriptors)


# --- rendering -----------------------------------------------------------------------------------

def _render(monkeypatch, tmp_path, module: str = _BINARY, golden: str = "record", **doc_fields):
    from helia_core_tester.generation.test_ops import generate_test

    entry = doc_fields.pop("entry", "arm_add_q15")
    cases, _ = _cases(monkeypatch, tmp_path, {"BasicMath/a.yaml": _doc("case_a", entry=entry, **doc_fields)},
                      modules={"basic": module})
    out = tmp_path / "out"
    kwargs = {"golden": "record", "kernel_numerics": "bounded"} if golden == "record" else {}
    generate_test(cases[0], str(out), seed=5, cpu="cortex-m55", **kwargs)
    case_dir = out / "BasicMath" / "case_a"
    return case_dir, (case_dir / "case_a_pack.c").read_text(), (case_dir / "includes" / "case_a_pack.h").read_text()


def test_a_case_renders_the_entry_call_from_the_contract(monkeypatch, tmp_path) -> None:
    case_dir, source, header = _render(monkeypatch, tmp_path, params={"n": 9})
    assert "arm_add_q15(" in source and "9u /* blockSize */" in source
    assert "EXACT_INT" in source and "case_a_expected_output" in source
    assert "static const q15_t case_a_pSrcA[]" in header
    assert "__builtin_types_compatible_p(__typeof__(arm_add_q15)" in source
    assert not (case_dir / "case_a.reference.json").exists()
    assert hk.classify_case(case_dir, "cortex-m55", "m0").status == hk.STATUS_BLOCKING
    assert (case_dir / "case_a.golden_placeholder").is_file()


def test_a_case_with_its_own_golden_needs_no_record(monkeypatch, tmp_path) -> None:
    case_dir, _, header = _render(monkeypatch, tmp_path, golden="hct_ref", params={"n": 9, "golden": True})
    assert "0, 4, 8, 12" in header
    assert not (case_dir / "case_a.golden_placeholder").exists()


def test_a_case_without_a_golden_needs_record_mode(monkeypatch, tmp_path) -> None:
    with pytest.raises(PackError, match="--golden record"):
        _render(monkeypatch, tmp_path, golden="hct_ref")


def test_a_zero_length_case_keeps_one_element_and_poisons_it(monkeypatch, tmp_path) -> None:
    _, source, header = _render(monkeypatch, tmp_path, params={"n": 0})
    assert "0u /* blockSize */" in source
    assert "true /* poisoned" in source or "must not be written" in source or "poison" in source


def _module(body: str) -> str:
    return ("import numpy as np\nfrom helia_core_tester.pack import CaseSpec\nHCT_PACK_API = '1.0'\n"
            f"def build(case):\n{body}\n"
            "def register(pack):\n    pack.generator('basic.binary', build, entries=['arm_*'])\n")


_A = "    a = np.zeros(4, dtype=np.int16)\n"


@pytest.mark.parametrize("body, said", [
    (_A + "    return CaseSpec(arrays={'pSrcA': a}, output='pDst', output_count=4, scalars={'blockSize': 4})",
     "must be exactly the input pointers"),
    (_A + "    return CaseSpec(arrays={'pSrcA': a, 'pSrcB': a, 'x': a}, output='pDst', output_count=4, "
     "scalars={'blockSize': 4})", "must be exactly the input pointers"),
    (_A + "    return CaseSpec(arrays={'pSrcA': a.astype(np.int32), 'pSrcB': a}, output='pDst', output_count=4, "
     "scalars={'blockSize': 4})", "gave int32"),
    (_A + "    return CaseSpec(arrays={'pSrcA': a, 'pSrcB': a}, output='pDst', output_count=4, scalars={})",
     "must be exactly"),
    (_A + "    return CaseSpec(arrays={'pSrcA': a, 'pSrcB': a}, output='pDst', output_count=4, "
     "scalars={'blockSize': -1})", "outside uint32_t"),
    (_A + "    return CaseSpec(arrays={'pSrcA': a, 'pSrcB': a}, output='pDst', output_count=4, "
     "scalars={'blockSize': 4.0})", "takes an integer"),
    (_A + "    return CaseSpec(arrays={'pSrcA': a, 'pSrcB': a}, output='pDst', output_count=4, "
     "scalars={'blockSize': True})", "takes an integer"),
    (_A + "    return CaseSpec(arrays={'pSrcA': a, 'pSrcB': a}, output='pSrcA', output_count=4, "
     "scalars={'blockSize': 4})", "one output pointer"),
    (_A + "    return CaseSpec(arrays={'pSrcA': a, 'pSrcB': a}, output='pDst', output_count=-1, "
     "scalars={'blockSize': 4})", "not a non-negative integer"),
    (_A + "    return CaseSpec(arrays={'pSrcA': a, 'pSrcB': a}, output='pDst', output_count=4, "
     "scalars={'blockSize': 4}, golden=np.zeros(3, dtype=np.int16))", "golden has 3 elements"),
    (_A + "    return CaseSpec(arrays={'pSrcA': a, 'pSrcB': a}, output='pDst', output_count=4, "
     "scalars={'blockSize': 4}, golden=np.zeros(4, dtype=np.int32))", "golden is int32"),
    ("    raise ValueError('bad draw')", "generator basic.binary \\(basic\\) failed: ValueError: bad draw"),
    ("    return {'arrays': {}}", "returned dict, not a CaseSpec"),
])
def test_a_bad_case_spec_is_refused_before_anything_is_written(monkeypatch, tmp_path, body, said) -> None:
    with pytest.raises(PackError, match=said):
        _render(monkeypatch, tmp_path, module=_module(body))
    assert not (tmp_path / "out" / "BasicMath" / "case_a" / "case_a_pack.c").exists()


@pytest.mark.parametrize("value, said", [(float("nan"), "non-finite"), (1e40, "overflows"), ("x", "takes a number")])
def test_a_bad_float_scalar_is_refused(value, said) -> None:
    from helia_core_tester.generation.pack_case import scalar_literal
    from helia_core_tester.pack import Param

    with pytest.raises(PackError, match=said):
        scalar_literal(Param("alpha", "float32_t", "in", False, "float32_t"), value, "case")
    assert scalar_literal(Param("alpha", "float32_t", "in", False, "float32_t"), 0.5, "case") == "0.5f"
    assert scalar_literal(Param("o", "q15_t", "in", False, "q15_t"), -3, "case") == "-3"


# --- the drift check -----------------------------------------------------------------------------

_SOURCE = _BINARY + '''

def source(contracts):
    return {"BasicMath/add.yaml": [{"schema": "hct/descriptor/1", "name": f"add_{n}", "entry": "arm_add_q15",
                                    "params": {"n": n}} for n in (1, 8, 9)]}


def register(pack):
    pack.generator("basic.binary", binary, entries=["arm_add_q15"])
    pack.descriptor_source(source)
'''


def _cli(*args):
    from helia_core_tester.cli import app

    return CliRunner().invoke(app, ["pack", *args])


def test_check_finds_drift_and_sync_fixes_it(monkeypatch, tmp_path) -> None:
    profile = _library(tmp_path, modules={"basic": _SOURCE}, descriptors={"Old/stale.yaml": _doc("stale")})
    library = str(profile.parent)
    result = _cli("check", "--library", library)
    assert result.exit_code == pack_cli.EXIT_DRIFT
    assert "BasicMath/add.yaml: missing" in result.output and "Old/stale.yaml: not generated" in result.output
    result = _cli("sync", "--library", library)
    assert result.exit_code == 0, result.output
    assert not (profile.parent / "descriptors" / "Old" / "stale.yaml").exists()
    result = _cli("check", "--library", library)
    assert result.exit_code == 0 and "3 case(s) in 1 file(s)" in result.output
    added = profile.parent / "descriptors" / "BasicMath" / "add.yaml"
    added.write_text(added.read_text().replace("n: 9", "n: 10"))
    result = _cli("check", "--library", library)
    assert result.exit_code == pack_cli.EXIT_DRIFT and "BasicMath/add.yaml: changed" in result.output


def test_a_nondeterministic_source_is_refused(tmp_path) -> None:
    text = _SOURCE.replace('for n in (1, 8, 9)]}', 'for n in (1, 8, int(__import__("time").perf_counter_ns()))]}')
    profile = _library(tmp_path, modules={"basic": text})
    result = _cli("check", "--library", str(profile.parent))
    assert result.exit_code == pack_cli.EXIT_UNAVAILABLE and "deterministic" in result.output


@pytest.mark.parametrize("rel", ["add.yaml", "../x/add.yaml", "A/b/add.yaml", "A/add.json"])
def test_a_source_must_name_family_files(tmp_path, rel) -> None:
    text = _SOURCE.replace('"BasicMath/add.yaml"', repr(rel))
    profile = _library(tmp_path, modules={"basic": text})
    result = _cli("check", "--library", str(profile.parent))
    assert result.exit_code == pack_cli.EXIT_UNAVAILABLE and "<Family>/<stem>.yaml" in result.output


def test_check_reports_a_bad_descriptor(tmp_path) -> None:
    profile = _library(tmp_path, descriptors={"A/a.yaml": _doc("a", entry="arm_nope")})
    result = _cli("check", "--library", str(profile.parent))
    assert result.exit_code == pack_cli.EXIT_DRIFT and "arm_nope is not in the library's contract" in result.output


def test_check_needs_a_pack(tmp_path) -> None:
    profile = _library(tmp_path)
    profile.write_text(profile.read_text()[:profile.read_text().index("pack:")])
    result = _cli("check", "--library", str(profile.parent))
    assert result.exit_code == pack_cli.EXIT_UNAVAILABLE and "must name pack" in result.output


# --- ns-cmsis-nn from a pack directory -----------------------------------------------------------

def test_an_nn_family_loads_from_a_pack_directory_as_it_does_built_in(monkeypatch, tmp_path) -> None:
    import shutil

    from helia_core_tester.core.discovery import find_descriptors_dir
    from helia_core_tester.generation.io.descriptors import load_all_descriptors
    from helia_core_tester.tests.test_host_reference import _fake_tree

    built_in = Path(find_descriptors_dir())
    family = "TileFunctions"
    tree = _fake_tree(tmp_path / "nn")
    pack_dir = tree / "Tests" / "hct"
    shutil.copytree(built_in / family, pack_dir / "descriptors" / family)
    (pack_dir / "hct-library.yaml").write_text(
        "schema: hct/library-profile/1\nlibrary: ns-cmsis-nn\nsource_root: ../..\n"
        "pack:\n  api: \"1.0\"\n  descriptors: descriptors\n")
    _use(monkeypatch, pack_dir / "hct-library.yaml")
    from_pack, pack = load_case_descriptors(load_all_descriptors)
    assert pack is not None and pack.registry.generators == {}
    expected = [d for d in load_all_descriptors(str(built_in)) if d["_source_family"] == family]
    assert from_pack and from_pack == expected


def test_template_contracts_follow_a_change_of_library(monkeypatch, tmp_path) -> None:
    from helia_core_tester.contract import render

    roots = []
    real = render.load_current_contracts

    def counting():
        roots.append(os.environ["CMSIS_NN_ROOT"])
        return real()

    monkeypatch.setattr(render, "load_current_contracts", counting)
    takes = render.contract_globals()["contract_takes"]
    first, second = _library(tmp_path / "a"), _library(tmp_path / "b")
    _use(monkeypatch, first)
    assert takes("arm_add_q15", "pSrcA") and takes("arm_add_q15", "pSrcB")
    _use(monkeypatch, second)
    assert takes("arm_add_q15", "pSrcA")
    _use(monkeypatch, first)
    assert takes("arm_add_q15", "pSrcA")
    assert [Path(r).name for r in roots] == ["a", "b"]


# --- review: check agrees with generation, and nothing leaks a raw exception ---------------------

_CLAIM_ALL = _BINARY.replace('"arm_max_q15"]', '"arm_max_q15", "arm_state_q15", "arm_mix_f32", "arm_sized_q15"]')


@pytest.mark.parametrize("entry, said", [
    ("arm_state_q15", "pState is an in,out pointer, which pack cases do not support yet"),
    ("arm_mix_f32", r"pSrc is const float64_t \*, which pack cases do not support yet"),
    ("arm_sized_q15", "blockSize is size_t, which pack scalars do not support yet"),
])
def test_check_refuses_what_generation_cannot_call(monkeypatch, tmp_path, entry, said) -> None:
    with pytest.raises(PackError, match=said):
        _cases(monkeypatch, tmp_path, {"BasicMath/a.yaml": _doc("a", entry=entry)}, modules={"basic": _CLAIM_ALL})


@pytest.mark.parametrize("comparison, said", [
    ({"atol": float("nan")}, "must be a finite number >= 0"),
    ({"rtol": float("inf")}, "must be a finite number >= 0"),
    ({"atol": -1.0e-9}, "must be a finite number >= 0|less than the minimum"),
])
def test_a_non_finite_or_negative_float_bound_is_refused(monkeypatch, tmp_path, comparison, said) -> None:
    text = _doc("a", entry="arm_add_f32").replace("schema:", "comparison: " + json.dumps(comparison) + "\nschema:")
    text = text.replace("NaN", ".nan").replace("Infinity", ".inf").replace("-1e-09", "-1.0e-9")
    with pytest.raises(PackError, match=said):
        _cases(monkeypatch, tmp_path, {"BasicMath/a.yaml": text})


@pytest.mark.parametrize("files, said", [
    ({"BasicMath/a.yaml": 'schema: hct/descriptor/1\nname: "a\\n"\nentry: arm_add_q15\n'}, "must match|does not match"),
    ({"BasicMath/a.yaml": 'schema: hct/descriptor/1\nname: a\nentry: "arm_add_q15\\n"\n'}, "must match|does not match"),
    ({"BasicMath/a.yml": _doc("a")}, "not .yml"),
    ({"BasicMath/a.yaml": "[" * 5000 + "]" * 5000}, "nested too deeply|not valid YAML|is not hct/descriptor/1"),
])
def test_odd_descriptor_files_are_refused(monkeypatch, tmp_path, files, said) -> None:
    with pytest.raises(PackError, match=said):
        _cases(monkeypatch, tmp_path, files)


def test_a_deeply_nested_file_is_a_pack_error(tmp_path, monkeypatch) -> None:
    from helia_core_tester.pack import descriptors as descriptors_mod

    path = tmp_path / "deep.yaml"
    path.write_text("a: 1\n")

    def deep(*_args, **_kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(descriptors_mod.yaml, "load_all", deep)
    with pytest.raises(PackError, match="nested too deeply"):
        descriptors_mod._read_documents(path)


def test_the_digest_covers_the_contract_and_data_files(tmp_path) -> None:
    profile = load_profile(_library(tmp_path))
    digest = lambda: pack_loader.files_digest(pack_loader.load_pack(profile))  # noqa: E731
    first = digest()
    contract = tmp_path / "Testing" / "hct" / "kernel_contracts.json"
    contract.write_text(contract.read_text().replace('"pSrcB"', '"pSrcC"'))
    second = digest()
    (tmp_path / "Testing" / "hct" / "table.csv").write_text("1,2\n")
    assert len({first, second, digest()}) == 3


def test_loading_a_pack_writes_no_bytecode(tmp_path) -> None:
    helper = "HCT_PACK_API = '1.0'\nfrom . import rules\n\n\ndef register(pack):\n    pass\n"
    profile = load_profile(_library(tmp_path, modules={"basic": helper}))
    (tmp_path / "Testing" / "hct" / "rules.py").write_text("X = 1\n")
    pack_loader.load_pack(profile)
    assert not list((tmp_path / "Testing" / "hct").rglob("__pycache__"))


@pytest.mark.parametrize("text, said", [
    ("raise SystemExit(3)\n", "import failed: SystemExit"),
    ("HCT_PACK_API = '1.0'\n\n\ndef register(pack):\n    raise SystemExit(4)\n", "register\\(\\) failed: SystemExit"),
])
def test_a_module_that_exits_is_a_pack_error(tmp_path, text, said) -> None:
    profile = load_profile(_library(tmp_path, modules={"basic": text}))
    with pytest.raises(PackError, match=said):
        pack_loader.load_pack(profile)


def test_a_huge_integer_float_scalar_is_refused() -> None:
    from helia_core_tester.generation.pack_case import scalar_literal
    from helia_core_tester.pack import Param

    with pytest.raises(PackError, match="overflows float32_t"):
        scalar_literal(Param("alpha", "float32_t", "in", False, "float32_t"), 10 ** 400, "case")


@pytest.mark.parametrize("key", ["entry_sizer", "autovectorize_declines"])
def test_a_case_the_manifest_gives_scratch_or_declines_is_refused(monkeypatch, tmp_path, key) -> None:
    from helia_core_tester.generation.pack_case import PackCaseGenerator

    cases, _ = _cases(monkeypatch, tmp_path, {"BasicMath/a.yaml": _doc("case_a")})
    desc = {**cases[0], key: "arm_add_q15_get_buffer_size" if key == "entry_sizer" else True}
    with pytest.raises(PackError, match=f"gives arm_add_q15 {key}, which pack cases do not support yet"):
        PackCaseGenerator(desc).generate_c_files(tmp_path / "out")


@pytest.mark.parametrize("produced, said", [
    ('{"BasicMath/add.yaml": [{"schema": "hct/descriptor/1", "name": "a", "entry": "arm_add_q15", '
     '"params": {"n": __import__("numpy").int64(3)}}]}', "YAML cannot write"),
    ('{"BasicMath/add.yaml": {"schema": "hct/descriptor/1"}}', "must map to a list of descriptor mappings"),
])
def test_a_source_with_values_yaml_cannot_write_is_refused(tmp_path, produced, said) -> None:
    text = _BINARY + f"\n\ndef source(contracts):\n    return {produced}\n\n\n" \
        "def register(pack):\n    pack.generator('basic.binary', binary, entries=['arm_add_q15'])\n" \
        "    pack.descriptor_source(source)\n"
    profile = _library(tmp_path, modules={"basic": text})
    result = _cli("check", "--library", str(profile.parent))
    assert result.exit_code == pack_cli.EXIT_UNAVAILABLE and said in result.output, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_check_reports_an_unreadable_committed_descriptor(tmp_path) -> None:
    profile = _library(tmp_path)
    bad = profile.parent / "descriptors" / "A" / "a.yaml"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"name: \xff\n")
    result = _cli("check", "--library", str(profile.parent))
    assert result.exit_code == pack_cli.EXIT_UNAVAILABLE and "not UTF-8 text" in result.output
    assert isinstance(result.exception, SystemExit)


def test_check_reports_a_broken_contract(tmp_path) -> None:
    profile = _library(tmp_path)
    (profile.parent / "kernel_contracts.json").write_text("{not json")
    result = _cli("check", "--library", str(profile.parent))
    assert result.exit_code == pack_cli.EXIT_UNAVAILABLE and "kernel_contracts.json" in result.output, result.output
    assert isinstance(result.exception, SystemExit)


def test_check_accepts_an_nn_pack_of_operator_descriptors(monkeypatch, tmp_path) -> None:
    import shutil

    from helia_core_tester.core.discovery import find_descriptors_dir
    from helia_core_tester.tests.test_host_reference import _fake_tree

    tree = _fake_tree(tmp_path / "nn")
    pack_dir = tree / "Tests" / "hct"
    shutil.copytree(Path(find_descriptors_dir()) / "TileFunctions", pack_dir / "descriptors" / "TileFunctions")
    (pack_dir / "hct-library.yaml").write_text(
        "schema: hct/library-profile/1\nlibrary: ns-cmsis-nn\nsource_root: ../..\n"
        "pack:\n  api: \"1.0\"\n  descriptors: descriptors\n")
    result = _cli("check", "--library", str(pack_dir))
    assert result.exit_code == 0, result.output


# --- FVP reporting -------------------------------------------------------------------------------

def test_the_fvp_report_counts_a_pack_librarys_cases(monkeypatch, tmp_path) -> None:
    from helia_core_tester.fvp.reporting import descriptor_tracker
    from helia_core_tester.reporting.models import TestStatus

    profile = _library(tmp_path / "lib", descriptors={"BasicMath/add.yaml": _doc("add_q15", params={"n": 9})})
    _use(monkeypatch, profile)
    tracker = descriptor_tracker()
    cases = tracker.load_all_descriptors()
    assert set(cases) == {"add_q15"}
    assert tracker.get_descriptor_path("add_q15") == profile.parent / "descriptors" / "BasicMath" / "add.yaml"
    build, generated = tmp_path / "build", tmp_path / "gen"
    assert tracker.elf_path_for("add_q15", build) == build / "tests" / "BasicMath" / "add_q15.elf"
    includes = generated / "BasicMath" / "add_q15" / "includes"
    includes.mkdir(parents=True)
    assert tracker.determine_descriptor_status("add_q15", None, build, generated)[0] == TestStatus.GENERATION_FAILED
    (includes / "add_q15_pack.h").write_text("")
    assert tracker.determine_descriptor_status("add_q15", None, build, generated)[0] == TestStatus.BUILD_FAILED


def test_the_fvp_report_of_hcts_own_library_is_unchanged(monkeypatch) -> None:
    from helia_core_tester.core.discovery import find_descriptors_dir
    from helia_core_tester.fvp.reporting import descriptor_tracker

    tracker = descriptor_tracker()
    assert tracker.descriptors_dir == find_descriptors_dir()
    assert tracker._loader is not None and len(tracker.load_all_descriptors()) > 0


def test_the_fvp_report_refuses_a_broken_pack_rather_than_counting_nothing(monkeypatch, tmp_path) -> None:
    from helia_core_tester.fvp.reporting import descriptor_tracker

    profile = _library(tmp_path, descriptors={"BasicMath/add.yaml": _doc("add_q15", entry="arm_nope_q15")})
    _use(monkeypatch, profile)
    with pytest.raises(PackError, match="arm_nope_q15 is not in the library's contract"):
        descriptor_tracker().load_all_descriptors()


def test_the_fvp_report_of_an_nn_pack_reads_the_packs_descriptors(monkeypatch, tmp_path) -> None:
    import shutil

    from helia_core_tester.core.discovery import find_descriptors_dir
    from helia_core_tester.fvp.reporting import descriptor_tracker
    from helia_core_tester.tests.test_host_reference import _fake_tree

    family = "TileFunctions"
    pack_dir = _fake_tree(tmp_path / "nn") / "Tests" / "hct"
    shutil.copytree(Path(find_descriptors_dir()) / family, pack_dir / "descriptors" / family)
    (pack_dir / "hct-library.yaml").write_text(
        "schema: hct/library-profile/1\nlibrary: ns-cmsis-nn\nsource_root: ../..\n"
        "pack:\n  api: \"1.0\"\n  descriptors: descriptors\n")
    _use(monkeypatch, pack_dir / "hct-library.yaml")
    tracker = descriptor_tracker()
    cases = tracker.load_all_descriptors()
    assert tracker.descriptors_dir == pack_dir / "descriptors"
    assert cases and {c["_source_family"] for c in cases.values()} == {family}
    name = next(iter(cases))
    assert tracker.get_descriptor_path(name).is_relative_to(pack_dir / "descriptors" / family)


# --- review 2: per-input types, signatures, exits, file names, sync, and one module instance -----

_GATHER = _module(
    "    a = np.arange(4, dtype=np.int16)\n    i = np.arange(4, dtype=np.int32)\n"
    "    return CaseSpec(arrays={'pSrc': a, 'pIndex': i}, output='pDst', output_count=4, scalars={'blockSize': 4})")


def test_each_input_is_passed_as_its_own_type(monkeypatch, tmp_path) -> None:
    _, source, header = _render(monkeypatch, tmp_path, module=_GATHER, entry="arm_gather_q15")
    assert "const q15_t* __restrict arg_pSrc" in source
    assert "const int32_t* __restrict arg_pIndex" in source
    assert "static const int32_t case_a_pIndex[]" in header


def test_a_pointer_to_pointers_is_refused(monkeypatch, tmp_path) -> None:
    with pytest.raises(PackError, match="a pointer to pointers"):
        _cases(monkeypatch, tmp_path, {"BasicMath/a.yaml": _doc("a", entry="arm_rows_q15")},
               modules={"basic": _module("    return None")})


def test_an_array_parameter_is_refused(monkeypatch, tmp_path) -> None:
    profile = _library(tmp_path, descriptors={"BasicMath/a.yaml": _doc("a", entry="arm_offset_q15")},
                       modules={"basic": _module("    return None")})
    contract = profile.parent / "kernel_contracts.json"
    data = json.loads(contract.read_text())
    record = next(r for r in data["functions"] if r["name"] == "arm_offset_q15")
    record["params"][1]["extent"] = "[4]"
    contract.write_text(json.dumps(data))
    _use(monkeypatch, profile)
    from helia_core_tester.generation.io.descriptors import load_all_descriptors

    with pytest.raises(PackError, match=r"offset is an array parameter \(q15_t\[4\]\)"):
        load_case_descriptors(load_all_descriptors)


def test_a_generator_that_exits_is_a_pack_error(monkeypatch, tmp_path) -> None:
    with pytest.raises(PackError, match="generator basic.binary \\(basic\\) failed: SystemExit"):
        _render(monkeypatch, tmp_path, module=_module("    raise SystemExit(0)"))


def test_a_source_that_exits_does_not_pass_check(tmp_path) -> None:
    text = _SOURCE.replace("def source(contracts):\n", "def source(contracts):\n    raise SystemExit(0)\n")
    profile = _library(tmp_path, modules={"basic": text})
    result = _cli("check", "--library", str(profile.parent))
    assert result.exit_code == pack_cli.EXIT_UNAVAILABLE and "descriptor_source failed: SystemExit" in result.output


@pytest.mark.parametrize("rel", ["./BasicMath/add.yaml", "BasicMath//add.yaml"])
def test_a_source_must_spell_its_file_names_plainly(tmp_path, rel) -> None:
    text = _SOURCE.replace('"BasicMath/add.yaml"', repr(rel))
    profile = _library(tmp_path, modules={"basic": text})
    result = _cli("check", "--library", str(profile.parent))
    assert result.exit_code == pack_cli.EXIT_UNAVAILABLE and "must be written as 'BasicMath/add.yaml'" in result.output


@pytest.mark.parametrize("link", ["family", "file"])
def test_sync_does_not_write_through_a_symlink(tmp_path, link) -> None:
    profile = _library(tmp_path / "lib", modules={"basic": _SOURCE})
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "add.yaml").write_text("keep\n")
    family = profile.parent / "descriptors" / "BasicMath"
    if link == "family":
        family.symlink_to(outside, target_is_directory=True)
    else:
        family.mkdir()
        (family / "add.yaml").symlink_to(outside / "add.yaml")
    result = _cli("sync", "--library", str(profile.parent))
    assert result.exit_code == pack_cli.EXIT_UNAVAILABLE and "is a symlink" in result.output
    assert (outside / "add.yaml").read_text() == "keep\n"


def test_check_refuses_a_status_a_void_entry_cannot_return(monkeypatch, tmp_path) -> None:
    with pytest.raises(PackError, match="returns void, so expected_status DSP_ARG can never be observed"):
        _cases(monkeypatch, tmp_path, {"BasicMath/a.yaml": _doc("a", expected_status="DSP_ARG")})


def test_check_refuses_an_unknown_capability(monkeypatch, tmp_path) -> None:
    with pytest.raises(PackError, match="unknown required capabilities \\['mve_fp64'\\]"):
        _cases(monkeypatch, tmp_path, {"BasicMath/a.yaml": _doc("a", required_capabilities=["mve_fp64"])})


def test_check_refuses_a_decline_the_manifest_gives(monkeypatch, tmp_path) -> None:
    from helia_core_tester.generation import entry as entry_mod

    monkeypatch.setattr(entry_mod, "apply_entry_manifest", lambda desc, contracts=None: {**desc, "autovectorize_declines": True})
    with pytest.raises(PackError, match="manifest gives arm_add_q15 autovectorize_declines"):
        _cases(monkeypatch, tmp_path, {"BasicMath/a.yaml": _doc("a")})


def test_a_module_another_module_imported_is_not_run_twice(tmp_path) -> None:
    log = tmp_path / "imports.log"
    second = (f"open({str(log)!r}, 'a').write('second\\n')\nHCT_PACK_API = '1.0'\n"
              "def register(pack):\n    pack.generator('g2', len, entries=['arm_mult_*'])\n")
    first = ("from . import second\nHCT_PACK_API = '1.0'\n"
             "def register(pack):\n    pack.generator('g1', len, entries=['arm_add_*'])\n")
    pack = _load(tmp_path, {"first": first, "second": second})
    assert log.read_text() == "second\n"
    assert set(pack.registry.generators) == {"g1", "g2"}


def test_a_helper_runs_from_its_source_even_beside_stale_bytecode(monkeypatch, tmp_path) -> None:
    import importlib.util
    import py_compile

    profile = _library(tmp_path, modules={"basic": _module(
        "    from . import helper\n    a = np.full(4, helper.VALUE, dtype=np.int16)\n"
        "    return CaseSpec(arrays={'pSrcA': a, 'pSrcB': a}, output='pDst', output_count=4, scalars={'blockSize': 4})")})
    helper = profile.parent / "helper.py"
    helper.write_text("VALUE = 1\n")
    stamp = helper.stat().st_mtime_ns
    py_compile.compile(str(helper), cfile=importlib.util.cache_from_source(str(helper)), doraise=True)
    helper.write_text("VALUE = 2\n")
    os.utime(helper, ns=(stamp, stamp))
    _use(monkeypatch, profile)
    from helia_core_tester.generation.io.descriptors import load_all_descriptors
    from helia_core_tester.generation.test_ops import generate_test

    (profile.parent / "descriptors" / "BasicMath").mkdir(parents=True, exist_ok=True)
    (profile.parent / "descriptors" / "BasicMath" / "a.yaml").write_text(_doc("case_a"))
    cases, _ = load_case_descriptors(load_all_descriptors)
    generate_test(cases[0], str(tmp_path / "out"), seed=5, cpu="cortex-m55", golden="record", kernel_numerics="bounded")
    header = (tmp_path / "out" / "BasicMath" / "case_a" / "includes" / "case_a_pack.h").read_text()
    assert "2, 2, 2, 2" in header
