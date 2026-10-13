"""The ABI spec is the single source of truth: header freshness, spec validation, struct layout."""

from __future__ import annotations

import copy
import ctypes
import subprocess
from pathlib import Path

import pytest
import yaml

from helia_core_tester.generation.reference import abi
from helia_core_tester.utils.host_compiler import find_host_cc


def _raw() -> dict:
    return yaml.safe_load(abi.SPEC_PATH.read_text())


def test_committed_header_matches_the_spec() -> None:
    assert abi.HEADER_PATH.read_text() == abi.render_header(abi.load_spec()), (
        "run `python -m helia_core_tester.generation.reference.abi --write`"
    )
    assert abi.main(["--check"]) == 0


def test_render_is_deterministic() -> None:
    assert abi.render_header(abi.parse_spec(_raw())) == abi.render_header(abi.parse_spec(_raw()))


def _mutated(path, value):
    raw = _raw()
    node = raw
    for key in path[:-1]:
        node = node[key]
    if value is KeyError:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return raw


@pytest.mark.parametrize(
    "path, value, match",
    [
        (("abi_version",), 0, "abi_version"),
        (("max_rank",), 0, "max_rank"),
        (("dtypes", "int8"), KeyError, "dtypes must be exactly"),
        (("status", "OK"), 9, "OK must be 0"),
        (("status", "E_NULL"), 2, "distinct"),
        (("activations", "RELU"), -1, "non-negative"),
        (("structs", "HctAddParams", "fields"), [], "no fields"),
        (("structs", "HctAddParams", "fields"), [["left_shift", "int8"]], "unknown type"),
        (("structs", "HctAddParams", "fields"), [["a", "int32"], ["a", "int32"]], "repeats"),
        (("structs", "HctAddParams", "fields"), [["bad-name", "int32"]], "not a C identifier"),
        (("kernels", "add_s8", "params"), "Nope", "unknown params"),
        (("kernels", "add_s8", "inputs"), {"input1": "uint4"}, "unknown dtype"),
        (("kernels", "add_s8", "outputs"), {}, "non-empty"),
        (("prepare", "add_prepare", "out"), "Nope", "unknown out"),
    ],
)
def test_malformed_specs_are_rejected(path, value, match) -> None:
    with pytest.raises(abi.SpecError, match=match):
        abi.parse_spec(_mutated(path, value))


def test_an_entry_cannot_be_both_kernel_and_prepare() -> None:
    raw = _raw()
    raw["prepare"]["add_s8"] = copy.deepcopy(raw["prepare"]["add_prepare"])
    with pytest.raises(abi.SpecError, match="both a kernel and a prepare"):
        abi.parse_spec(raw)


def test_tensor_struct_is_defined_once() -> None:
    rank = abi.load_spec().max_rank
    assert abi.tensor_struct(rank) is abi.tensor_struct(rank)
    with pytest.raises(abi.SpecError, match="already defined"):
        abi.tensor_struct(rank + 1)


def test_spec_codes_are_looked_up_by_name() -> None:
    assert abi.dtype_code("int8") == 1 and abi.dtype_code("float16") == 6
    assert abi.activation_code("relu6") == abi.activation_code("RELU6")
    with pytest.raises(KeyError, match="unknown tensor dtype"):
        abi.dtype_code("uint8")
    with pytest.raises(KeyError, match="unknown fused activation"):
        abi.activation_code("TANH")


def test_ctypes_layouts_match_the_c_compiler(tmp_path: Path) -> None:
    """sizeof and every field offset, as the host C compiler lays them out."""
    spec = abi.load_spec()
    lines = ["#include <stddef.h>", "#include <stdio.h>", '#include "hct_ref_abi.h"', "int main(void){"]
    lines.append('printf("HctTensor %zu\\n", sizeof(HctTensor));')
    expected = {"HctTensor": ctypes.sizeof(abi.tensor_struct(spec.max_rank))}
    for name, s in spec.structs.items():
        cls = abi.ctypes_struct(spec, name)
        lines.append(f'printf("{name} %zu\\n", sizeof({name}));')
        expected[name] = ctypes.sizeof(cls)
        for field, _ in s.fields:
            lines.append(f'printf("{name}.{field} %zu\\n", offsetof({name}, {field}));')
            expected[f"{name}.{field}"] = getattr(cls, field).offset
    lines.append("return 0;}")
    src = tmp_path / "layout.c"
    src.write_text("\n".join(lines) + "\n")
    exe = tmp_path / "layout"
    build = subprocess.run([find_host_cc(), "-I", str(abi.HEADER_PATH.parent), str(src), "-o", str(exe)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    got = dict(line.split() for line in subprocess.run([str(exe)], capture_output=True, text=True).stdout.splitlines())
    assert {k: int(v) for k, v in got.items()} == expected
