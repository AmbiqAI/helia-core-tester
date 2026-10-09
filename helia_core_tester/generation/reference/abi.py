"""The hct_ref C ABI, from its single source of truth (reference/spec/entries.yaml).

Renders the C header and builds the matching ctypes types, so the two sides
cannot drift: a spec change regenerates both, and a stale header fails
tests/test_reference_abi.py.
"""

from __future__ import annotations

import argparse
import ctypes
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import yaml

REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "reference"
SPEC_PATH = REFERENCE_ROOT / "spec" / "entries.yaml"
HEADER_PATH = REFERENCE_ROOT / "include" / "hct_ref_abi.h"

FIELD_TYPES: Dict[str, Tuple[str, type]] = {
    "int32": ("int32_t", ctypes.c_int32),
    "int64": ("int64_t", ctypes.c_int64),
    "float32": ("float", ctypes.c_float),
    "float64": ("double", ctypes.c_double),
}

TENSOR_NUMPY: Dict[str, np.dtype] = {
    "int8": np.dtype(np.int8),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "bool": np.dtype(np.uint8),
}

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SpecError(ValueError):
    """The ABI spec is malformed."""


@dataclass(frozen=True)
class StructSpec:
    name: str
    fields: Tuple[Tuple[str, str], ...]
    doc: str = ""


@dataclass(frozen=True)
class KernelSpec:
    name: str
    params: str
    inputs: Tuple[Tuple[str, str], ...]
    outputs: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class PrepareSpec:
    name: str
    in_struct: str
    out_struct: str


@dataclass(frozen=True)
class Spec:
    abi_version: int
    max_rank: int
    dtypes: Mapping[str, int]
    status: Mapping[str, int]
    activations: Mapping[str, int]
    structs: Mapping[str, StructSpec]
    kernels: Mapping[str, KernelSpec]
    prepare: Mapping[str, PrepareSpec]
    ctypes_structs: Dict[str, type] = field(default_factory=dict, compare=False, repr=False)


def _require_ident(value: object, what: str) -> str:
    if not isinstance(value, str) or not _IDENT.match(value):
        raise SpecError(f"{what}: {value!r} is not a C identifier")
    return value


def _enum(raw: object, what: str) -> Dict[str, int]:
    if not isinstance(raw, dict) or not raw:
        raise SpecError(f"{what} must be a non-empty mapping")
    out = {_require_ident(k, what): v for k, v in raw.items()}
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in out.values()):
        raise SpecError(f"{what} values must be non-negative integers")
    if len(set(out.values())) != len(out):
        raise SpecError(f"{what} values must be distinct")
    return out


def parse_spec(raw: Mapping) -> Spec:
    """Validate a loaded spec document and turn it into a Spec."""
    if not isinstance(raw, Mapping):
        raise SpecError("spec must be a mapping")
    version = raw.get("abi_version")
    max_rank = raw.get("max_rank")
    if not isinstance(version, int) or version < 1:
        raise SpecError("abi_version must be a positive integer")
    if not isinstance(max_rank, int) or not 1 <= max_rank <= 16:
        raise SpecError("max_rank must be an integer in [1, 16]")
    dtypes = _enum(raw.get("dtypes"), "dtypes")
    if set(dtypes) != set(TENSOR_NUMPY):
        raise SpecError(f"dtypes must be exactly {sorted(TENSOR_NUMPY)}")
    status = _enum(raw.get("status"), "status")
    if status.get("OK") != 0:
        raise SpecError("status OK must be 0")
    activations = _enum(raw.get("activations"), "activations")

    structs: Dict[str, StructSpec] = {}
    for name, body in (raw.get("structs") or {}).items():
        _require_ident(name, "struct")
        fields = body.get("fields") if isinstance(body, dict) else None
        if not fields:
            raise SpecError(f"struct {name} has no fields")
        parsed = []
        for entry in fields:
            if not (isinstance(entry, list) and len(entry) == 2):
                raise SpecError(f"struct {name}: field {entry!r} must be [name, type]")
            fname, ftype = _require_ident(entry[0], f"{name} field"), entry[1]
            if ftype not in FIELD_TYPES:
                raise SpecError(f"struct {name}: field {fname} has unknown type {ftype!r}")
            parsed.append((fname, ftype))
        if len({f for f, _ in parsed}) != len(parsed):
            raise SpecError(f"struct {name} repeats a field")
        structs[name] = StructSpec(name, tuple(parsed), str(body.get("doc", "")))

    def tensors(raw_t: object, what: str) -> Tuple[Tuple[str, str], ...]:
        if not isinstance(raw_t, dict) or not raw_t:
            raise SpecError(f"{what} must be a non-empty mapping")
        for tname, dtype in raw_t.items():
            _require_ident(tname, what)
            if dtype not in TENSOR_NUMPY:
                raise SpecError(f"{what}: tensor {tname} has unknown dtype {dtype!r}")
        return tuple(raw_t.items())

    kernels: Dict[str, KernelSpec] = {}
    for name, body in (raw.get("kernels") or {}).items():
        _require_ident(name, "kernel")
        params = body.get("params")
        if params not in structs:
            raise SpecError(f"kernel {name}: unknown params struct {params!r}")
        kernels[name] = KernelSpec(name, params, tensors(body.get("inputs"), f"{name} inputs"),
                                   tensors(body.get("outputs"), f"{name} outputs"))

    prepare: Dict[str, PrepareSpec] = {}
    for name, body in (raw.get("prepare") or {}).items():
        _require_ident(name, "prepare")
        if name in kernels:
            raise SpecError(f"{name} is both a kernel and a prepare entry")
        for key in ("in", "out"):
            if body.get(key) not in structs:
                raise SpecError(f"prepare {name}: unknown {key} struct {body.get(key)!r}")
        prepare[name] = PrepareSpec(name, body["in"], body["out"])

    return Spec(version, max_rank, dtypes, status, activations, structs, kernels, prepare)


@lru_cache(maxsize=None)
def load_spec(path: Path = SPEC_PATH) -> Spec:
    return parse_spec(yaml.safe_load(Path(path).read_text()))


def dtype_code(name: str) -> int:
    """The HctDtype value of tensor dtype `name` (int8, int16, ..., float16)."""
    dtypes = load_spec().dtypes
    if name not in dtypes:
        raise KeyError(f"unknown tensor dtype {name!r}; known: {sorted(dtypes)}")
    return dtypes[name]


def activation_code(name: str) -> int:
    """The HctActivation value of fused activation `name` (NONE, RELU, RELU6, RELU_N1_TO_1)."""
    acts = load_spec().activations
    key = str(name).upper()
    if key not in acts:
        raise KeyError(f"unknown fused activation {name!r}; known: {sorted(acts)}")
    return acts[key]


def _comment(text: str) -> str:
    return f"/* {text} */\n" if text else ""


def render_header(spec: Spec) -> str:
    """The C header for `spec`; byte-stable for a given spec."""
    out: List[str] = [
        "/*\n * SPDX-FileCopyrightText: 2026 Ambiq\n * SPDX-License-Identifier: Apache-2.0\n *\n"
        " * Generated from spec/entries.yaml by helia_core_tester.generation.reference.abi.\n"
        " * Do not edit; run `python -m helia_core_tester.generation.reference.abi --write`.\n */\n",
        "#ifndef HCT_REF_ABI_H\n#define HCT_REF_ABI_H\n\n#include <stdint.h>\n\n",
        "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n",
        "#if defined(__GNUC__)\n#pragma GCC visibility push(default)\n#endif\n\n",
        f"#define HCT_REF_ABI_VERSION {spec.abi_version}\n#define HCT_MAX_RANK {spec.max_rank}\n\n",
    ]
    for title, prefix, table in (("HctDtype", "HCT_", spec.dtypes), ("HctStatus", "HCT_", spec.status),
                                 ("HctActivation", "HCT_ACT_", spec.activations)):
        body = ",\n".join(f"    {prefix}{k.upper()} = {v}" for k, v in table.items())
        out.append(f"typedef enum\n{{\n{body}\n}} {title};\n\n")
    out.append(
        "/* A tensor: dtype is an HctDtype, dims[0..rank-1] row-major (NHWC for 4-D), data\n"
        " * points at rank-product elements (1 for rank 0). */\n"
        "typedef struct\n{\n    int32_t dtype;\n    int32_t rank;\n"
        "    int32_t dims[HCT_MAX_RANK];\n    void *data;\n} HctTensor;\n\n"
    )
    for s in spec.structs.values():
        fields = "".join(f"    {FIELD_TYPES[t][0]} {f};\n" for f, t in s.fields)
        out.append(f"{_comment(s.doc)}typedef struct\n{{\n{fields}}} {s.name};\n\n")
    out.append("int32_t hct_ref_abi_version(void);\n\n")
    for p in spec.prepare.values():
        out.append(f"int32_t hct_ref_{p.name}(const {p.in_struct} *in, {p.out_struct} *out);\n")
    out.append("\n")
    for k in spec.kernels.values():
        ins = ", ".join(f"{n}:{d}" for n, d in k.inputs)
        outs = ", ".join(f"{n}:{d}" for n, d in k.outputs)
        out.append(
            f"/* inputs [{ins}] -> outputs [{outs}] */\n"
            f"int32_t hct_ref_{k.name}(const {k.params} *params,\n"
            f"    const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs, int32_t num_outputs);\n"
        )
    out.append(
        "\n#if defined(__GNUC__)\n#pragma GCC visibility pop\n#endif\n\n"
        "#ifdef __cplusplus\n}\n#endif\n\n#endif /* HCT_REF_ABI_H */\n"
    )
    return "".join(out)


class HctTensor(ctypes.Structure):
    pass


def tensor_struct(max_rank: int) -> type:
    """HctTensor sized for `max_rank` (the spec's, in practice); defined once per process."""
    if getattr(HctTensor, "_fields_", None) is None:
        HctTensor._fields_ = [("dtype", ctypes.c_int32), ("rank", ctypes.c_int32),
                              ("dims", ctypes.c_int32 * max_rank), ("data", ctypes.c_void_p)]
        HctTensor.max_rank = max_rank
    elif HctTensor.max_rank != max_rank:
        raise SpecError(f"HctTensor already defined for max_rank {HctTensor.max_rank}, not {max_rank}")
    return HctTensor


def ctypes_struct(spec: Spec, name: str) -> type:
    """The ctypes Structure for spec struct `name` (one class per name)."""
    if name not in spec.structs:
        raise KeyError(f"unknown struct {name!r}")
    if name not in spec.ctypes_structs:
        s = spec.structs[name]
        spec.ctypes_structs[name] = type(name, (ctypes.Structure,),
                                         {"_fields_": [(f, FIELD_TYPES[t][1]) for f, t in s.fields]})
    return spec.ctypes_structs[name]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render the hct_ref ABI header from the spec.")
    parser.add_argument("--write", action="store_true", help="write include/hct_ref_abi.h")
    parser.add_argument("--check", action="store_true", help="exit 1 if the header is stale")
    args = parser.parse_args(argv)
    text = render_header(load_spec())
    if args.write:
        HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEADER_PATH.write_text(text)
        print(f"wrote {HEADER_PATH}")
    elif args.check:
        current = HEADER_PATH.read_text() if HEADER_PATH.exists() else ""
        if current != text:
            print(f"{HEADER_PATH} is stale; run with --write")
            return 1
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
