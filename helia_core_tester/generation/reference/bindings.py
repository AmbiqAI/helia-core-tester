"""ctypes bindings to the host reference library, generated from the ABI spec.

Every kernel entry goes through `Bindings.run` and every prepare entry through
`Bindings.prepare`: there is no per-operator calling code. Arguments are checked
against the spec before the call (exact tensor names and dtypes, C-contiguous
arrays, exact struct fields); a non-OK status raises ReferenceKernelError.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from helia_core_tester.generation.reference.abi import TENSOR_NUMPY, Spec, ctypes_struct, load_spec, tensor_struct
from helia_core_tester.generation.reference.build import ReferenceLibrary, ensure_reference_library


class ReferenceKernelError(RuntimeError):
    """A reference entry returned a non-OK status."""

    def __init__(self, entry: str, code: int, status: str) -> None:
        super().__init__(f"hct_ref_{entry} returned {status} ({code})")
        self.entry = entry
        self.code = code
        self.status = status


class ReferenceAbiMismatch(RuntimeError):
    """The loaded library was built from a different ABI version than the spec."""


def struct_from_fields(spec: Spec, name: str, fields: Mapping[str, object]) -> ctypes.Structure:
    """Spec struct `name` filled from `fields`, which must name every field exactly once."""
    cls = ctypes_struct(spec, name)
    want = [f for f, _ in spec.structs[name].fields]
    missing = [f for f in want if f not in fields]
    unknown = [f for f in fields if f not in want]
    if missing or unknown:
        raise KeyError(f"{name}: missing fields {missing}, unknown fields {unknown}")
    values = []
    for fname, ftype in spec.structs[name].fields:
        value = fields[fname]
        if ftype.startswith("int"):
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise TypeError(f"{name}.{fname} must be an integer, got {value!r}")
            values.append(int(value))
        else:
            values.append(float(value))
    return cls(*values)


def struct_to_dict(spec: Spec, name: str, value: ctypes.Structure) -> Dict[str, object]:
    return {f: getattr(value, f) for f, _ in spec.structs[name].fields}


class Bindings:
    """The loaded library, with every spec entry bound."""

    def __init__(self, library: ReferenceLibrary, spec: Optional[Spec] = None) -> None:
        self.library = library
        self.spec = spec or load_spec()
        self._lib = ctypes.CDLL(str(library.path))
        self._lib.hct_ref_abi_version.restype = ctypes.c_int32
        self._lib.hct_ref_abi_version.argtypes = []
        version = int(self._lib.hct_ref_abi_version())
        if version != self.spec.abi_version:
            raise ReferenceAbiMismatch(f"library ABI {version}, spec ABI {self.spec.abi_version}")
        self._tensor = tensor_struct(self.spec.max_rank)
        self._status_names = {v: k for k, v in self.spec.status.items()}
        self._fns: Dict[str, object] = {}
        tensor_ptr = ctypes.POINTER(self._tensor)
        for name, k in self.spec.kernels.items():
            fn = getattr(self._lib, f"hct_ref_{name}")
            fn.argtypes = [ctypes.POINTER(ctypes_struct(self.spec, k.params)), tensor_ptr, ctypes.c_int32,
                           tensor_ptr, ctypes.c_int32]
            fn.restype = ctypes.c_int32
            self._fns[name] = fn
        for name, p in self.spec.prepare.items():
            fn = getattr(self._lib, f"hct_ref_{name}")
            fn.argtypes = [ctypes.POINTER(ctypes_struct(self.spec, p.in_struct)),
                           ctypes.POINTER(ctypes_struct(self.spec, p.out_struct))]
            fn.restype = ctypes.c_int32
            self._fns[name] = fn

    @property
    def key(self) -> str:
        return self.library.key

    def _check(self, entry: str, code: int) -> None:
        if code != 0:
            raise ReferenceKernelError(entry, code, self._status_names.get(code, "UNKNOWN"))

    def _tensor_for(self, array: np.ndarray, dtype_name: str, role: str) -> ctypes.Structure:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{role} must be a numpy array, got {type(array).__name__}")
        if array.dtype != TENSOR_NUMPY[dtype_name]:
            raise TypeError(f"{role} must be {dtype_name}, got {array.dtype}")
        if not array.flags.c_contiguous:
            raise TypeError(f"{role} must be C-contiguous")
        if array.ndim > self.spec.max_rank:
            raise ValueError(f"{role} has rank {array.ndim} > {self.spec.max_rank}")
        dims = list(array.shape) + [0] * (self.spec.max_rank - array.ndim)
        data = array.ctypes.data if array.size else None
        return self._tensor(self.spec.dtypes[dtype_name], array.ndim,
                            (ctypes.c_int32 * self.spec.max_rank)(*dims), data)

    def prepare(self, entry: str, fields: Mapping[str, object]) -> Dict[str, object]:
        """Run prepare entry `entry` on an input struct given as a field mapping."""
        if entry not in self.spec.prepare:
            raise KeyError(f"unknown prepare entry {entry!r}")
        p = self.spec.prepare[entry]
        inp = struct_from_fields(self.spec, p.in_struct, fields)
        out = ctypes_struct(self.spec, p.out_struct)()
        self._check(entry, self._fns[entry](ctypes.byref(inp), ctypes.byref(out)))
        return struct_to_dict(self.spec, p.out_struct, out)

    def run(
        self,
        entry: str,
        params: Mapping[str, object],
        inputs: Mapping[str, np.ndarray],
        output_shapes: Mapping[str, Sequence[int]],
    ) -> Dict[str, np.ndarray]:
        """Run kernel entry `entry`; returns the outputs by name."""
        if entry not in self.spec.kernels:
            raise KeyError(f"unknown kernel entry {entry!r}")
        k = self.spec.kernels[entry]
        want_in = [n for n, _ in k.inputs]
        want_out = [n for n, _ in k.outputs]
        if sorted(inputs) != sorted(want_in):
            raise KeyError(f"{entry}: inputs {sorted(inputs)}, expected {sorted(want_in)}")
        if sorted(output_shapes) != sorted(want_out):
            raise KeyError(f"{entry}: outputs {sorted(output_shapes)}, expected {sorted(want_out)}")
        struct = struct_from_fields(self.spec, k.params, params)
        in_arrays = [inputs[n] for n, _ in k.inputs]
        out_arrays = [np.zeros(tuple(int(d) for d in output_shapes[n]), dtype=TENSOR_NUMPY[d]) for n, d in k.outputs]
        in_tensors = (self._tensor * len(in_arrays))(
            *[self._tensor_for(a, d, f"{entry} input {n}") for a, (n, d) in zip(in_arrays, k.inputs)])
        out_tensors = (self._tensor * len(out_arrays))(
            *[self._tensor_for(a, d, f"{entry} output {n}") for a, (n, d) in zip(out_arrays, k.outputs)])
        code = self._fns[entry](ctypes.byref(struct), in_tensors, len(in_arrays), out_tensors, len(out_arrays))
        self._check(entry, code)
        return dict(zip(want_out, out_arrays))

    def raw(self, entry: str):
        """The bound ctypes function, for negative-path tests."""
        return self._fns[entry]

    def tensor(self, array: np.ndarray, dtype_name: str) -> ctypes.Structure:
        return self._tensor_for(array, dtype_name, "tensor")


@lru_cache(maxsize=1)
def get_bindings() -> Bindings:
    """The reference library for this environment, built on first use."""
    return Bindings(ensure_reference_library())


def loaded_library_key() -> Optional[str]:
    """Key of the library this process loaded, or None if none was loaded yet."""
    return get_bindings().key if get_bindings.cache_info().currsize else None


def output_shape_for(entry: str, *shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """numpy broadcast shape of `shapes` (entries that broadcast their inputs)."""
    try:
        return tuple(int(d) for d in np.broadcast_shapes(*shapes))
    except ValueError as exc:
        raise ValueError(f"{entry}: shapes {shapes} do not broadcast") from exc
