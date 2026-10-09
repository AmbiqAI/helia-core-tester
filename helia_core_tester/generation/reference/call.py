"""ReferenceCall: one reference-kernel invocation, and its provenance record.

Every golden in the tester is the output of a ReferenceCall. A call is fully
explicit -- the kernel entry, its params struct as plain values, its input
tensors and its output shapes -- and is validated against the ABI spec when it
is built, not when it runs. `<case>.reference.json` records it (tensors by
shape, dtype and sha256) with the seeds and the library key: the artifact that
says how a golden was produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from helia_core_tester.generation.reference.abi import TENSOR_NUMPY, load_spec

SCHEMA = "helia-core-tester/reference-call/2"


@dataclass(frozen=True)
class ReferenceCall:
    entry: str
    params: Mapping[str, Any]
    inputs: Mapping[str, np.ndarray]
    output_shapes: Mapping[str, Tuple[int, ...]]
    quant: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spec = load_spec()
        if self.entry not in spec.kernels:
            raise KeyError(f"unknown reference entry {self.entry!r}")
        k = spec.kernels[self.entry]
        if sorted(self.inputs) != sorted(n for n, _ in k.inputs):
            raise KeyError(f"{self.entry}: inputs {sorted(self.inputs)}, expected {sorted(n for n, _ in k.inputs)}")
        for name, dtype in k.inputs:
            array = self.inputs[name]
            if not isinstance(array, np.ndarray) or array.dtype != TENSOR_NUMPY[dtype]:
                raise TypeError(f"{self.entry}: input {name} must be a {dtype} numpy array")
        shapes = {}
        for name, _ in k.outputs:
            if name not in self.output_shapes:
                raise KeyError(f"{self.entry}: missing output shape {name!r}")
            shape = tuple(int(d) for d in self.output_shapes[name])
            if any(d < 0 for d in shape):
                raise ValueError(f"{self.entry}: invalid output shape {shape}")
            shapes[name] = shape
        if set(self.output_shapes) - set(shapes):
            raise KeyError(f"{self.entry}: unknown outputs {sorted(set(self.output_shapes) - set(shapes))}")
        object.__setattr__(self, "output_shapes", shapes)
        object.__setattr__(self, "inputs", {n: np.ascontiguousarray(a) for n, a in self.inputs.items()})

    @property
    def output_names(self) -> Tuple[str, ...]:
        return tuple(n for n, _ in load_spec().kernels[self.entry].outputs)

    def run(self, bindings=None) -> Dict[str, np.ndarray]:
        from helia_core_tester.generation.reference.bindings import get_bindings

        return (bindings or get_bindings()).run(self.entry, self.params, self.inputs, self.output_shapes)

    def output(self, bindings=None) -> np.ndarray:
        """The single output of a one-output entry."""
        names = self.output_names
        if len(names) != 1:
            raise ValueError(f"{self.entry} has outputs {names}; use run()")
        return self.run(bindings)[names[0]]

    def with_inputs(self, **inputs: np.ndarray) -> "ReferenceCall":
        """The same call on replaced input tensors (e.g. nonfinite probes)."""
        unknown = set(inputs) - set(self.inputs)
        if unknown:
            raise KeyError(f"{self.entry}: unknown inputs {sorted(unknown)}")
        return replace(self, inputs={**self.inputs, **inputs})

    def provenance(self, seeds: Optional[Mapping[str, Any]] = None, library_key: Optional[str] = None) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "abi_version": load_spec().abi_version,
            "entry": self.entry,
            "params": _jsonable(dict(self.params)),
            "inputs": {name: _tensor_record(a) for name, a in sorted(self.inputs.items())},
            "outputs": {name: list(shape) for name, shape in sorted(self.output_shapes.items())},
            "quant": _jsonable(dict(self.quant)),
            "seeds": _jsonable(dict(seeds or {})),
            "library_key": library_key,
        }

    def to_json(self, path: Path, seeds: Optional[Mapping[str, Any]] = None, library_key: Optional[str] = None) -> Path:
        path = Path(path)
        text = json.dumps(self.provenance(seeds, library_key), indent=2, sort_keys=True, allow_nan=False)
        path.write_text(text + "\n", encoding="utf-8")
        return path


def _tensor_record(array: np.ndarray) -> Dict[str, Any]:
    return {"shape": list(array.shape), "dtype": array.dtype.name, "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        # Strict JSON has no inf/nan: an unbounded float bound records as "inf".
        return float(value) if np.isfinite(value) else str(float(value))
    return value
