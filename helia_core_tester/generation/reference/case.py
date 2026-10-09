"""ReferenceCall: one reference-kernel invocation, plus its provenance record.

A call is fully explicit: the kernel entry, its parameters (plain JSON values,
mirroring the shim structs), the integer/float tensors it consumes, and the
output shape. `<case>.reference.json` records all of it (tensors by shape,
dtype and sha256) next to the seeds and the library key, replacing the .tflite
as the artifact that says how a golden was produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

SCHEMA = "helia-core-tester/reference-call/1"


@dataclass
class ReferenceCall:
    kernel: str
    params: Dict[str, Any]
    tensors: Dict[str, np.ndarray]
    output_shape: Tuple[int, ...]
    output_dtype: str
    quant: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kernel or "_" not in self.kernel:
            raise ValueError(f"kernel must look like '<family>_<kind>', got {self.kernel!r}")
        self.output_shape = tuple(int(d) for d in self.output_shape)
        if not self.output_shape or any(d <= 0 for d in self.output_shape):
            raise ValueError(f"invalid output shape {self.output_shape}")
        for name, tensor in self.tensors.items():
            if tensor is not None and not isinstance(tensor, np.ndarray):
                raise TypeError(f"tensor {name!r} must be a numpy array")
        np.dtype(self.output_dtype)  # raises on an unknown dtype

    @property
    def family(self) -> str:
        return self.kernel.split("_", 1)[0]

    def provenance(self, seeds: Optional[Mapping[str, Any]] = None, library_key: Optional[str] = None) -> Dict[str, Any]:
        tensors = {}
        for name, tensor in sorted(self.tensors.items()):
            if tensor is None:
                tensors[name] = None
                continue
            contiguous = np.ascontiguousarray(tensor)
            tensors[name] = {
                "shape": list(contiguous.shape),
                "dtype": contiguous.dtype.name,
                "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
            }
        return {
            "schema": SCHEMA,
            "kernel": self.kernel,
            "params": _jsonable(self.params),
            "tensors": tensors,
            "output": {"shape": list(self.output_shape), "dtype": np.dtype(self.output_dtype).name},
            "quant": _jsonable(self.quant),
            "seeds": _jsonable(dict(seeds or {})),
            "library_key": library_key,
        }

    def to_json(self, path: Path, seeds: Optional[Mapping[str, Any]] = None, library_key: Optional[str] = None) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.provenance(seeds, library_key), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        # Strict JSON has no inf/nan; an unbounded float activation records as "inf".
        return float(value) if np.isfinite(value) else str(float(value))
    if hasattr(value, "to_json"):
        return _jsonable(value.to_json())
    return value
