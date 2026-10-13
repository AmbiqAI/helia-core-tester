"""Descriptor dtypes and preset quantization, as the reference entries take them."""

from __future__ import annotations

from typing import Dict, Tuple

from helia_core_tester.generation.reference.abi import dtype_code

# Entry-name suffix and tensor dtype of each descriptor dtype.
_KINDS: Dict[str, Tuple[str, str]] = {
    "S8": ("s8", "int8"),
    "S16": ("s16", "int16"),
    "FP32": ("f32", "float32"),
    "FP16": ("f16", "float16"),
}

# Per-tensor (scale, zero point) of the operators whose quantization is a fixed preset
# rather than a policy over the draw: the values the LiteRT builder used to assign.
PRESET_QUANT: Dict[str, Tuple[float, int]] = {"S8": (0.125, 0), "S16": (1.0 / 32768.0, 0)}


def kind(descriptor_dtype: str) -> str:
    """Entry suffix (s8, s16, f32, f16) of a descriptor dtype."""
    key = str(descriptor_dtype).upper()
    if key not in _KINDS:
        raise ValueError(f"no reference entry kind for dtype {descriptor_dtype!r}")
    return _KINDS[key][0]


def hct_dtype(descriptor_dtype: str) -> int:
    """HctDtype code of a descriptor dtype."""
    key = str(descriptor_dtype).upper()
    if key not in _KINDS:
        raise ValueError(f"no reference dtype for {descriptor_dtype!r}")
    return dtype_code(_KINDS[key][1])


def preset_quant(descriptor_dtype: str) -> Tuple[float, int]:
    key = str(descriptor_dtype).upper()
    if key not in PRESET_QUANT:
        raise ValueError(f"no preset quantization for {descriptor_dtype!r}")
    return PRESET_QUANT[key]
