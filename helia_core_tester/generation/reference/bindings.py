"""ctypes bindings for the hct_ref shim (reference_kernels/shim/hct_ref.h).

Structures mirror the header field for field; their sizes are checked against
the library's own sizeof at load, together with the ABI version, so a header
edit without the matching edit here fails at load instead of corrupting
arguments. Arrays must already have the exact dtype and be C-contiguous: the
bindings never copy or cast silently, since a silent cast is exactly the kind
of quantization bug the reference exists to catch.
"""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

ABI_VERSION = 1
MAX_RANK = 6

OK = 0
E_DIMS = 1
E_PARAM = 2
E_UNSUPPORTED = 3
E_NULL = 4
_ERROR_NAMES = {E_DIMS: "E_DIMS", E_PARAM: "E_PARAM", E_UNSUPPORTED: "E_UNSUPPORTED", E_NULL: "E_NULL"}

ACT_NONE = 0
ACT_RELU = 1
ACT_RELU_N1_TO_1 = 2
ACT_RELU6 = 3

c_int32 = ctypes.c_int32
c_int32_p = ctypes.POINTER(ctypes.c_int32)
c_void_p = ctypes.c_void_p


class ReferenceKernelError(RuntimeError):
    """A shim entry rejected its arguments."""

    def __init__(self, entry: str, code: int, detail: str = ""):
        self.entry = entry
        self.code = code
        name = _ERROR_NAMES.get(code, f"code {code}")
        super().__init__(f"{entry} returned {name}" + (f": {detail}" if detail else ""))


class ReferenceAbiMismatch(RuntimeError):
    """The loaded library does not match these bindings."""


class HctShape(ctypes.Structure):
    _fields_ = [("rank", c_int32), ("dims", c_int32 * MAX_RANK)]


class HctActivation(ctypes.Structure):
    _fields_ = [("min", c_int32), ("max", c_int32), ("fmin", ctypes.c_float), ("fmax", ctypes.c_float)]


class HctConvParams(ctypes.Structure):
    _fields_ = [
        ("stride_h", c_int32),
        ("stride_w", c_int32),
        ("dilation_h", c_int32),
        ("dilation_w", c_int32),
        ("pad_h", c_int32),
        ("pad_w", c_int32),
        ("pad_h_offset", c_int32),
        ("pad_w_offset", c_int32),
        ("input_offset", c_int32),
        ("output_offset", c_int32),
        ("act", HctActivation),
    ]


class HctDwConvParams(ctypes.Structure):
    _fields_ = [("conv", HctConvParams), ("depth_multiplier", c_int32)]


class HctFcParams(ctypes.Structure):
    _fields_ = [
        ("input_offset", c_int32),
        ("weights_offset", c_int32),
        ("output_offset", c_int32),
        ("act", HctActivation),
    ]


class HctPoolParams(ctypes.Structure):
    _fields_ = [
        ("stride_h", c_int32),
        ("stride_w", c_int32),
        ("filter_h", c_int32),
        ("filter_w", c_int32),
        ("pad_h", c_int32),
        ("pad_w", c_int32),
        ("pad_h_offset", c_int32),
        ("pad_w_offset", c_int32),
        ("act", HctActivation),
    ]


class HctQuant(ctypes.Structure):
    _fields_ = [("multiplier", c_int32), ("shift", c_int32)]


class HctPerChannelQuant(ctypes.Structure):
    _fields_ = [("multiplier", c_int32_p), ("shift", c_int32_p), ("count", c_int32)]


STRUCTS = {
    "HctShape": HctShape,
    "HctActivation": HctActivation,
    "HctConvParams": HctConvParams,
    "HctDwConvParams": HctDwConvParams,
    "HctFcParams": HctFcParams,
    "HctPoolParams": HctPoolParams,
    "HctQuant": HctQuant,
    "HctPerChannelQuant": HctPerChannelQuant,
}

_P = ctypes.POINTER

# Kernel families: (input dtype, filter dtype, bias dtype, output dtype).
_QUANT_KINDS: Dict[str, Tuple[type, type, type, type]] = {
    "s8": (np.int8, np.int8, np.int32, np.int8),
    "s4": (np.int8, np.int8, np.int32, np.int8),
    "s16": (np.int16, np.int8, np.int64, np.int16),
    "s16_b32": (np.int16, np.int8, np.int32, np.int16),
}
_FLOAT_KIND = (np.float32, np.float32, np.float32, np.float32)

_QUANT_SIG = [c_void_p, _P(HctPerChannelQuant), _P(HctShape), c_void_p, _P(HctShape), c_void_p, c_void_p, c_int32, _P(HctShape), c_void_p]
_FLOAT_SIG = [c_void_p, _P(HctShape), c_void_p, _P(HctShape), c_void_p, c_void_p, c_int32, _P(HctShape), c_void_p]
_POOL_SIG = [_P(HctPoolParams), _P(HctShape), c_void_p, _P(HctShape), c_void_p]

_ENTRIES: Dict[str, list] = {
    "hct_ref_abi_version": [],
    "hct_ref_sizeof": [ctypes.c_char_p],
    "hct_ref_quantize_multiplier": [ctypes.c_double, _P(HctQuant)],
    "hct_ref_quantize_multiplier_smaller_than_one_exp": [ctypes.c_double, _P(HctQuant)],
    "hct_ref_preprocess_softmax_scaling": [ctypes.c_double, ctypes.c_double, c_int32, _P(HctQuant)],
    "hct_ref_calculate_input_radius": [c_int32, c_int32, c_int32, c_int32_p],
    "hct_ref_activation_range_quantized": [c_int32, ctypes.c_float, c_int32, c_int32, c_int32, c_int32_p, c_int32_p],
    "hct_ref_downscale_multiplier_to_s16": [c_int32, ctypes.POINTER(ctypes.c_int16)],
}
for _family in ("conv", "dwconv", "fc", "tconv"):
    _kinds = ["s8", "s16"] + (["s4"] if _family in ("conv", "dwconv", "fc") else []) + (["s16_b32"] if _family == "conv" else [])
    for _kind in _kinds:
        _ENTRIES[f"hct_ref_{_family}_{_kind}"] = _QUANT_SIG
    _ENTRIES[f"hct_ref_{_family}_f32"] = _FLOAT_SIG
for _pool in ("avgpool", "maxpool"):
    for _kind in ("s8", "s16", "f32"):
        _ENTRIES[f"hct_ref_{_pool}_{_kind}"] = _POOL_SIG


def make_shape(shape: Sequence[int]) -> HctShape:
    dims = [int(d) for d in shape]
    if not 1 <= len(dims) <= MAX_RANK:
        raise ValueError(f"rank {len(dims)} outside 1..{MAX_RANK}")
    out = HctShape()
    out.rank = len(dims)
    for i, d in enumerate(dims):
        out.dims[i] = d
    return out


def make_activation(qmin: int, qmax: int, fmin: float = -np.inf, fmax: float = np.inf) -> HctActivation:
    return HctActivation(int(qmin), int(qmax), float(fmin), float(fmax))


def _require(array: np.ndarray, dtype: type, name: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must be {np.dtype(dtype).name}, got {array.dtype.name}")
    if not array.flags.c_contiguous:
        raise TypeError(f"{name} must be C-contiguous")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def _ptr(array: Optional[np.ndarray]) -> Optional[int]:
    return None if array is None else array.ctypes.data


def packed_int4_size(count: int) -> int:
    return (int(count) + 1) // 2


@dataclass(frozen=True)
class PerChannel:
    """Requantization multipliers/shifts; one entry selects per-tensor (FC only)."""

    multiplier: np.ndarray
    shift: np.ndarray

    def __post_init__(self) -> None:
        _require(self.multiplier, np.int32, "multiplier")
        _require(self.shift, np.int32, "shift")
        if self.multiplier.ndim != 1 or self.multiplier.shape != self.shift.shape:
            raise ValueError("multiplier and shift must be 1-D and the same length")

    def as_struct(self) -> HctPerChannelQuant:
        return HctPerChannelQuant(
            self.multiplier.ctypes.data_as(c_int32_p), self.shift.ctypes.data_as(c_int32_p), int(self.multiplier.size)
        )


class Bindings:
    """Loaded hct_ref library with typed wrappers."""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self._lib = ctypes.CDLL(str(self.path))
        except OSError as exc:
            raise ReferenceAbiMismatch(f"cannot load {self.path}: {exc}") from exc
        for name, argtypes in _ENTRIES.items():
            try:
                fn = getattr(self._lib, name)
            except AttributeError as exc:
                raise ReferenceAbiMismatch(f"{self.path} has no symbol {name}") from exc
            fn.argtypes = argtypes
            fn.restype = c_int32
        version = self._lib.hct_ref_abi_version()
        if version != ABI_VERSION:
            raise ReferenceAbiMismatch(f"library ABI {version}, bindings expect {ABI_VERSION}")
        for type_name, struct in STRUCTS.items():
            native = self._lib.hct_ref_sizeof(type_name.encode())
            if native != ctypes.sizeof(struct):
                raise ReferenceAbiMismatch(
                    f"sizeof({type_name}): library {native}, bindings {ctypes.sizeof(struct)}"
                )

    def _check(self, entry: str, code: int) -> None:
        if code != OK:
            raise ReferenceKernelError(entry, code)

    def _fn(self, entry: str) -> Callable[..., int]:
        return getattr(self._lib, entry)

    # ---- parameter preparation ----
    def quantize_multiplier(self, real_multiplier: float) -> Tuple[int, int]:
        out = HctQuant()
        self._check("hct_ref_quantize_multiplier", self._lib.hct_ref_quantize_multiplier(float(real_multiplier), ctypes.byref(out)))
        return int(out.multiplier), int(out.shift)

    def quantize_multiplier_smaller_than_one_exp(self, real_multiplier: float) -> Tuple[int, int]:
        out = HctQuant()
        entry = "hct_ref_quantize_multiplier_smaller_than_one_exp"
        self._check(entry, self._fn(entry)(float(real_multiplier), ctypes.byref(out)))
        return int(out.multiplier), int(out.shift)

    def preprocess_softmax_scaling(self, beta: float, input_scale: float, input_integer_bits: int) -> Tuple[int, int]:
        out = HctQuant()
        entry = "hct_ref_preprocess_softmax_scaling"
        self._check(entry, self._fn(entry)(float(beta), float(input_scale), int(input_integer_bits), ctypes.byref(out)))
        return int(out.multiplier), int(out.shift)

    def calculate_input_radius(self, input_integer_bits: int, input_left_shift: int, total_signed_bits: int) -> int:
        out = c_int32()
        entry = "hct_ref_calculate_input_radius"
        self._check(entry, self._fn(entry)(int(input_integer_bits), int(input_left_shift), int(total_signed_bits), ctypes.byref(out)))
        return int(out.value)

    def activation_range_quantized(self, activation: int, scale: float, zero_point: int, qmin: int, qmax: int) -> Tuple[int, int]:
        lo, hi = c_int32(), c_int32()
        entry = "hct_ref_activation_range_quantized"
        code = self._fn(entry)(int(activation), float(scale), int(zero_point), int(qmin), int(qmax), ctypes.byref(lo), ctypes.byref(hi))
        self._check(entry, code)
        return int(lo.value), int(hi.value)

    def downscale_multiplier_to_s16(self, multiplier: int) -> int:
        out = ctypes.c_int16()
        entry = "hct_ref_downscale_multiplier_to_s16"
        self._check(entry, self._fn(entry)(int(multiplier), ctypes.byref(out)))
        return int(out.value)

    # ---- kernels ----
    def _weighted(
        self,
        family: str,
        kind: str,
        params: ctypes.Structure,
        quant: Optional[PerChannel],
        input: np.ndarray,
        filter: np.ndarray,
        bias: Optional[np.ndarray],
        output_shape: Sequence[int],
        filter_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        entry = f"hct_ref_{family}_{kind}"
        if kind == "f32":
            in_t, w_t, b_t, out_t = _FLOAT_KIND
        elif kind in _QUANT_KINDS:
            in_t, w_t, b_t, out_t = _QUANT_KINDS[kind]
        else:
            raise ValueError(f"unknown kernel kind {kind!r}")
        if entry not in _ENTRIES:
            raise ValueError(f"{entry} is not a shim entry")
        _require(input, in_t, "input")
        _require(filter, w_t, "filter")
        if bias is not None:
            _require(bias, b_t, "bias")
        if kind == "s4":
            if filter_shape is None:
                raise ValueError("s4 needs the unpacked filter_shape")
            if filter.size != packed_int4_size(int(np.prod(filter_shape))):
                raise ValueError(
                    f"packed s4 filter has {filter.size} bytes, shape {tuple(filter_shape)} needs "
                    f"{packed_int4_size(int(np.prod(filter_shape)))}"
                )
        else:
            filter_shape = filter.shape if filter_shape is None else filter_shape
            if int(np.prod(filter_shape)) != filter.size:
                raise ValueError(f"filter_shape {tuple(filter_shape)} does not match {filter.size} elements")
        output = np.zeros(tuple(int(d) for d in output_shape), dtype=out_t)
        in_shape, f_shape, o_shape = make_shape(input.shape), make_shape(filter_shape), make_shape(output.shape)
        bias_len = 0 if bias is None else int(bias.size)
        args = [ctypes.byref(params)]
        if kind != "f32":
            if quant is None:
                raise ValueError(f"{entry} needs per-channel quantization")
            args.append(ctypes.byref(quant.as_struct()))
        elif quant is not None:
            raise ValueError(f"{entry} takes no quantization")
        args += [
            ctypes.byref(in_shape),
            _ptr(input),
            ctypes.byref(f_shape),
            _ptr(filter),
            _ptr(bias),
            bias_len,
            ctypes.byref(o_shape),
            _ptr(output),
        ]
        self._check(entry, self._fn(entry)(*args))
        return output

    def conv(self, kind: str, params: HctConvParams, quant: Optional[PerChannel], input: np.ndarray, filter: np.ndarray, bias: Optional[np.ndarray], output_shape: Sequence[int], filter_shape: Optional[Sequence[int]] = None) -> np.ndarray:
        return self._weighted("conv", kind, params, quant, input, filter, bias, output_shape, filter_shape)

    def dwconv(self, kind: str, params: HctDwConvParams, quant: Optional[PerChannel], input: np.ndarray, filter: np.ndarray, bias: Optional[np.ndarray], output_shape: Sequence[int], filter_shape: Optional[Sequence[int]] = None) -> np.ndarray:
        return self._weighted("dwconv", kind, params, quant, input, filter, bias, output_shape, filter_shape)

    def fc(self, kind: str, params: HctFcParams, quant: Optional[PerChannel], input: np.ndarray, filter: np.ndarray, bias: Optional[np.ndarray], output_shape: Sequence[int], filter_shape: Optional[Sequence[int]] = None) -> np.ndarray:
        return self._weighted("fc", kind, params, quant, input, filter, bias, output_shape, filter_shape)

    def tconv(self, kind: str, params: HctConvParams, quant: Optional[PerChannel], input: np.ndarray, filter: np.ndarray, bias: Optional[np.ndarray], output_shape: Sequence[int]) -> np.ndarray:
        return self._weighted("tconv", kind, params, quant, input, filter, bias, output_shape)

    def pool(self, op: str, kind: str, params: HctPoolParams, input: np.ndarray, output_shape: Sequence[int]) -> np.ndarray:
        if op not in ("avgpool", "maxpool"):
            raise ValueError(f"unknown pool {op!r}")
        dtype = {"s8": np.int8, "s16": np.int16, "f32": np.float32}.get(kind)
        if dtype is None:
            raise ValueError(f"unknown pool kind {kind!r}")
        entry = f"hct_ref_{op}_{kind}"
        _require(input, dtype, "input")
        output = np.zeros(tuple(int(d) for d in output_shape), dtype=dtype)
        in_shape, o_shape = make_shape(input.shape), make_shape(output.shape)
        self._check(entry, self._fn(entry)(ctypes.byref(params), ctypes.byref(in_shape), _ptr(input), ctypes.byref(o_shape), _ptr(output)))
        return output


_lock = threading.Lock()
_loaded: Optional[Bindings] = None
_loaded_key: Optional[str] = None


def get_bindings() -> Bindings:
    """Process-wide bindings over the environment's reference library."""
    global _loaded, _loaded_key
    from helia_core_tester.generation.reference.host_build import ensure_reference_library

    with _lock:
        if _loaded is None:
            library = ensure_reference_library()
            _loaded = Bindings(library.path)
            _loaded_key = library.key
        return _loaded


def loaded_library_key() -> Optional[str]:
    """Key of the library get_bindings() loaded, for provenance records."""
    get_bindings()
    return _loaded_key
