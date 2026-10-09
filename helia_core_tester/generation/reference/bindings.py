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

ABI_VERSION = 4
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


class HctBinaryParams(ctypes.Structure):
    _fields_ = [
        ("left_shift", c_int32),
        ("input1_offset", c_int32),
        ("input1_multiplier", c_int32),
        ("input1_shift", c_int32),
        ("input2_offset", c_int32),
        ("input2_multiplier", c_int32),
        ("input2_shift", c_int32),
        ("output_offset", c_int32),
        ("output_multiplier", c_int32),
        ("output_shift", c_int32),
        ("act", HctActivation),
    ]



def _int_struct(name: str, fields: Sequence[str]) -> type:
    return type(name, (ctypes.Structure,), {"_fields_": [(f, c_int32) for f in fields]})


HctSoftmaxParams = _int_struct("HctSoftmaxParams", ("input_multiplier", "input_left_shift", "diff_min"))
HctLutActParams = _int_struct(
    "HctLutActParams", ("input_zero_point", "input_range_radius", "input_multiplier", "input_left_shift")
)
HctLeakyReluParams = _int_struct(
    "HctLeakyReluParams",
    ("input_zero_point", "output_zero_point", "multiplier_alpha", "shift_alpha", "multiplier_identity", "shift_identity"),
)
HctPreluParams = _int_struct(
    "HctPreluParams",
    ("input_offset", "alpha_offset", "output_offset", "multiplier_1", "shift_1", "multiplier_2", "shift_2"),
)
HctHardSwishParams = _int_struct(
    "HctHardSwishParams",
    (
        "input_zero_point", "output_zero_point", "reluish_multiplier_fixedpoint_int16", "reluish_multiplier_exponent",
        "output_multiplier_fixedpoint_int16", "output_multiplier_exponent",
    ),
)

HctReluParams = _int_struct(
    "HctReluParams",
    ("input_zero_point", "output_zero_point", "output_multiplier", "output_shift", "act_min", "act_max"),
)

HctMeanParams = _int_struct("HctMeanParams", ("input_zero_point", "output_zero_point", "multiplier", "shift", "keep_dims"))

HctRsqrtParams = _int_struct("HctRsqrtParams", ("input_zero_point", "output_zero_point", "multiplier", "shift"))


class HctBmmParams(ctypes.Structure):
    _fields_ = [
        ("lhs_offset", c_int32),
        ("rhs_offset", c_int32),
        ("output_offset", c_int32),
        ("output_multiplier", c_int32),
        ("output_shift", c_int32),
        ("act", HctActivation),
    ]


class HctLstmParams(ctypes.Structure):
    _fields_ = [
        ("batch", c_int32),
        ("time_steps", c_int32),
        ("input_size", c_int32),
        ("hidden_size", c_int32),
        ("time_major", c_int32),
        ("input_scale", ctypes.c_float),
        ("input_zero_point", c_int32),
        ("output_scale", ctypes.c_float),
        ("output_zero_point", c_int32),
        ("cell_scale", ctypes.c_float),
        ("cell_clip", ctypes.c_float),
        ("weight_scales", ctypes.c_float * 8),
    ]


HctSvdfParams = _int_struct(
    "HctSvdfParams",
    ("batch", "input_size", "num_filters", "memory_size", "rank", "input_zero_point", "output_zero_point",
     "scale1_multiplier", "scale1_shift", "scale2_multiplier", "scale2_shift"),
)


# HctLstmParams.weight_scales / the weights pointer array order.
LSTM_WEIGHT_ORDER = tuple(f"{g}_gate_{k}" for k in ("input", "hidden") for g in ("input", "forget", "cell", "output"))
LSTM_BIAS_ORDER = tuple(f"{g}_gate_bias" for g in ("input", "forget", "cell", "output"))


def struct_to_dict(struct: ctypes.Structure) -> Dict[str, int]:
    """The integer fields of a flat params struct, for provenance and harness contexts."""
    return {name: int(getattr(struct, name)) for name, _ in struct._fields_}


STRUCTS = {
    "HctShape": HctShape,
    "HctActivation": HctActivation,
    "HctConvParams": HctConvParams,
    "HctDwConvParams": HctDwConvParams,
    "HctFcParams": HctFcParams,
    "HctPoolParams": HctPoolParams,
    "HctQuant": HctQuant,
    "HctPerChannelQuant": HctPerChannelQuant,
    "HctBinaryParams": HctBinaryParams,
    "HctSoftmaxParams": HctSoftmaxParams,
    "HctLutActParams": HctLutActParams,
    "HctLeakyReluParams": HctLeakyReluParams,
    "HctPreluParams": HctPreluParams,
    "HctHardSwishParams": HctHardSwishParams,
    "HctReluParams": HctReluParams,
    "HctMeanParams": HctMeanParams,
    "HctRsqrtParams": HctRsqrtParams,
    "HctBmmParams": HctBmmParams,
    "HctLstmParams": HctLstmParams,
    "HctSvdfParams": HctSvdfParams,
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
_BINARY_SIG = [_P(HctBinaryParams), _P(HctShape), c_void_p, _P(HctShape), c_void_p, _P(HctShape), c_void_p]
for _op in ("add", "sub", "mul"):
    for _kind in ("s8", "s16"):
        _ENTRIES[f"hct_ref_{_op}_{_kind}"] = _BINARY_SIG
for _pool in ("avgpool", "maxpool"):
    for _kind in ("s8", "s16", "f32"):
        _ENTRIES[f"hct_ref_{_pool}_{_kind}"] = _POOL_SIG
_c_float = ctypes.c_float
_ENTRIES.update({
    "hct_ref_softmax_s8": [_P(HctSoftmaxParams), _P(HctShape), c_void_p, c_void_p],
    "hct_ref_softmax_s16": [_P(HctSoftmaxParams), _P(HctShape), c_void_p, c_void_p],
    "hct_ref_tanh_logistic_s16_prepare": [c_int32, _c_float, _c_float, _P(HctLutActParams)],
    "hct_ref_tanh_s16": [_P(HctLutActParams), _P(HctShape), c_void_p, c_void_p],
    "hct_ref_logistic_s16": [_P(HctLutActParams), _P(HctShape), c_void_p, c_void_p],
    "hct_ref_leaky_relu_prepare": [_c_float, c_int32, _c_float, _c_float, c_int32, _P(HctLeakyReluParams)],
    "hct_ref_leaky_relu_s8": [_P(HctLeakyReluParams), _P(HctShape), c_void_p, c_void_p],
    "hct_ref_leaky_relu_s16": [_P(HctLeakyReluParams), _P(HctShape), c_void_p, c_void_p],
    "hct_ref_prelu_prepare": [_c_float, c_int32, _c_float, c_int32, _c_float, c_int32, _P(HctPreluParams)],
    "hct_ref_prelu_s8": [_P(HctPreluParams), _P(HctShape), c_void_p, _P(HctShape), c_void_p, _P(HctShape), c_void_p],
    "hct_ref_hard_swish_prepare": [_c_float, c_int32, _c_float, c_int32, _P(HctHardSwishParams)],
    "hct_ref_hard_swish_s8": [_P(HctHardSwishParams), _P(HctShape), c_void_p, c_void_p],
    "hct_ref_relu_prepare": [_c_float, c_int32, _c_float, c_int32, _c_float, _c_float, c_int32, c_int32, _P(HctReluParams)],
    "hct_ref_relu_s8": [_P(HctReluParams), _P(HctShape), c_void_p, c_void_p],
    "hct_ref_relu_s16": [_P(HctReluParams), _P(HctShape), c_void_p, c_void_p],
    "hct_ref_rsqrt_prepare": [_c_float, c_int32, _c_float, c_int32, _P(HctRsqrtParams)],
    "hct_ref_rsqrt_s8": [_P(HctRsqrtParams), _P(HctShape), c_void_p, c_void_p],
    "hct_ref_rsqrt_s16": [_P(HctRsqrtParams), _c_float, _c_float, _P(HctShape), c_void_p, c_void_p],
    "hct_ref_quantize_f32_s8": [_c_float, c_int32, _P(HctShape), c_void_p, c_void_p],
    "hct_ref_quantize_f32_s16": [_c_float, c_int32, _P(HctShape), c_void_p, c_void_p],
    "hct_ref_bmm_s8": [_P(HctBmmParams), _P(HctShape), c_void_p, _P(HctShape), c_void_p, _P(HctShape), c_void_p],
    "hct_ref_bmm_s16": [_P(HctBmmParams), _P(HctShape), c_void_p, _P(HctShape), c_void_p, _P(HctShape), c_void_p],
    "hct_ref_lstm_s8": [_P(HctLstmParams), c_void_p, ctypes.POINTER(c_void_p), ctypes.POINTER(c_void_p), c_void_p],
    "hct_ref_lstm_s16": [_P(HctLstmParams), c_void_p, ctypes.POINTER(c_void_p), ctypes.POINTER(c_void_p), c_void_p],
    "hct_ref_svdf_s8": [_P(HctSvdfParams), c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p],
    "hct_ref_svdf_s8_state_s16": [_P(HctSvdfParams), c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p],
    "hct_ref_mean_fold": [c_int32, c_int32, ctypes.c_int64, _P(HctQuant)],
    "hct_ref_mean_s8": [_P(HctMeanParams), _P(HctShape), c_void_p, c_int32_p, c_int32, _P(HctShape), c_void_p],
    "hct_ref_mean_s16": [_P(HctMeanParams), _P(HctShape), c_void_p, c_int32_p, c_int32, _P(HctShape), c_void_p],
})

# Elementwise unary entries: (params struct, element dtype).
UNARY_ENTRIES: Dict[str, Tuple[type, type]] = {
    "softmax_s8": (HctSoftmaxParams, np.int8),
    "softmax_s16": (HctSoftmaxParams, np.int16),
    "tanh_s16": (HctLutActParams, np.int16),
    "logistic_s16": (HctLutActParams, np.int16),
    "leaky_relu_s8": (HctLeakyReluParams, np.int8),
    "leaky_relu_s16": (HctLeakyReluParams, np.int16),
    "hard_swish_s8": (HctHardSwishParams, np.int8),
    "relu_s8": (HctReluParams, np.int8),
    "relu_s16": (HctReluParams, np.int16),
    "rsqrt_s8": (HctRsqrtParams, np.int8),
}


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

    def _prepare(self, entry: str, struct_type: type, *args) -> ctypes.Structure:
        out = struct_type()
        self._check(entry, self._fn(entry)(*args, ctypes.byref(out)))
        return out

    def tanh_logistic_s16_prepare(self, logistic: bool, input_scale: float, output_scale: float):
        return self._prepare(
            "hct_ref_tanh_logistic_s16_prepare", HctLutActParams, int(bool(logistic)), float(input_scale), float(output_scale)
        )

    def leaky_relu_prepare(self, input_scale: float, input_zp: int, alpha: float, output_scale: float, output_zp: int):
        return self._prepare(
            "hct_ref_leaky_relu_prepare", HctLeakyReluParams,
            float(input_scale), int(input_zp), float(alpha), float(output_scale), int(output_zp),
        )

    def prelu_prepare(self, input_scale: float, input_zp: int, alpha_scale: float, alpha_zp: int, output_scale: float, output_zp: int):
        return self._prepare(
            "hct_ref_prelu_prepare", HctPreluParams,
            float(input_scale), int(input_zp), float(alpha_scale), int(alpha_zp), float(output_scale), int(output_zp),
        )

    def hard_swish_prepare(self, input_scale: float, input_zp: int, output_scale: float, output_zp: int):
        return self._prepare(
            "hct_ref_hard_swish_prepare", HctHardSwishParams, float(input_scale), int(input_zp), float(output_scale), int(output_zp)
        )

    def relu_prepare(self, input_scale: float, input_zp: int, output_scale: float, output_zp: int,
                     act_min_real: float, act_max_real: float, qmin: int, qmax: int):
        return self._prepare(
            "hct_ref_relu_prepare", HctReluParams, float(input_scale), int(input_zp), float(output_scale),
            int(output_zp), float(act_min_real), float(act_max_real), int(qmin), int(qmax),
        )

    def rsqrt_prepare(self, input_scale: float, input_zp: int, output_scale: float, output_zp: int):
        return self._prepare(
            "hct_ref_rsqrt_prepare", HctRsqrtParams, float(input_scale), int(input_zp), float(output_scale), int(output_zp)
        )

    def mean_fold(self, multiplier: int, shift: int, count: int) -> Tuple[int, int]:
        out = HctQuant()
        self._check("hct_ref_mean_fold", self._fn("hct_ref_mean_fold")(int(multiplier), int(shift), int(count), ctypes.byref(out)))
        return int(out.multiplier), int(out.shift)

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

    def binary(self, op: str, kind: str, params: HctBinaryParams, input1: np.ndarray, input2: np.ndarray, output_shape: Sequence[int]) -> np.ndarray:
        """add/sub/mul with TFLM broadcasting; output_shape is the broadcast shape."""
        if op not in ("add", "sub", "mul"):
            raise ValueError(f"unknown binary op {op!r}")
        dtype = {"s8": np.int8, "s16": np.int16}.get(kind)
        if dtype is None:
            raise ValueError(f"unknown binary kind {kind!r}")
        entry = f"hct_ref_{op}_{kind}"
        _require(input1, dtype, "input1")
        _require(input2, dtype, "input2")
        output = np.zeros(tuple(int(d) for d in output_shape), dtype=dtype)
        shapes = [make_shape(a.shape) for a in (input1, input2, output)]
        code = self._fn(entry)(
            ctypes.byref(params), ctypes.byref(shapes[0]), _ptr(input1), ctypes.byref(shapes[1]), _ptr(input2),
            ctypes.byref(shapes[2]), _ptr(output),
        )
        self._check(entry, code)
        return output

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

    def unary(self, name: str, params: ctypes.Structure, input: np.ndarray) -> np.ndarray:
        """An elementwise entry of UNARY_ENTRIES; the output has the input's shape and dtype."""
        try:
            struct_type, dtype = UNARY_ENTRIES[name]
        except KeyError as exc:
            raise ValueError(f"unknown unary entry {name!r}") from exc
        if not isinstance(params, struct_type):
            raise TypeError(f"{name} takes {struct_type.__name__}, got {type(params).__name__}")
        entry = f"hct_ref_{name}"
        _require(input, dtype, "input")
        output = np.zeros(input.shape, dtype=dtype)
        shape = make_shape(input.shape)
        self._check(entry, self._fn(entry)(ctypes.byref(params), ctypes.byref(shape), _ptr(input), _ptr(output)))
        return output

    def mean(self, kind: str, params: ctypes.Structure, input: np.ndarray, axes: Sequence[int], output_shape: Sequence[int]) -> np.ndarray:
        dtype = {"s8": np.int8, "s16": np.int16}.get(kind)
        if dtype is None:
            raise ValueError(f"unknown mean kind {kind!r}")
        if not isinstance(params, HctMeanParams):
            raise TypeError(f"mean takes HctMeanParams, got {type(params).__name__}")
        entry = f"hct_ref_mean_{kind}"
        _require(input, dtype, "input")
        axis_arr = np.ascontiguousarray(axes, dtype=np.int32)
        if axis_arr.ndim != 1 or axis_arr.size == 0:
            raise ValueError("axes must be a non-empty 1-D sequence")
        output = np.zeros(tuple(int(d) for d in output_shape), dtype=dtype)
        in_shape, o_shape = make_shape(input.shape), make_shape(output.shape)
        code = self._fn(entry)(
            ctypes.byref(params), ctypes.byref(in_shape), _ptr(input),
            axis_arr.ctypes.data_as(c_int32_p), int(axis_arr.size), ctypes.byref(o_shape), _ptr(output),
        )
        self._check(entry, code)
        return output

    def rsqrt_s16(self, params: ctypes.Structure, input_scale: float, output_scale: float, input: np.ndarray) -> np.ndarray:
        if not isinstance(params, HctRsqrtParams):
            raise TypeError(f"rsqrt_s16 takes HctRsqrtParams, got {type(params).__name__}")
        entry = "hct_ref_rsqrt_s16"
        _require(input, np.int16, "input")
        output = np.zeros(input.shape, dtype=np.int16)
        shape = make_shape(input.shape)
        code = self._fn(entry)(ctypes.byref(params), float(input_scale), float(output_scale), ctypes.byref(shape), _ptr(input), _ptr(output))
        self._check(entry, code)
        return output

    def quantize_f32(self, kind: str, scale: float, zero_point: int, input: np.ndarray) -> np.ndarray:
        dtype = {"s8": np.int8, "s16": np.int16}.get(kind)
        if dtype is None:
            raise ValueError(f"unknown quantize kind {kind!r}")
        entry = f"hct_ref_quantize_f32_{kind}"
        _require(input, np.float32, "input")
        output = np.zeros(input.shape, dtype=dtype)
        shape = make_shape(input.shape)
        self._check(entry, self._fn(entry)(float(scale), int(zero_point), ctypes.byref(shape), _ptr(input), _ptr(output)))
        return output

    def bmm(self, kind: str, params: ctypes.Structure, lhs: np.ndarray, rhs: np.ndarray, output_shape: Sequence[int]) -> np.ndarray:
        dtype = {"s8": np.int8, "s16": np.int16}.get(kind)
        if dtype is None:
            raise ValueError(f"unknown bmm kind {kind!r}")
        if not isinstance(params, HctBmmParams):
            raise TypeError(f"bmm takes HctBmmParams, got {type(params).__name__}")
        entry = f"hct_ref_bmm_{kind}"
        _require(lhs, dtype, "lhs")
        _require(rhs, dtype, "rhs")
        output = np.zeros(tuple(int(d) for d in output_shape), dtype=dtype)
        shapes = [make_shape(a.shape) for a in (lhs, rhs, output)]
        code = self._fn(entry)(
            ctypes.byref(params), ctypes.byref(shapes[0]), _ptr(lhs), ctypes.byref(shapes[1]), _ptr(rhs),
            ctypes.byref(shapes[2]), _ptr(output),
        )
        self._check(entry, code)
        return output

    def lstm(self, kind: str, params: ctypes.Structure, input: np.ndarray, weights: Dict[str, np.ndarray],
             biases: Dict[str, np.ndarray]) -> np.ndarray:
        """TFLM integer LSTM; weights/biases keyed as LSTM_WEIGHT_ORDER / LSTM_BIAS_ORDER."""
        types = {"s8": (np.int8, np.int32), "s16": (np.int16, np.int64)}.get(kind)
        if types is None:
            raise ValueError(f"unknown lstm kind {kind!r}")
        if not isinstance(params, HctLstmParams):
            raise TypeError(f"lstm takes HctLstmParams, got {type(params).__name__}")
        act, bias_t = types
        b, t, i, h = params.batch, params.time_steps, params.input_size, params.hidden_size
        lead = (t, b) if params.time_major else (b, t)
        _require(input, act, "input")
        if input.shape != lead + (i,):
            raise ValueError(f"input shape {input.shape}, params say {lead + (i,)}")
        w_arrays = [_require(weights[k], np.int8, k) for k in LSTM_WEIGHT_ORDER]
        for k, w in zip(LSTM_WEIGHT_ORDER, w_arrays):
            if w.shape != ((h, i) if k.endswith("_input") else (h, h)):
                raise ValueError(f"{k} shape {w.shape} does not match hidden {h} / input {i}")
        b_arrays = [_require(biases[k], bias_t, k) for k in LSTM_BIAS_ORDER]
        for k, v in zip(LSTM_BIAS_ORDER, b_arrays):
            if v.shape != (h,):
                raise ValueError(f"{k} shape {v.shape}, expected ({h},)")
        output = np.zeros(lead + (h,), dtype=act)
        w_ptrs = (c_void_p * 8)(*[w.ctypes.data for w in w_arrays])
        b_ptrs = (c_void_p * 4)(*[v.ctypes.data for v in b_arrays])
        entry = f"hct_ref_lstm_{kind}"
        self._check(entry, self._fn(entry)(ctypes.byref(params), _ptr(input), w_ptrs, b_ptrs, _ptr(output)))
        return output

    def svdf(self, state_kind: str, params: ctypes.Structure, input: np.ndarray, weights_feature: np.ndarray,
             weights_time: np.ndarray, bias: Optional[np.ndarray], state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One SVDF step; returns (output, the updated state). The caller's state is not modified."""
        state_t = {"s8": np.int8, "s16": np.int16}.get(state_kind)
        if state_t is None:
            raise ValueError(f"unknown svdf state kind {state_kind!r}")
        if not isinstance(params, HctSvdfParams):
            raise TypeError(f"svdf takes HctSvdfParams, got {type(params).__name__}")
        n, i, f, m, r = params.batch, params.input_size, params.num_filters, params.memory_size, params.rank
        if r < 1 or f % r:
            raise ValueError(f"rank {r} does not divide num_filters {f}")
        for name, arr, dtype, shape in (("input", input, np.int8, (n, i)), ("weights_feature", weights_feature, np.int8, (f, i)),
                                        ("weights_time", weights_time, state_t, (f, m)), ("state", state, state_t, (n, f, m))):
            _require(arr, dtype, name)
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape}, params say {shape}")
        if bias is not None:
            _require(bias, np.int32, "bias")
            if bias.shape != (f // r,):
                raise ValueError(f"bias shape {bias.shape}, expected ({f // r},)")
        new_state = np.array(state, copy=True)
        output = np.zeros((n, f // r), dtype=np.int8)
        entry = "hct_ref_svdf_s8" if state_kind == "s8" else "hct_ref_svdf_s8_state_s16"
        code = self._fn(entry)(ctypes.byref(params), _ptr(input), _ptr(weights_feature), _ptr(weights_time),
                               _ptr(bias), _ptr(new_state), _ptr(output))
        self._check(entry, code)
        return output, new_state

    def prelu_s8(self, params: ctypes.Structure, input: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if not isinstance(params, HctPreluParams):
            raise TypeError(f"prelu_s8 takes HctPreluParams, got {type(params).__name__}")
        entry = "hct_ref_prelu_s8"
        _require(input, np.int8, "input")
        _require(alpha, np.int8, "alpha")
        output = np.zeros(input.shape, dtype=np.int8)
        shapes = [make_shape(a.shape) for a in (input, alpha, output)]
        code = self._fn(entry)(
            ctypes.byref(params), ctypes.byref(shapes[0]), _ptr(input), ctypes.byref(shapes[1]), _ptr(alpha),
            ctypes.byref(shapes[2]), _ptr(output),
        )
        self._check(entry, code)
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
