"""Execute a ReferenceCall against the host reference library.

Parameter dictionaries (the JSON form recorded in <case>.reference.json):

  conv / tconv:  stride [h, w], dilation [h, w], pad [h, w], pad_offset [h, w],
                 input_offset, output_offset, act {min, max, fmin, fmax},
                 multiplier [...], shift [...]   (quantized kinds only)
                 filter_shape [O, H, W, I]        (s4 only: the unpacked shape)
  dwconv:        conv keys + depth_multiplier
  fc:            input_offset, weights_offset, output_offset, act,
                 multiplier [...], shift [...] (one entry = per-tensor kernel),
                 filter_shape [out, in]           (s4 only)
  avgpool / maxpool: stride [h, w], filter [h, w], pad [h, w], pad_offset [h, w], act
  add / sub / mul: left_shift, input{1,2}_offset, input{1,2}_multiplier, input{1,2}_shift,
                 output_offset, output_multiplier, output_shift, act
                 (mul: only the offsets, output multiplier/shift and act matter)
  softmax_*, tanh_s16, logistic_s16, leaky_relu_*, relu_*, rsqrt_s8, hard_swish_s8, prelu_s8:
                 the fields of the matching Hct*Params struct (bindings.UNARY_ENTRIES),
                 as the shim's *_prepare entry returned them
  rsqrt_s16:     the HctRsqrtParams fields plus input_scale, output_scale (TFLite's LUT route)
  bmm_s8 / bmm_s16: lhs_offset, rhs_offset, output_offset, output_multiplier, output_shift, act
                 (tensors lhs [..., M, K], rhs [..., N, K])
  quantize_s8 / quantize_s16: scale, zero_point (float32 input -> int output)
  mean_s8 / mean_s16: input_zero_point, output_zero_point, multiplier, shift (unfolded),
                 keep_dims, axes [...]

Tensors: input, filter, bias (bias may be absent or None); input1, input2 for add/sub/mul;
input, alpha for prelu.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from helia_core_tester.generation.reference import bindings as b
from helia_core_tester.generation.reference.case import ReferenceCall


def _pair(params: Mapping[str, Any], key: str, default: Optional[Sequence[int]] = None) -> Sequence[int]:
    value = params.get(key, default)
    if value is None:
        raise KeyError(f"missing parameter {key!r}")
    pair = [int(v) for v in value]
    if len(pair) != 2:
        raise ValueError(f"{key} must be [h, w], got {value!r}")
    return pair


def _activation(params: Mapping[str, Any]) -> b.HctActivation:
    act = params.get("act")
    if not act:
        raise KeyError("missing parameter 'act'")
    return b.make_activation(
        int(act["min"]), int(act["max"]), float(act.get("fmin", -np.inf)), float(act.get("fmax", np.inf))
    )


def _quant(params: Mapping[str, Any]) -> b.PerChannel:
    return b.PerChannel(
        np.ascontiguousarray(params["multiplier"], dtype=np.int32),
        np.ascontiguousarray(params["shift"], dtype=np.int32),
    )


def conv_struct(params: Mapping[str, Any]) -> b.HctConvParams:
    sh, sw = _pair(params, "stride")
    dh, dw = _pair(params, "dilation", (1, 1))
    ph, pw = _pair(params, "pad")
    oh, ow = _pair(params, "pad_offset", (0, 0))
    return b.HctConvParams(
        sh, sw, dh, dw, ph, pw, oh, ow, int(params.get("input_offset", 0)), int(params.get("output_offset", 0)), _activation(params)
    )


def dwconv_struct(params: Mapping[str, Any]) -> b.HctDwConvParams:
    return b.HctDwConvParams(conv_struct(params), int(params["depth_multiplier"]))


def fc_struct(params: Mapping[str, Any]) -> b.HctFcParams:
    return b.HctFcParams(
        int(params.get("input_offset", 0)),
        int(params.get("weights_offset", 0)),
        int(params.get("output_offset", 0)),
        _activation(params),
    )


def pool_struct(params: Mapping[str, Any]) -> b.HctPoolParams:
    sh, sw = _pair(params, "stride")
    fh, fw = _pair(params, "filter")
    ph, pw = _pair(params, "pad")
    oh, ow = _pair(params, "pad_offset", (0, 0))
    return b.HctPoolParams(sh, sw, fh, fw, ph, pw, oh, ow, _activation(params))


_BINARY_FIELDS = (
    "left_shift", "input1_offset", "input1_multiplier", "input1_shift", "input2_offset",
    "input2_multiplier", "input2_shift", "output_offset", "output_multiplier", "output_shift",
)


def binary_struct(params: Mapping[str, Any]) -> b.HctBinaryParams:
    return b.HctBinaryParams(*(int(params.get(k, 0)) for k in _BINARY_FIELDS), _activation(params))


def _tensor(tensors: Mapping[str, Optional[np.ndarray]], name: str) -> np.ndarray:
    tensor = tensors.get(name)
    if tensor is None:
        raise KeyError(f"missing tensor {name!r}")
    return tensor


def flat_struct(struct_type: type, params: Mapping[str, Any]):
    """A flat Hct*Params struct from its field dict; every field is required."""
    names = [name for name, _ in struct_type._fields_]
    missing = [n for n in names if n not in params]
    unknown = sorted(set(params) - set(names))
    if missing or unknown:
        raise KeyError(f"{struct_type.__name__}: missing {missing}, unknown {unknown}")
    return struct_type(*(int(params[n]) for n in names))


def _run_flat(lib: b.Bindings, call: ReferenceCall) -> np.ndarray:
    if call.kernel == "prelu_s8":
        out = lib.prelu_s8(flat_struct(b.HctPreluParams, call.params), _tensor(call.tensors, "input"), _tensor(call.tensors, "alpha"))
    else:
        struct_type, _ = b.UNARY_ENTRIES[call.kernel]
        out = lib.unary(call.kernel, flat_struct(struct_type, call.params), _tensor(call.tensors, "input"))
    if tuple(out.shape) != tuple(call.output_shape):
        raise ValueError(f"{call.kernel}: output shape {out.shape}, call expects {call.output_shape}")
    return out


def run_reference(call: ReferenceCall, lib: Optional[b.Bindings] = None) -> np.ndarray:
    """Run the call's kernel and return its output (call.output_dtype, call.output_shape)."""
    lib = lib or b.get_bindings()
    if call.kernel in b.UNARY_ENTRIES or call.kernel == "prelu_s8":
        out = _run_flat(lib, call)
        if out.dtype != np.dtype(call.output_dtype):
            raise TypeError(f"{call.kernel} produced {out.dtype}, call expects {call.output_dtype}")
        return out
    if call.kernel in ("bmm_s8", "bmm_s16"):
        p = call.params
        params = b.HctBmmParams(
            *(int(p[k]) for k in ("lhs_offset", "rhs_offset", "output_offset", "output_multiplier", "output_shift")),
            _activation(p),
        )
        out = lib.bmm(call.kernel[4:], params, _tensor(call.tensors, "lhs"), _tensor(call.tensors, "rhs"), call.output_shape)
        if out.dtype != np.dtype(call.output_dtype):
            raise TypeError(f"{call.kernel} produced {out.dtype}, call expects {call.output_dtype}")
        return out
    if call.kernel in ("quantize_s8", "quantize_s16"):
        out = lib.quantize_f32(call.kernel[9:], float(call.params["scale"]), int(call.params["zero_point"]), _tensor(call.tensors, "input"))
        if out.dtype != np.dtype(call.output_dtype):
            raise TypeError(f"{call.kernel} produced {out.dtype}, call expects {call.output_dtype}")
        return out
    if call.kernel == "rsqrt_s16":
        p = dict(call.params)
        in_s, out_s = float(p.pop("input_scale")), float(p.pop("output_scale"))
        out = lib.rsqrt_s16(flat_struct(b.HctRsqrtParams, p), in_s, out_s, _tensor(call.tensors, "input"))
        if out.dtype != np.dtype(call.output_dtype):
            raise TypeError(f"{call.kernel} produced {out.dtype}, call expects {call.output_dtype}")
        return out
    if call.kernel in ("mean_s8", "mean_s16"):
        p = dict(call.params)
        axes = [int(a) for a in p.pop("axes")]
        out = lib.mean(call.kernel[5:], flat_struct(b.HctMeanParams, p), _tensor(call.tensors, "input"), axes, call.output_shape)
        if out.dtype != np.dtype(call.output_dtype):
            raise TypeError(f"{call.kernel} produced {out.dtype}, call expects {call.output_dtype}")
        return out
    family, kind = call.kernel.split("_", 1)
    p = call.params
    t = call.tensors
    filter_shape = p.get("filter_shape")
    if family in ("conv", "dwconv", "fc", "tconv"):
        quant = None if kind == "f32" else _quant(p)
        args = (quant, _tensor(t, "input"), _tensor(t, "filter"), t.get("bias"), call.output_shape)
        if family == "conv":
            out = lib.conv(kind, conv_struct(p), *args, filter_shape=filter_shape)
        elif family == "dwconv":
            out = lib.dwconv(kind, dwconv_struct(p), *args, filter_shape=filter_shape)
        elif family == "fc":
            out = lib.fc(kind, fc_struct(p), *args, filter_shape=filter_shape)
        else:
            out = lib.tconv(kind, conv_struct(p), *args)
    elif family in ("add", "sub", "mul"):
        out = lib.binary(family, kind, binary_struct(p), _tensor(t, "input1"), _tensor(t, "input2"), call.output_shape)
    elif family in ("avgpool", "maxpool"):
        out = lib.pool(family, kind, pool_struct(p), _tensor(t, "input"), call.output_shape)
    else:
        raise ValueError(f"no reference entry for kernel {call.kernel!r}")
    if out.dtype != np.dtype(call.output_dtype):
        raise TypeError(f"{call.kernel} produced {out.dtype}, call expects {call.output_dtype}")
    return out


def describe(call: ReferenceCall) -> Dict[str, Any]:
    """Short, log-friendly description of a call."""
    return {
        "kernel": call.kernel,
        "output_shape": list(call.output_shape),
        "tensors": {k: (None if v is None else list(v.shape)) for k, v in call.tensors.items()},
    }
