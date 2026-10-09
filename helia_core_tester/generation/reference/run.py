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

Tensors: input, filter, bias (bias may be absent or None).
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


def _tensor(tensors: Mapping[str, Optional[np.ndarray]], name: str) -> np.ndarray:
    tensor = tensors.get(name)
    if tensor is None:
        raise KeyError(f"missing tensor {name!r}")
    return tensor


def run_reference(call: ReferenceCall, lib: Optional[b.Bindings] = None) -> np.ndarray:
    """Run the call's kernel and return its output (call.output_dtype, call.output_shape)."""
    lib = lib or b.get_bindings()
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
