"""Quantized cases for the weighted operators (FullyConnected, Convolve,
DepthwiseConv, TransposeConv) on the TFLM reference kernels.

One flow for all four, replacing Keras + TFLiteConverter calibration:

1. draw the float input (the operator's own draw) and quantize it with the
   policy's input quantization (explicit descriptor block, else the legacy
   calibration range, else the draw range);
2. draw float weights (Glorot, `weight_gain` scaled) and quantize them
   symmetrically, per output channel unless the descriptor forces per-tensor;
3. run the float reference on the dequantized input and weights to calibrate
   the output range (fused activation applied), widened by the bias margin and
   the `headroom` knob, unless the descriptor gives the output quantization;
4. draw a bias worth 3..8 output steps per channel, in the accumulator scale;
5. derive the requantization and activation clamp through the shim's TFLM code
   and run the quantized reference kernel for the golden.

S4 weights keep the legacy fixed-quantization path the S4 descriptors were
written for (`hint.extras`: input_scale 4.0, input_zero_point 3,
weight_scale 1.0, output_scale 4.0, output_zero_point 0 by default), with
integer weights and biases drawn directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from helia_core_tester.generation.reference import bindings as b
from helia_core_tester.generation.reference import draw, params, policy
from helia_core_tester.generation.reference.case import ReferenceCall
from helia_core_tester.generation.reference.run import run_reference

DEFAULT_DRAW_RANGE = (-32.0, 32.0)
_S4_DEFAULTS = {"input_scale": 4.0, "input_zero_point": 3, "weight_scale": 1.0, "output_scale": 4.0, "output_zero_point": 0}

# Keys of the descriptor `quantization:` block.
QUANT_ROLES = ("input", "weights", "output")


class ReferenceCaseError(ValueError):
    """A descriptor cannot be expressed as a reference-kernel case."""


@dataclass
class WeightedCase:
    """Everything a generator needs to render one quantized weighted case."""

    kernel: str
    input_q: np.ndarray
    weights_q: np.ndarray  # kernel layout, unpacked int8 (s4 values in -8..7)
    weights_c: np.ndarray  # what the C harness stores (packed for s4)
    bias_q: Optional[np.ndarray]
    output_q: np.ndarray
    input_quant: policy.TensorQuant
    weight_quant: policy.WeightQuant
    output_quant: policy.TensorQuant
    requant: b.PerChannel
    act_min: int
    act_max: int
    weights_offset: int
    call: ReferenceCall
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def per_channel(self) -> bool:
        return self.requant.multiplier.size > 1 or self.weight_quant.per_channel

    def quant_context(self, builder) -> Dict[str, Any]:
        """The `quant_params` dict the templates read."""
        if self.per_channel:
            return {
                "multiplier_array": builder.format_array_as_c_literal(self.requant.multiplier),
                "shift_array": builder.format_array_as_c_literal(self.requant.shift),
                "multiplier": [int(m) for m in self.requant.multiplier],
                "shift": [int(s) for s in self.requant.shift],
                "per_channel": True,
            }
        return {"multiplier": int(self.requant.multiplier[0]), "shift": int(self.requant.shift[0]), "per_channel": False}


@dataclass(frozen=True)
class WeightedSpec:
    """Operator-specific geometry, given to the shared flow."""

    family: str  # conv | dwconv | fc | tconv
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    # Float weights in the kernel layout, and the axis of the output channel.
    weight_shape: Tuple[int, ...]
    channel_axis: int
    fan_in: int
    fan_out: int
    params: Mapping[str, Any]  # geometry keys of the ReferenceCall params (stride, pad, ...)
    per_channel_default: bool = True


def dtype_key(dtype: str) -> str:
    key = str(dtype).lower()
    if key not in ("s8", "s16"):
        raise ReferenceCaseError(f"reference-kernel cases are s8/s16 activations, got {dtype!r}")
    return key


def quantization_block(desc: Mapping[str, Any]) -> Dict[str, Any]:
    block = desc.get("quantization") or {}
    if not isinstance(block, Mapping):
        raise ReferenceCaseError(f"{desc.get('name')}: quantization must be a mapping")
    unknown = set(block) - set(QUANT_ROLES) - {"headroom", "input_2"}
    if unknown:
        raise ReferenceCaseError(f"{desc.get('name')}: unknown quantization keys {sorted(unknown)}")
    return dict(block)


def _legacy_extras(desc: Mapping[str, Any]) -> Dict[str, Any]:
    hint = desc.get("hint") or {}
    extras = hint.get("extras") if isinstance(hint, Mapping) else None
    return dict(extras or {})


def _check_exclusive(desc: Mapping[str, Any], block: Mapping[str, Any]) -> None:
    """An explicit quantization block and the legacy knobs for the same role
    would silently disagree: refuse the descriptor instead."""
    extras = _legacy_extras(desc)
    clashes = []
    if "input" in block and ("calibration_range" in desc or {"input_scale", "input_zero_point"} & set(extras)):
        clashes.append("input")
    if "output" in block and {"output_scale", "output_zero_point"} & set(extras):
        clashes.append("output")
    if "weights" in block and ({"weight_scale", "per_channel"} & set(extras) or (desc.get("hint") or {}).get("force_per_tensor")):
        clashes.append("weights")
    if clashes:
        raise ReferenceCaseError(
            f"{desc.get('name')}: quantization.{'/'.join(clashes)} and the legacy hint/calibration knobs "
            "both set it; keep one"
        )


def input_quant(desc: Mapping[str, Any], dtype: str, block: Mapping[str, Any]) -> policy.TensorQuant:
    explicit = policy.descriptor_quant(block.get("input"), dtype)
    if explicit is not None:
        return explicit
    for key in ("calibration_range", "input_range"):
        if desc.get(key):
            lo, hi = desc[key]
            return policy.activation_quant(np.zeros(1), dtype, (float(lo), float(hi)))
    return policy.activation_quant(np.zeros(1), dtype, DEFAULT_DRAW_RANGE)


def _per_channel(desc: Mapping[str, Any], block: Mapping[str, Any], default: bool) -> bool:
    weights = block.get("weights") or {}
    if "per_channel" in weights:
        return bool(weights["per_channel"])
    hint = desc.get("hint") or {}
    if isinstance(hint, Mapping) and hint.get("force_per_tensor"):
        return False
    return default


def _activation_bounds(desc: Mapping[str, Any], out_q: policy.TensorQuant, dtype: str) -> Tuple[int, int]:
    """Fused activation range, narrowed by any descriptor activation_min/max."""
    activation = str(desc.get("activation", "NONE")).upper()
    if activation not in params.ACTIVATIONS:
        raise ReferenceCaseError(f"{desc.get('name')}: fused activation {activation} has no reference")
    lo, hi = params.activation_range(activation, out_q.scale, out_q.zero_point, dtype)
    if "activation_min" in desc:
        lo = max(lo, int(desc["activation_min"]))
    if "activation_max" in desc:
        hi = min(hi, int(desc["activation_max"]))
    if lo > hi:
        raise ReferenceCaseError(f"{desc.get('name')}: activation range [{lo}, {hi}] is empty")
    return lo, hi


def _float_call(spec: WeightedSpec, x: np.ndarray, w: np.ndarray, bias: Optional[np.ndarray], fmin: float, fmax: float) -> np.ndarray:
    p = dict(spec.params)
    p["act"] = {"min": 0, "max": 0, "fmin": fmin, "fmax": fmax}
    call = ReferenceCall(
        f"{spec.family}_f32",
        p,
        {"input": x.astype(np.float32), "filter": w.astype(np.float32), "bias": None if bias is None else bias.astype(np.float32)},
        spec.output_shape,
        "float32",
    )
    return run_reference(call)


def _balance_clamped_channels(spec: WeightedSpec, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Negate the weights of every output channel whose pre-activation is mostly <= 0.

    Under a fused ReLU those channels collapse onto the clamp (the int8 minimum when the
    output zero point sits there), so a draw where most channels lean negative leaves a
    saturated, uninformative golden. The weights are random draws either way; flipping a
    channel only moves its pre-activations across zero."""
    pre = _float_call(spec, x, w, None, -np.inf, np.inf)
    per_channel = pre.reshape(-1, pre.shape[-1])
    flip = np.mean(per_channel <= 0.0, axis=0) > 0.5
    if not flip.any():
        return w
    shape = [1] * w.ndim
    shape[spec.channel_axis] = -1
    if w.shape[spec.channel_axis] != flip.size:
        raise ReferenceCaseError(f"{w.shape[spec.channel_axis]} weight channels for {flip.size} output channels")
    return np.where(flip.reshape(shape), -w, w).astype(w.dtype)


def _output_quant(
    desc: Mapping[str, Any],
    spec: WeightedSpec,
    dtype: str,
    block: Mapping[str, Any],
    x_dq: np.ndarray,
    w_dq: np.ndarray,
    has_bias: bool,
) -> policy.TensorQuant:
    explicit = policy.descriptor_quant(block.get("output"), dtype)
    if explicit is not None:
        return explicit
    fmin, fmax = params.float_activation_range(str(desc.get("activation", "NONE")))
    reference = _float_call(spec, x_dq, w_dq, None, fmin, fmax)
    lo, hi = float(reference.min()), float(reference.max())
    if dtype == "s16":
        margin = max(abs(lo), abs(hi)) * (draw.BIAS_MAX_STEPS / 32767.0) if has_bias else 0.0
    else:
        margin = (max(hi, 0.0) - min(lo, 0.0)) * (draw.BIAS_MAX_STEPS / 255.0) if has_bias else 0.0
    headroom = float(block.get("headroom", 1.0))
    if headroom < 1.0:
        raise ReferenceCaseError(f"{desc.get('name')}: quantization.headroom must be >= 1")
    lo, hi = (lo - margin) * headroom, (hi + margin) * headroom
    # The fused activation bounds what the kernel can emit; calibrate to it.
    lo, hi = max(lo, fmin), min(hi, fmax)
    if hi <= lo:
        hi = lo + 1e-3
    return policy.activation_quant(np.zeros(1), dtype, (lo, hi))


def _dequantize_weights(w_q: np.ndarray, wq: policy.WeightQuant) -> np.ndarray:
    scales = np.asarray(wq.scales, dtype=np.float64)
    if wq.per_channel:
        shape = [1] * w_q.ndim
        shape[wq.axis] = scales.size
        scales = scales.reshape(shape)
    return (w_q.astype(np.float64) * scales).astype(np.float32)


def build_weighted_case(
    desc: Mapping[str, Any],
    spec: WeightedSpec,
    rng: np.random.Generator,
    draw_input: Callable[[], np.ndarray],
    weights_offset: int = 0,
    bias_ctype: Optional[str] = None,
) -> WeightedCase:
    """Draw, quantize and run one weighted case on the reference kernels.

    bias_ctype is the kernel's bias C type when it differs from the TFLite
    default for the activation dtype (an s16 convolution taking int32 bias)."""
    dtype = dtype_key(desc.get("activation_dtype", "S8"))
    activation = str(desc.get("activation", "NONE")).upper()
    if activation not in params.ACTIVATIONS:
        raise ReferenceCaseError(f"{desc.get('name')}: fused activation {activation} has no reference")
    weight_dtype = str(desc.get("weight_dtype", "S8")).lower()
    if weight_dtype not in ("s8", "s4"):
        raise ReferenceCaseError(f"{desc.get('name')}: weight dtype {weight_dtype} has no reference")
    if weight_dtype == "s4":
        if dtype != "s8":
            raise ReferenceCaseError(f"{desc.get('name')}: s4 weights need s8 activations")
        return _build_s4_case(desc, spec, rng, draw_input)
    block = quantization_block(desc)
    _check_exclusive(desc, block)
    use_bias = bool(desc.get("use_bias", True))
    channels = spec.weight_shape[spec.channel_axis]

    in_q = input_quant(desc, dtype, block)
    x_f = np.asarray(draw_input(), dtype=np.float32).reshape(spec.input_shape)
    x_q = policy.quantize(x_f, in_q)
    x_dq = policy.dequantize(x_q, in_q)

    gain = float(desc.get("weight_gain", 1.0))
    w_f = draw.glorot_uniform(rng, spec.weight_shape, spec.fan_in, spec.fan_out, gain=gain)
    if activation in ("RELU", "RELU6", "RELU_N1_TO_1") and spec.family != "tconv":
        w_f = _balance_clamped_channels(spec, x_dq, w_f)
    explicit_w = block.get("weights") or {}
    wq = policy.weight_quant(w_f, "s8", axis=spec.channel_axis, per_channel=_per_channel(desc, block, spec.per_channel_default))
    if "scale" in explicit_w:
        scales = explicit_w["scale"]
        scales = list(scales) if isinstance(scales, (list, tuple)) else [scales]
        wq = policy.WeightQuant(policy.as_tuple(np.float32(scales).tolist()), "s8", spec.channel_axis)
    w_q = policy.quantize_weights(w_f, wq)
    # Calibrate on the weights the kernel effectively multiplies by: a forced
    # filter offset (FullyConnected force_filter_offset) shifts every weight.
    w_dq = _dequantize_weights(w_q.astype(np.int16) + int(weights_offset), wq)

    out_q = _output_quant(desc, spec, dtype, block, x_dq, w_dq, use_bias)
    bias_np = np.int64 if dtype == "s16" else np.int32
    kernel = f"{spec.family}_{dtype}"
    if bias_ctype == "int32_t" and dtype == "s16":
        if spec.family != "conv":
            raise ReferenceCaseError(f"{desc.get('name')}: no s16 {spec.family} reference with int32 bias")
        bias_np, kernel = np.int32, "conv_s16_b32"
    bias_q = None
    if use_bias:
        acc_scales = policy.bias_quant_scales(in_q.scale, wq, channels)
        bias_q = draw.bias_in_output_steps(rng, channels, out_q.scale, acc_scales, bias_np)
    requant = params.per_channel(in_q.scale, list(wq.scales), out_q.scale)
    act_min, act_max = _activation_bounds(desc, out_q, dtype)

    call = _quantized_call(kernel, spec, x_q, w_q, bias_q, requant, in_q, out_q, act_min, act_max, weights_offset,
                           quant={"input": in_q, "weights": wq, "output": out_q})
    output = run_reference(call)
    return WeightedCase(kernel, x_q, w_q, w_q, bias_q, output, in_q, wq, out_q, requant, act_min, act_max,
                        weights_offset, call)


def _quantized_call(kernel, spec, x_q, w_q, bias_q, requant, in_q, out_q, act_min, act_max, weights_offset, quant, filter_shape=None, w_store=None):
    p = dict(spec.params)
    p.update(
        input_offset=-in_q.zero_point,
        output_offset=out_q.zero_point,
        act={"min": int(act_min), "max": int(act_max)},
        multiplier=requant.multiplier,
        shift=requant.shift,
    )
    if spec.family == "fc":
        p["weights_offset"] = int(weights_offset)
    if filter_shape is not None:
        p["filter_shape"] = list(filter_shape)
    return ReferenceCall(
        kernel,
        p,
        {"input": x_q, "filter": w_q if w_store is None else w_store, "bias": bias_q},
        spec.output_shape,
        "int16" if "_s16" in kernel else "int8",
        quant=quant,
    )


def _s4_scales(extras: Mapping[str, Any], channels: int, per_channel: bool) -> Tuple[float, ...]:
    raw = extras.get("weight_scale", _S4_DEFAULTS["weight_scale"])
    values = [float(v) for v in raw] if isinstance(raw, (list, tuple)) else [float(raw)]
    if per_channel:
        if len(values) == 1:
            values = values * channels
        if len(values) != channels:
            raise ReferenceCaseError(f"{len(values)} weight scales for {channels} channels")
    elif len(values) != 1:
        raise ReferenceCaseError("per-tensor s4 weights take one weight_scale")
    return policy.as_tuple(np.float32(values).tolist())


def _build_s4_case(desc: Mapping[str, Any], spec: WeightedSpec, rng: np.random.Generator, draw_input) -> WeightedCase:
    """Fixed quantization from hint.extras, integer weights in [-8, 7] and an
    integer bias in [-128, 127], as the S4 descriptors were written."""
    if desc.get("quantization"):
        raise ReferenceCaseError(f"{desc.get('name')}: s4 cases take their quantization from hint.extras")
    extras = {**_S4_DEFAULTS, **_legacy_extras(desc)}
    channels = spec.weight_shape[spec.channel_axis]
    # arm_fully_connected_s4 takes per-tensor quantization; conv/depthwise s4 are per-channel.
    per_channel = bool(extras.get("per_channel", spec.per_channel_default and spec.family != "fc"))
    in_q = policy.TensorQuant(float(np.float32(extras["input_scale"])), int(extras["input_zero_point"]), "s8")
    scales = _s4_scales(extras, channels, per_channel)
    wq = policy.WeightQuant(scales, "s4", spec.channel_axis)
    output_scale = float(extras["output_scale"])
    if spec.family == "fc" and len(scales) == 1:
        # The FC s4 builder pinned the output scale to input * weight scale.
        output_scale = in_q.scale * scales[0]
    out_q = policy.TensorQuant(float(np.float32(output_scale)), int(extras["output_zero_point"]), "s8")

    x_f = np.asarray(draw_input(), dtype=np.float32).reshape(spec.input_shape)
    x_q = policy.quantize(x_f, in_q)
    w_q = rng.integers(-8, 8, size=spec.weight_shape).astype(np.int8)
    bias_q = rng.integers(-128, 128, size=(channels,)).astype(np.int32) if desc.get("use_bias", True) else None
    requant = params.per_channel(in_q.scale, list(scales), out_q.scale)
    act_min, act_max = _activation_bounds(desc, out_q, "s8")
    packed = policy.pack_int4(w_q)
    kernel = f"{spec.family}_s4"
    call = _quantized_call(kernel, spec, x_q, w_q, bias_q, requant, in_q, out_q, act_min, act_max, 0,
                           quant={"input": in_q, "weights": wq, "output": out_q},
                           filter_shape=spec.weight_shape, w_store=packed)
    output = run_reference(call)
    return WeightedCase(kernel, x_q, w_q, packed, bias_q, output, in_q, wq, out_q, requant, act_min, act_max, 0, call)


# ---- operator geometry ----


def _pair(value: Any, default: Tuple[int, int], name: str) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        pair = (int(value), int(value))
    else:
        items = [int(v) for v in value]
        if len(items) != 2:
            raise ReferenceCaseError(f"{name} must be one integer or [h, w], got {value!r}")
        pair = (items[0], items[1])
    if pair[0] < 1 or pair[1] < 1:
        raise ReferenceCaseError(f"{name} must be positive, got {value!r}")
    return pair


def _shape4(desc: Mapping[str, Any], key: str) -> Tuple[int, int, int, int]:
    shape = tuple(int(d) for d in desc[key])
    if len(shape) != 4 or any(d < 1 for d in shape):
        raise ReferenceCaseError(f"{desc.get('name')}: {key} must be a positive rank-4 shape, got {list(shape)}")
    return shape  # type: ignore[return-value]


def _padding(desc: Mapping[str, Any]) -> str:
    padding = str(desc.get("padding") or "valid").upper()
    if padding not in ("SAME", "VALID"):
        raise ReferenceCaseError(f"{desc.get('name')}: padding must be same or valid, got {desc.get('padding')!r}")
    return padding


def _window_params(stride, dilation, ph, pw) -> Dict[str, Any]:
    return {"stride": list(stride), "dilation": list(dilation), "pad": [ph.pad, pw.pad], "pad_offset": [ph.offset, pw.offset]}


def conv_spec(desc: Mapping[str, Any]) -> WeightedSpec:
    """Convolve: descriptor filter_shape is HWIO [kh, kw, in, out]; groups from `groups`."""
    n, h, w, c = _shape4(desc, "input_shape")
    kh, kw, _, cout = _shape4(desc, "filter_shape")
    groups = int(desc.get("groups", 1))
    if groups < 1 or c % groups or cout % groups:
        raise ReferenceCaseError(f"{desc.get('name')}: {c} input / {cout} output channels do not split into {groups} groups")
    cin = c // groups
    stride = _pair(desc.get("strides"), (1, 1), "strides")
    dilation = _pair(desc.get("dilation"), (1, 1), "dilation")
    try:
        (oh, ow), ph, pw = params.conv_geometry(_padding(desc), (h, w), (kh, kw), stride, dilation)
    except ValueError as exc:
        raise ReferenceCaseError(f"{desc.get('name')}: {exc}") from exc
    return WeightedSpec(
        family="conv",
        input_shape=(n, h, w, c),
        output_shape=(n, oh, ow, cout),
        weight_shape=(cout, kh, kw, cin),
        channel_axis=0,
        fan_in=kh * kw * cin,
        fan_out=kh * kw * cout,
        params=_window_params(stride, dilation, ph, pw),
    )


def dwconv_spec(desc: Mapping[str, Any]) -> WeightedSpec:
    """DepthwiseConv: filter 1HWC with C = in_channels * depth_multiplier."""
    n, h, w, c = _shape4(desc, "input_shape")
    fs = [int(d) for d in desc["filter_shape"]]
    if len(fs) < 2 or fs[0] < 1 or fs[1] < 1:
        raise ReferenceCaseError(f"{desc.get('name')}: filter_shape must start [kh, kw], got {fs}")
    kh, kw = fs[0], fs[1]
    mult = int(desc.get("depth_multiplier", 1))
    if mult < 1:
        raise ReferenceCaseError(f"{desc.get('name')}: depth_multiplier must be positive")
    stride = _pair(desc.get("strides"), (1, 1), "strides")
    dilation = _pair(desc.get("dilation"), (1, 1), "dilation")
    try:
        (oh, ow), ph, pw = params.conv_geometry(_padding(desc), (h, w), (kh, kw), stride, dilation)
    except ValueError as exc:
        raise ReferenceCaseError(f"{desc.get('name')}: {exc}") from exc
    return WeightedSpec(
        family="dwconv",
        input_shape=(n, h, w, c),
        output_shape=(n, oh, ow, c * mult),
        weight_shape=(1, kh, kw, c * mult),
        channel_axis=3,
        fan_in=kh * kw * c,
        fan_out=kh * kw * mult,
        params={**_window_params(stride, dilation, ph, pw), "depth_multiplier": mult},
    )


def fc_spec(desc: Mapping[str, Any]) -> WeightedSpec:
    """FullyConnected: input [batch, ...] flattened to [batch, features]; weights [out, features]."""
    input_shape = tuple(int(d) for d in desc["input_shape"])
    if len(input_shape) < 2 or any(d < 1 for d in input_shape):
        raise ReferenceCaseError(f"{desc.get('name')}: input_shape must be [batch, ...features], got {list(input_shape)}")
    batch = input_shape[0]
    features = int(np.prod(input_shape[1:]))
    out = int(desc["filter_shape"][0])
    if out < 1:
        raise ReferenceCaseError(f"{desc.get('name')}: filter_shape[0] (output units) must be positive")
    return WeightedSpec(
        family="fc",
        input_shape=input_shape,
        output_shape=(batch, out),
        weight_shape=(out, features),
        channel_axis=0,
        fan_in=features,
        fan_out=out,
        params={},
    )


def tconv_spec(desc: Mapping[str, Any]) -> WeightedSpec:
    """TransposeConv: descriptor filter_shape [kh, kw, out, in]; Keras output size
    (SAME: in * stride, VALID: in * stride + max(k - stride, 0))."""
    n, h, w, c = _shape4(desc, "input_shape")
    # filter_shape[3] is not read: the in-channels are the input's (as the Keras
    # layer the descriptors were written for took them).
    kh, kw, cout, _ = _shape4(desc, "filter_shape")
    cin = c
    stride = _pair(desc.get("strides"), (1, 1), "strides")
    if _pair(desc.get("dilation"), (1, 1), "dilation") != (1, 1):
        raise ReferenceCaseError(f"{desc.get('name')}: transpose conv has no dilation")
    padding = _padding(desc)
    if padding == "SAME":
        oh, ow = h * stride[0], w * stride[1]
    else:
        oh, ow = h * stride[0] + max(kh - stride[0], 0), w * stride[1] + max(kw - stride[1], 0)
    ph, pw = params.tconv_padding(padding, (oh, ow), (kh, kw), stride)
    return WeightedSpec(
        family="tconv",
        input_shape=(n, h, w, c),
        output_shape=(n, oh, ow, cout),
        weight_shape=(cout, kh, kw, cin),
        channel_axis=0,
        fan_in=kh * kw * cout,
        fan_out=kh * kw * cin,
        params=_window_params(stride, (1, 1), ph, pw),
    )
