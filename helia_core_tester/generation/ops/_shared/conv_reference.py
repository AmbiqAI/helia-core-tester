"""Reference case construction for Conv2D (and the shared pieces DepthwiseConv/TransposeConv use):
the draw, the quantization the converter used to produce, and the golden from the C reference."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# Unbounded float activations reach the kernels as +-1e30 (what the harness has always
# passed); the reference clamps at the same value so golden and kernel agree on infinities.
FLOAT_UNBOUNDED = 1.0e30


def _extras(desc) -> Dict[str, Any]:
    hint = desc.get("hint") or {}
    return (hint.get("extras") or {}) if isinstance(hint, dict) else {}


def input_calibration_range(desc) -> Tuple[float, float]:
    """The range the converter calibrated the input over: calibration_range, else input_range, else
    the [-32, 31] integers generate_input_data draws."""
    from helia_core_tester.generation.ops._shared.quant_knobs import value_range

    if desc.get("calibration_range"):
        return value_range(desc, "calibration_range", ())
    if desc.get("input_range"):
        return value_range(desc, "input_range", ())
    return -32.0, 31.0


def float_bounds(op) -> Tuple[float, float]:
    from helia_core_tester.generation.reference.weighted import float_activation

    lo, hi = float_activation(op.activation_name())
    lo = max(lo, float(op.desc.get("activation_min", -FLOAT_UNBOUNDED)))
    hi = min(hi, float(op.desc.get("activation_max", FLOAT_UNBOUNDED)))
    if lo > hi:
        raise ValueError(f"{op.desc.get('name')}: activation_min {lo} > activation_max {hi}")
    # The sentinel stays as written (it narrows to the same binary32 either way) so headers print it as before.
    narrow = lambda v: v if abs(v) == FLOAT_UNBOUNDED else float(np.float32(v))
    return narrow(lo), narrow(hi)


def quantized_bounds(op, kind: str, out_scale: float, out_zp: int) -> Tuple[int, int]:
    """TFLite's fused-activation range on the output quantization, narrowed by any descriptor
    activation_min/activation_max (codes)."""
    from helia_core_tester.generation.reference.abi import activation_code, dtype_code
    from helia_core_tester.generation.reference.bindings import get_bindings

    act = op.activation_name()
    r = get_bindings().prepare("activation_range_quantized", {
        "activation": activation_code(act), "dtype": dtype_code("int8" if kind == "s8" else "int16"),
        "scale": out_scale, "zero_point": out_zp})
    lo = max(r["min"], int(op.desc.get("activation_min", r["min"])))
    hi = min(r["max"], int(op.desc.get("activation_max", r["max"])))
    if lo > hi:
        raise ValueError(f"{op.desc.get('name')}: activation range [{lo}, {hi}] is empty")
    return lo, hi


def per_channel_multipliers(input_scale: float, output_scale: float, filter_scales: np.ndarray):
    from helia_core_tester.generation.reference.bindings import get_bindings

    n = int(np.asarray(filter_scales).size)
    r = get_bindings().run("per_channel_quant", {"input_scale": input_scale, "output_scale": output_scale},
                           {"filter_scale": np.ascontiguousarray(filter_scales, dtype=np.float32)},
                           {"multiplier": (n,), "shift": (n,)})
    return r["multiplier"], r["shift"]


def weight_rng(op) -> np.random.Generator:
    """A stream for weights and bias, independent of the input draw (which uses the case seed)."""
    return np.random.default_rng([int(op.seed), 0x57])


def conv_reference_case(op, kernel_info: Dict[str, Any], float_kernel: bool, float_dtype) -> Dict[str, Any]:
    from helia_core_tester.generation.reference import policy, weighted
    from helia_core_tester.generation.reference.bindings import get_bindings
    from helia_core_tester.generation.reference.call import ReferenceCall
    from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

    desc = op.desc
    name = desc["name"]
    input_shape = tuple(int(d) for d in desc["input_shape"])
    filter_hwio = tuple(int(d) for d in desc["filter_shape"])
    if len(input_shape) != 4 or len(filter_hwio) != 4:
        raise ValueError(f"{name}: Convolve needs NHWC input_shape and HWIO filter_shape")
    n, h, w, in_c = input_shape
    kh, kw, _, out_c = filter_hwio
    groups = int(desc.get("groups", 1))
    if groups < 1 or in_c % groups or out_c % groups:
        raise ValueError(f"{name}: input depth {in_c} and filters {out_c} must divide into {groups} groups")
    filt_c = in_c // groups
    sh, sw = weighted.pair(desc, "strides")
    dh, dw = weighted.pair(desc, "dilation")
    oh, ph = weighted.same_or_valid(desc.get("padding", "valid"), h, kh, sh, dh)
    ow, pw = weighted.same_or_valid(desc.get("padding", "valid"), w, kw, sw, dw)
    output_shape = (n, oh, ow, out_c)
    window = {"stride_h": sh, "stride_w": sw, "dilation_h": dh, "dilation_w": dw, "pad_h": ph, "pad_w": pw}
    use_bias = bool(desc.get("use_bias", True))
    wrng = weight_rng(op)
    filter_shape = (out_c, kh, kw, filt_c)
    fan_in, fan_out = kh * kw * filt_c, kh * kw * out_c
    builder = TemplateContextBuilder()
    lib = get_bindings()

    if float_kernel:
        x = np.asarray(op._sample_uniform(input_shape), dtype=float_dtype)
        weights = weighted.glorot_uniform(wrng, filter_shape, fan_in, fan_out, desc.get("weight_gain")).astype(float_dtype)
        bias = (wrng.uniform(-0.25, 0.25, out_c) if use_bias else np.zeros(0)).astype(float_dtype)
        lo, hi = float_bounds(op)
        entry = "conv_f16" if float_dtype == np.float16 else "conv_f32"
        params = {**window, "activation_min": lo, "activation_max": hi}
        output = op.reference_golden(ReferenceCall(
            entry, params, {"input": np.ascontiguousarray(x), "filter": weights, "bias": bias},
            {"output": output_shape}))
        conv_params = {**window, "input_offset": 0, "output_offset": 0, "activation_min": lo, "activation_max": hi}
        return {"input_shape": input_shape, "output_shape": output_shape, "weights_shape": filter_shape,
                "input": x, "weights": weights, "biases": bias if use_bias else None, "output": output,
                "conv_params": conv_params, "quant_params": None, "filter_operand": weights, "bias_operand": bias}

    kind = "s16" if kernel_info["input_c_type"] == "int16_t" else "s8"
    weight_dtype = str(desc.get("weight_dtype", "S8")).upper()
    extras = _extras(desc) if weight_dtype == "S4" else {}
    x_float = op.generate_input_data()
    if extras:
        # The int4 cases pin the input and filter scales (a hand-built quantized model); the
        # output is calibrated like every other case, so a long accumulation cannot saturate it.
        if "output_scale" in extras or "output_zero_point" in extras:
            raise ValueError(f"{name}: an int4 case's output quantization is calibrated, not pinned in extras")
        in_quant = policy.TensorQuant(float(np.float32(extras.get("input_scale", 4.0))),
                                      int(extras.get("input_zero_point", 3)), kind)
        out_quant = None
        w_scale = extras.get("weight_scale", 1.0)
        values = wrng.integers(-8, 8, size=filter_shape).astype(np.int8)
        scales = np.asarray(w_scale if isinstance(w_scale, (list, tuple)) else [w_scale] * out_c, dtype=np.float32)
        if not bool(extras.get("per_channel", True)):
            scales = np.full(out_c, scales[0], dtype=np.float32)
        qfilter = weighted.QuantizedFilter(values, scales, "S4")
        bias_q = (wrng.integers(-128, 128, size=out_c).astype(np.int64 if kind == "s16" else np.int32)
                  if use_bias else None)
    else:
        in_quant = op.activation_quant("input", input_calibration_range(desc), kind)
        weights = weighted.glorot_uniform(wrng, filter_shape, fan_in, fan_out, desc.get("weight_gain"))
        qfilter = weighted.quantize_filter(weights, "S4" if weight_dtype == "S4" else "S8", channel_axis=0)
        bias_q = (weighted.quantize_bias(weighted.signed_magnitude(wrng, out_c), in_quant.scale, qfilter.scales,
                                         wide=kind == "s16") if use_bias else None)
        out_quant = None
    if bias_q is None and kind == "s16":
        # TFLite's int16 conv always carries a bias tensor (the converter writes zeros), and the
        # s16 wrapper reads its bias struct unconditionally.
        bias_q = np.zeros(out_c, np.int64)
    x_q = policy.quantize(x_float, in_quant)

    if out_quant is None:
        # Calibrate the output as the converter would: the float conv of the dequantized
        # operands with the fused activation, over this draw.
        f_lo, f_hi = float_bounds(op)
        real_bias = (bias_q.astype(np.float64) * in_quant.scale * qfilter.scales.astype(np.float64)
                     if bias_q is not None else np.zeros(0))
        real = lib.run("conv_f32", {**window, "activation_min": f_lo, "activation_max": f_hi},
                       {"input": policy.dequantize(x_q, in_quant).astype(np.float32),
                        "filter": qfilter.dequantized().astype(np.float32),
                        "bias": real_bias.astype(np.float32)}, {"output": output_shape})["output"]
        out_quant = op.activation_quant("output", policy.data_range(real), kind)

    multiplier, shift = per_channel_multipliers(in_quant.scale, out_quant.scale, qfilter.scales)
    act_min, act_max = quantized_bounds(op, kind, out_quant.scale, out_quant.zero_point)
    params = {**window, "input_offset": -in_quant.zero_point, "output_offset": out_quant.zero_point,
              "activation_min": act_min, "activation_max": act_max}
    bias_operand = bias_q if bias_q is not None else np.zeros(0, np.int64 if kind == "s16" else np.int32)
    output = op.reference_golden(ReferenceCall(
        f"conv_{kind}", params,
        {"input": np.ascontiguousarray(x_q), "filter": np.ascontiguousarray(qfilter.values), "bias": bias_operand,
         "multiplier": multiplier, "shift": shift},
        {"output": output_shape},
        quant={"input": in_quant.to_json(), "output": out_quant.to_json(),
               "filter_scales": [float(v) for v in qfilter.scales]}))
    weights_out = weighted.pack_int4(qfilter.values) if weight_dtype == "S4" else qfilter.values
    quant_params = {
        "per_channel": True,
        "multiplier_array": builder.format_array_as_c_literal(multiplier),
        "shift_array": builder.format_array_as_c_literal(shift),
    }
    return {"input_shape": input_shape, "output_shape": output_shape, "weights_shape": filter_shape,
            "input": x_q, "weights": weights_out, "biases": bias_q, "output": output, "conv_params": params,
            "quant_params": quant_params, "filter_operand": qfilter.values, "bias_operand": bias_operand}


def _uniform_input(op, value_range: Tuple[float, float]) -> np.ndarray:
    shape = tuple(int(d) for d in op.desc["input_shape"])
    return op._seeded_rng().uniform(value_range[0], value_range[1], size=shape).astype(np.float32)


def depthwise_reference_case(op, kernel_info: Dict[str, Any], float_kernel: bool, float_dtype) -> Dict[str, Any]:
    """DepthwiseConv2D: filter 1HWO with O = input depth * depth_multiplier (channel i*M + m)."""
    from helia_core_tester.generation.ops._shared.quant_knobs import value_range
    from helia_core_tester.generation.reference import policy, weighted
    from helia_core_tester.generation.reference.bindings import get_bindings
    from helia_core_tester.generation.reference.call import ReferenceCall
    from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

    desc = op.desc
    name = desc["name"]
    input_shape = tuple(int(d) for d in desc["input_shape"])
    fs = tuple(int(d) for d in desc["filter_shape"])
    if len(input_shape) != 4 or len(fs) != 4:
        raise ValueError(f"{name}: DepthwiseConv needs NHWC input_shape and HWIM filter_shape")
    n, h, w, in_c = input_shape
    kh, kw = fs[0], fs[1]
    mult = int(desc.get("depth_multiplier", 1))
    if mult < 1:
        raise ValueError(f"{name}: depth_multiplier must be positive, got {mult}")
    out_c = in_c * mult
    sh, sw = weighted.pair(desc, "strides")
    dh, dw = weighted.pair(desc, "dilation")
    oh, ph = weighted.same_or_valid(desc.get("padding", "valid"), h, kh, sh, dh)
    ow, pw = weighted.same_or_valid(desc.get("padding", "valid"), w, kw, sw, dw)
    output_shape = (n, oh, ow, out_c)
    window = {"stride_h": sh, "stride_w": sw, "dilation_h": dh, "dilation_w": dw, "pad_h": ph, "pad_w": pw}
    use_bias = bool(desc.get("use_bias", True))
    wrng = weight_rng(op)
    filter_shape = (1, kh, kw, out_c)
    builder = TemplateContextBuilder()
    lib = get_bindings()

    def draw_float_filter():
        hwim = weighted.glorot_uniform(wrng, (kh, kw, in_c, mult), kh * kw * in_c, kh * kw * mult,
                                       desc.get("weight_gain"))
        return hwim.reshape(filter_shape)

    if float_kernel:
        x = np.asarray(op._sample_uniform(input_shape), dtype=float_dtype)
        weights = draw_float_filter().astype(float_dtype)
        bias = (wrng.uniform(-1.0, 1.0, out_c) if use_bias else np.zeros(0)).astype(float_dtype)
        lo, hi = float_bounds(op)
        entry = "depthwise_conv_f16" if float_dtype == np.float16 else "depthwise_conv_f32"
        output = op.reference_golden(ReferenceCall(
            entry, {**window, "activation_min": lo, "activation_max": hi},
            {"input": np.ascontiguousarray(x), "filter": weights, "bias": bias}, {"output": output_shape}))
        params = {**window, "input_offset": 0, "output_offset": 0, "activation_min": lo, "activation_max": hi}
        return {"input_shape": input_shape, "output_shape": output_shape, "depth_multiplier": mult,
                "input": x, "weights": weights, "biases": bias if use_bias else None, "output": output,
                "params": params, "quant_params": None, "filter_operand": weights, "bias_operand": bias}

    kind = "s16" if kernel_info["input_c_type"] == "int16_t" else "s8"
    weight_dtype = str(desc.get("weight_dtype", "S8")).upper()
    extras = _extras(desc) if weight_dtype == "S4" else {}
    in_range = (value_range(desc, "calibration_range", ()) if desc.get("calibration_range")
                else value_range(desc, "input_range", ()) if desc.get("input_range") else (-1.0, 1.0))
    x_float = _uniform_input(op, in_range)
    if extras:
        if "output_scale" in extras or "output_zero_point" in extras:
            raise ValueError(f"{name}: an int4 case's output quantization is calibrated, not pinned in extras")
        in_quant = policy.TensorQuant(float(np.float32(extras.get("input_scale", 4.0))),
                                      int(extras.get("input_zero_point", 3)), kind)
        w_scale = extras.get("weight_scale", 1.0)
        scales = np.asarray(w_scale if isinstance(w_scale, (list, tuple)) else [w_scale] * out_c, dtype=np.float32)
        qfilter = weighted.QuantizedFilter(wrng.integers(-8, 8, size=filter_shape).astype(np.int8), scales, "S4")
        bias_q = (wrng.integers(-128, 128, size=out_c).astype(np.int64 if kind == "s16" else np.int32)
                  if use_bias else None)
        x_float = op.generate_input_data()
    else:
        in_quant = op.activation_quant("input", in_range, kind)
        qfilter = weighted.quantize_filter(draw_float_filter(), "S4" if weight_dtype == "S4" else "S8",
                                           channel_axis=3)
        bias_q = (weighted.quantize_bias(weighted.signed_magnitude(wrng, out_c), in_quant.scale, qfilter.scales,
                                         wide=kind == "s16") if use_bias else None)
    if bias_q is None and kind == "s16":
        bias_q = np.zeros(out_c, np.int64)
    x_q = policy.quantize(x_float, in_quant)

    f_lo, f_hi = float_bounds(op)
    filt_real = qfilter.values.astype(np.float64) * qfilter.scales.astype(np.float64).reshape(1, 1, 1, -1)
    real_bias = (bias_q.astype(np.float64) * in_quant.scale * qfilter.scales.astype(np.float64)
                 if bias_q is not None else np.zeros(0))
    real = lib.run("depthwise_conv_f32", {**window, "activation_min": f_lo, "activation_max": f_hi},
                   {"input": policy.dequantize(x_q, in_quant).astype(np.float32),
                    "filter": filt_real.astype(np.float32), "bias": real_bias.astype(np.float32)},
                   {"output": output_shape})["output"]
    out_quant = op.activation_quant("output", policy.data_range(real), kind)

    multiplier, shift = per_channel_multipliers(in_quant.scale, out_quant.scale, qfilter.scales)
    act_min, act_max = quantized_bounds(op, kind, out_quant.scale, out_quant.zero_point)
    params = {**window, "input_offset": -in_quant.zero_point, "output_offset": out_quant.zero_point,
              "activation_min": act_min, "activation_max": act_max}
    bias_operand = bias_q if bias_q is not None else np.zeros(0, np.int32)
    output = op.reference_golden(ReferenceCall(
        f"depthwise_conv_{kind}", params,
        {"input": np.ascontiguousarray(x_q), "filter": np.ascontiguousarray(qfilter.values), "bias": bias_operand,
         "multiplier": multiplier, "shift": shift},
        {"output": output_shape},
        quant={"input": in_quant.to_json(), "output": out_quant.to_json(),
               "filter_scales": [float(v) for v in qfilter.scales]}))
    quant_params = {
        "per_channel": True,
        "multiplier_array": builder.format_array_as_c_literal(multiplier),
        "shift_array": builder.format_array_as_c_literal(shift),
    }
    weights_out = weighted.pack_int4(qfilter.values) if weight_dtype == "S4" else qfilter.values
    return {"input_shape": input_shape, "output_shape": output_shape, "depth_multiplier": mult,
            "input": x_q, "input_zero_point": in_quant.zero_point, "weights": weights_out,
            "weights_unpacked": qfilter.values, "biases": bias_q, "output": output, "params": params,
            "quant_params": quant_params, "filter_operand": qfilter.values, "bias_operand": bias_operand}


def transpose_conv_reference_case(op, kernel_info: Dict[str, Any], float_kernel: bool, float_dtype) -> Dict[str, Any]:
    """TransposeConv2D: filter OHWI (descriptor HWOI), Keras Conv2DTranspose output size."""
    from helia_core_tester.generation.ops._shared.quant_knobs import value_range
    from helia_core_tester.generation.reference import policy, weighted
    from helia_core_tester.generation.reference.bindings import get_bindings
    from helia_core_tester.generation.reference.call import ReferenceCall
    from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

    desc = op.desc
    name = desc["name"]
    input_shape = tuple(int(d) for d in desc["input_shape"])
    fs = tuple(int(d) for d in desc["filter_shape"])  # KH, KW, OutCh, InCh
    if len(input_shape) != 4 or len(fs) != 4:
        raise ValueError(f"{name}: TransposeConv needs NHWC input_shape and [KH, KW, Out, In] filter_shape")
    n, h, w, in_c = input_shape
    # Keras read only the kernel size and the filter count; the input depth comes from the input.
    kh, kw, out_c = fs[0], fs[1], fs[2]
    if weighted.pair(desc, "dilation") != (1, 1):
        raise ValueError(f"{name}: TransposeConv has no dilation")
    sh, sw = weighted.pair(desc, "strides")
    padding = str(desc.get("padding", "valid")).lower()
    if padding == "same":
        oh, ow = h * sh, w * sw
    elif padding == "valid":
        oh, ow = (h - 1) * sh + kh, (w - 1) * sw + kw
    else:
        raise ValueError(f"{name}: unknown padding {padding!r}")
    output_shape = (n, oh, ow, out_c)
    builder = TemplateContextBuilder()
    geometry = builder.build_transpose_conv_params(desc, input_shape, (kh, kw), output_shape, {}, {})
    window = {"stride_h": sh, "stride_w": sw, "dilation_h": 1, "dilation_w": 1,
              "pad_h": int(geometry["pad_h"]), "pad_w": int(geometry["pad_w"])}
    use_bias = bool(desc.get("use_bias", True))
    wrng = weight_rng(op)
    filter_shape = (out_c, kh, kw, in_c)
    # Keras Conv2DTranspose: kernel (kh, kw, out, in), fan_in = kh*kw*out, fan_out = kh*kw*in.
    draw = lambda: weighted.glorot_uniform(wrng, filter_shape, kh * kw * out_c, kh * kw * in_c,  # noqa: E731
                                           desc.get("weight_gain"))
    lib = get_bindings()

    if float_kernel:
        x = np.asarray(op._sample_uniform(input_shape), dtype=float_dtype)
        weights = draw().astype(float_dtype)
        bias = (wrng.uniform(-0.5, 0.5, out_c) if use_bias else np.zeros(0)).astype(float_dtype)
        lo, hi = float_bounds(op)
        entry = "transpose_conv_f16" if float_dtype == np.float16 else "transpose_conv_f32"
        output = op.reference_golden(ReferenceCall(
            entry, {**window, "activation_min": lo, "activation_max": hi},
            {"input": np.ascontiguousarray(x), "filter": weights, "bias": bias}, {"output": output_shape}))
        params = {**geometry, "input_offset": 0, "output_offset": 0, "activation_min": lo, "activation_max": hi}
        return {"input_shape": input_shape, "output_shape": output_shape, "input": x, "weights": weights,
                "biases": bias if use_bias else None, "output": output, "params": params, "quant_params": None,
                "filter_operand": weights, "bias_operand": bias}

    kind = "s16" if kernel_info["input_c_type"] == "int16_t" else "s8"
    in_range = (value_range(desc, "calibration_range", ()) if desc.get("calibration_range")
                else value_range(desc, "input_range", ()) if desc.get("input_range") else (-8.0, 8.0))
    in_quant = op.activation_quant("input", in_range, kind)
    x_q = policy.quantize(_uniform_input(op, in_range), in_quant)
    qfilter = weighted.quantize_filter(draw(), "S8", channel_axis=0)
    bias_q = (weighted.quantize_bias(wrng.uniform(-0.5, 0.5, out_c), in_quant.scale, qfilter.scales,
                                     wide=kind == "s16") if use_bias else None)
    if bias_q is None and kind == "s16":
        bias_q = np.zeros(out_c, np.int64)
    f_lo, f_hi = float_bounds(op)
    real_bias = (bias_q.astype(np.float64) * in_quant.scale * qfilter.scales.astype(np.float64)
                 if bias_q is not None else np.zeros(0))
    real = lib.run("transpose_conv_f32", {**window, "activation_min": f_lo, "activation_max": f_hi},
                   {"input": policy.dequantize(x_q, in_quant).astype(np.float32),
                    "filter": qfilter.dequantized().astype(np.float32), "bias": real_bias.astype(np.float32)},
                   {"output": output_shape})["output"]
    out_quant = op.activation_quant("output", policy.data_range(real), kind)
    multiplier, shift = per_channel_multipliers(in_quant.scale, out_quant.scale, qfilter.scales)
    act_min, act_max = quantized_bounds(op, kind, out_quant.scale, out_quant.zero_point)
    ref_params = {**window, "input_offset": -in_quant.zero_point, "output_offset": out_quant.zero_point,
                  "activation_min": act_min, "activation_max": act_max}
    bias_operand = bias_q if bias_q is not None else np.zeros(0, np.int32)
    output = op.reference_golden(ReferenceCall(
        f"transpose_conv_{kind}", ref_params,
        {"input": np.ascontiguousarray(x_q), "filter": np.ascontiguousarray(qfilter.values), "bias": bias_operand,
         "multiplier": multiplier, "shift": shift},
        {"output": output_shape},
        quant={"input": in_quant.to_json(), "output": out_quant.to_json(),
               "filter_scales": [float(v) for v in qfilter.scales]}))
    params = {**geometry, "input_offset": -in_quant.zero_point, "output_offset": out_quant.zero_point,
              "activation_min": act_min, "activation_max": act_max}
    quant_params = {"per_channel": True, "multiplier_array": builder.format_array_as_c_literal(multiplier),
                    "shift_array": builder.format_array_as_c_literal(shift)}
    return {"input_shape": input_shape, "output_shape": output_shape, "input": x_q, "weights": qfilter.values,
            "biases": bias_q, "output": output, "params": params, "quant_params": quant_params,
            "filter_operand": qfilter.values, "bias_operand": bias_operand}
