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


def operand_magnitude(call) -> np.ndarray:
    """sum|terms| per output of an exact accumulating entry: `call` re-run on its operands' absolute values with
    the activation clamp off (+-inf, so a lane a nonfinite token reaches stays nonfinite and linear_bound skips
    it). Evaluated here, never recorded as the case's reference."""
    from dataclasses import replace

    if "activation_min" not in call.params or "activation_max" not in call.params:
        raise ValueError(f"{call.entry}: the magnitude run needs an entry with an activation clamp")
    inputs = {k: np.abs(v) if np.issubdtype(v.dtype, np.floating) else v for k, v in call.inputs.items()}
    params = {**call.params, "activation_min": float("-inf"), "activation_max": float("inf")}
    return replace(call, params=params, inputs=inputs).output()


def window_taps(size: int, kernel: int, stride: int, pad: int, dilation: int, out: int) -> int:
    """The most kernel taps of one axis that land inside the input over the `out` output positions."""
    if min(kernel, stride, dilation) < 1 or size < 0 or out < 0:
        raise ValueError(f"window needs kernel, stride and dilation >= 1, got {kernel}, {stride}, {dilation}")
    pos = (np.arange(out) * stride - pad)[:, None] + dilation * np.arange(kernel)[None, :]
    return int(np.max(((pos >= 0) & (pos < size)).sum(axis=1), initial=0))


def transpose_window_taps(size: int, kernel: int, stride: int, pad: int, out: int) -> int:
    """The most (input, tap) pairs of one axis that a transpose convolution scatters onto one output position."""
    if kernel < 1 or stride < 1 or size < 0 or out < 0:
        raise ValueError(f"transpose window needs kernel and stride >= 1, got {kernel}, {stride}")
    counts = np.zeros(max(out, 0), dtype=np.int64)
    pos = (np.arange(size) * stride - pad)[:, None] + np.arange(kernel)[None, :]
    pos = pos[(pos >= 0) & (pos < out)]
    np.add.at(counts, pos, 1)
    return int(np.max(counts, initial=0))


def exact_comparison(op, call, k: int, dtype) -> Dict[str, Any]:
    """The comparison of an exact accumulating golden: assert_close's defaults widened by kf.linear_bound over
    the k terms each output sums (bias included), at the unit roundoff of the kernel's dtype."""
    from helia_core_tester.generation.ops._shared import kernel_faithful as kf

    return kf.comparison_context(op, dtype, *kf.linear_bound(operand_magnitude(call), k, dtype))


def weight_rng(op) -> np.random.Generator:
    """A stream for weights and bias, independent of the input draw (which uses the case seed)."""
    return np.random.default_rng([int(op.seed), 0x57])


# Entries that are arm_convolve_1_x_n_f16's body; the routers below reach it for 1xN shapes the conv1d k3/k5
# specializations do not claim.
_CONV_1XN_F16_ENTRIES = frozenset(
    f"arm_convolve_1_x_n_{stem}f16{acc}" for stem in ("", "nhwc_", "nhwc_ohwi_", "nhwc_packed_")
    for acc in ("", "_acc16"))
_CONV_F16_ROUTERS = frozenset(
    f"arm_convolve_{stem}f16{acc}" for stem in ("", "nhwc_", "wrapper_") for acc in ("", "_acc16"))


def conv_1xn_f16_route(op, kernel_info: Dict[str, Any], input_shape, filter_shape, output_shape,
                       window: Dict[str, int], groups: int) -> Optional[Dict[str, bool]]:
    """{acc16, packed} when this float16 case runs arm_convolve_1_x_n_f16's body on a Helium target, else None."""
    from helia_core_tester.generation.ops._shared import kernel_faithful as kf

    if not kf.mve_float16_route(op):
        return None
    fn = str(kernel_info.get("kernel_fn", ""))
    n, h, _, in_c = input_shape
    _, kh, kw, filt_c = filter_shape
    is_1xn = (h == 1 and output_shape[1] == 1 and kh == 1 and kw > 1 and window["stride_h"] == 1
              and window["stride_w"] > 0 and window["pad_h"] == 0 and window["dilation_h"] == 1
              and window["dilation_w"] == 1 and groups == 1 and in_c == filt_c)
    if not is_1xn:
        return None
    if fn in _CONV_F16_ROUTERS:
        conv1d = n == 1 and kw in (3, 5) and window["stride_w"] == 1 and window["pad_w"] == 0
        if conv1d:
            return None
    elif fn not in _CONV_1XN_F16_ENTRIES:
        return None
    hint = op.desc.get("hint") or {}
    weight_format = str(hint.get("weight_format", "STANDARD") if isinstance(hint, dict) else "STANDARD").upper()
    return {"acc16": fn.endswith("_acc16"),
            "packed": weight_format in {"NT_N_PACKED", "ARM_NN_WEIGHT_FORMAT_NT_N_PACKED"}}


def conv_1xn_patches(x: np.ndarray, kw: int, stride_w: int, pad_w: int, out_w: int) -> np.ndarray:
    """[N, OW, KW * C] patch rows of a [N, 1, W, C] input, zero outside the row."""
    n, _, w, c = x.shape
    padded = np.zeros((n, w + 2 * pad_w + kw + out_w * stride_w, c), dtype=x.dtype)
    padded[:, pad_w:pad_w + w] = x[:, 0]
    rows = [padded[:, ox * stride_w:ox * stride_w + kw].reshape(n, kw * c) for ox in range(out_w)]
    return np.stack(rows, axis=1)


def _weight_format_packed(op) -> bool:
    hint = op.desc.get("hint") or {}
    weight_format = str(hint.get("weight_format", "STANDARD") if isinstance(hint, dict) else "STANDARD").upper()
    return weight_format in {"NT_N_PACKED", "ARM_NN_WEIGHT_FORMAT_NT_N_PACKED"}


def conv_f16_route(op, kernel_info: Dict[str, Any], input_shape, filter_shape, output_shape,
                   window: Dict[str, int], groups: int) -> Optional[Dict[str, int]]:
    """{route, acc16} of conv_f16_mve when this float16 case runs one of the routes it models on a Helium target:
    the routers' dispatch (arm_convolve_nhwc_f16_body: 1x1, the 1xN kernel, conv1d k5/k3, small C, patch GEMM,
    direct) and the per-route entries, each declining as the kernel does. None for the 1xN kernel (conv_1xn_f16_route
    models it), for an entry that returns an error status, and off Helium."""
    from helia_core_tester.generation.ops._shared import kernel_faithful as kf

    if not kf.mve_float16_route(op):
        return None
    fn = str(kernel_info.get("kernel_fn", ""))
    acc16 = fn.endswith("_acc16")
    stem = fn[:-len("_acc16")] if acc16 else fn
    n, h, w, in_c = input_shape
    out_c, kh, kw, filt_c = filter_shape
    _, oh, ow, _ = output_shape
    sh, sw, ph, pw = window["stride_h"], window["stride_w"], window["pad_h"], window["pad_w"]
    dh, dw = window["dilation_h"], window["dilation_w"]
    if groups != 1 or in_c != filt_c:
        return None
    packed = _weight_format_packed(op)
    is_1x1 = kh == 1 and kw == 1 and ph == 0 and pw == 0

    def is_1d(k: int) -> bool:
        return (n == 1 and h == 1 and oh == 1 and kh == 1 and kw == k and sh == 1 and sw == 1 and ph == 0
                and pw == 0 and dh == 1 and dw == 1)

    is_1xn = h == 1 and oh == 1 and kh == 1 and kw > 1 and sh == 1 and sw > 0 and ph == 0 and dh == 1 and dw == 1
    # arm_nn_conv_f16_is_small_c: every gather/scatter offset within 16 bits.
    small_c = (0 < in_c < 8 and out_c > 0 and ow > 0 and w >= 0 and pw >= 0 and sw > 0 and dw > 0 and kw > 0
               and w + pw + 7 * sw + (kw - 1) * dw <= 0xFFFF // in_c and out_c <= 0xFFFF // 8)
    patch_route = kf.CONV_F16_PATCH_PACKED if packed else kf.CONV_F16_PATCH_NT_T
    direct_route = kf.CONV_F16_DIRECT_PACKED if packed else kf.CONV_F16_DIRECT_OHWI
    conv1d_route = kf.CONV_F16_CONV1D_PACKED if packed else kf.CONV_F16_CONV1D

    def fmt(name: str) -> Optional[bool]:
        """The filter format an `_ohwi_` / `_packed_` entry requires, or None for a format-agnostic one."""
        return True if "_packed_" in name else False if "_ohwi_" in name else None

    route = None
    if stem in {"arm_convolve_f16", "arm_convolve_nhwc_f16", "arm_convolve_wrapper_f16"}:
        if is_1x1:
            route = patch_route
        elif is_1xn and not (is_1d(3) or is_1d(5)):
            return None
        elif is_1d(5) or is_1d(3):
            route = conv1d_route
        elif small_c:
            route = kf.CONV_F16_SMALL_C
        elif out_c >= 8 and oh * ow >= 8:
            # The sizer always reserves at least one patch row here, so the router takes patch GEMM.
            route = patch_route
        else:
            route = direct_route
    elif stem in {"arm_convolve_1x1_f16", "arm_convolve_1x1_nhwc_f16", "arm_convolve_1x1_nhwc_ohwi_f16",
                  "arm_convolve_1x1_nhwc_packed_f16"}:
        # Every 1x1 leg (whole-tensor matmul, strided patch rows, per-pixel matmul) is the same row arithmetic.
        if is_1x1 and fmt(stem) in (None, packed):
            route = patch_route
    elif stem in {f"arm_convolve_1d_k{k}_nhwc_{f}_f16" for k in (3, 5) for f in ("ohwi", "packed")}:
        if fmt(stem) == packed and is_1d(int(stem[len("arm_convolve_1d_k")])):
            route = conv1d_route
    elif stem == "arm_convolve_small_c_nhwc_f16":
        if small_c and not acc16:
            route = kf.CONV_F16_SMALL_C
    elif stem in {"arm_convolve_patch_gemm_nhwc_ohwi_f16", "arm_convolve_patch_gemm_nhwc_packed_f16"}:
        if fmt(stem) == packed:
            route = patch_route
    elif stem in {"arm_convolve_direct_nhwc_ohwi_f16", "arm_convolve_direct_nhwc_packed_f16"}:
        if fmt(stem) == packed:
            route = direct_route
    if route is None:
        return None
    return {"route": route, "acc16": acc16 and route != kf.CONV_F16_SMALL_C}


def conv_patches(x: np.ndarray, kh: int, kw: int, window: Dict[str, int], out_hw) -> np.ndarray:
    """[N, OH * OW, KH * KW * C] zero-padded patch rows (arm_nn_pack_conv_patch_f16's order: ky, kx, channel)."""
    n, h, w, c = x.shape
    oh, ow = out_hw
    rows = np.zeros((n, oh, ow, kh, kw, c), dtype=x.dtype)
    for oy in range(oh):
        for ox in range(ow):
            for ky in range(kh):
                iy = oy * window["stride_h"] - window["pad_h"] + ky * window["dilation_h"]
                for kx in range(kw):
                    ix = ox * window["stride_w"] - window["pad_w"] + kx * window["dilation_w"]
                    if 0 <= iy < h and 0 <= ix < w:
                        rows[:, oy, ox, ky, kx] = x[:, iy, ix]
    return rows.reshape(n, oh * ow, kh * kw * c)


def conv_f16_faithful(op, kernel_info: Dict[str, Any], input_shape, filter_shape, output_shape,
                      window: Dict[str, int], groups: int, activation):
    """A builder (x, weights, bias or None) -> ReferenceCall of the kernel's own float16 arithmetic for this case's
    route on Helium, or None when no modelled route runs it."""
    from helia_core_tester.generation.ops._shared import kernel_faithful as kf

    route = conv_1xn_f16_route(op, kernel_info, input_shape, filter_shape, output_shape, window, groups)
    if route is not None:
        def build_1xn(x, w, b):
            return kf.conv_1xn_f16_mve_call(x, w, b, stride_w=window["stride_w"], pad_w=window["pad_w"],
                                            out_w=output_shape[2], acc16=route["acc16"], packed=route["packed"],
                                            activation=activation)
        return build_1xn
    conv = conv_f16_route(op, kernel_info, input_shape, filter_shape, output_shape, window, groups)
    if conv is None:
        return None

    def build(x, w, b):
        return kf.conv_f16_mve_call(x, w, b, route=conv["route"], window=window, output_shape=output_shape,
                                    acc16=conv["acc16"], activation=activation)
    return build


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
        conv_params = {**window, "input_offset": 0, "output_offset": 0, "activation_min": lo, "activation_max": hi}
        result = {"input_shape": input_shape, "output_shape": output_shape, "weights_shape": filter_shape,
                  "input": x, "weights": weights, "biases": bias if use_bias else None,
                  "conv_params": conv_params, "quant_params": None, "filter_operand": weights, "bias_operand": bias}
        build = (conv_f16_faithful(op, kernel_info, input_shape, filter_shape, output_shape, window, groups, (lo, hi))
                 if float_dtype == np.float16 else None)
        if build is not None:
            from helia_core_tester.generation.ops._shared import kernel_faithful as kf

            # The kernel's own float16 arithmetic on Helium for this case's route, bit for bit.
            b_or_none = bias if use_bias else None
            output = op.reference_golden(build(x, weights, b_or_none))
            patches = conv_patches(x, kh, kw, window, output_shape[1:3])
            kf.check_within_bound(name, output.reshape(n, -1, out_c), patches, np.broadcast_to(
                weights.reshape(1, out_c, -1), (n, out_c, patches.shape[-1])), b_or_none, float_dtype)

            def probe(ops):
                b_op = np.asarray(ops[2]).reshape(-1)
                return build(np.asarray(ops[0]).reshape(x.shape), np.asarray(ops[1]).reshape(weights.shape),
                             b_op if b_op.size else None).output()

            return {**result, "output": output, "comparison": kf.comparison_context(op, float_dtype),
                    "reference_probe": probe}
        entry = "conv_f16" if float_dtype == np.float16 else "conv_f32"
        params = {**window, "activation_min": lo, "activation_max": hi}
        call = ReferenceCall(entry, params, {"input": np.ascontiguousarray(x), "filter": weights, "bias": bias},
                             {"output": output_shape})
        output = op.reference_golden(call)
        # Each output sums the in-range taps of its group's filt_c channels, plus the bias.
        k = (window_taps(h, kh, sh, ph, dh, oh) * window_taps(w, kw, sw, pw, dw, ow) * filt_c + int(use_bias))
        return {**result, "output": output, "comparison": exact_comparison(op, call, k, float_dtype)}

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


# arm_depthwise_conv_f16's routers (dispatch: conv1d k3 / 2x5 specialisations, then ch_mult 1 direct, then the
# one-input-channel to-convolution route when ctx holds it, then generic) and its single-route entries.
_DW_F16_ROUTERS = frozenset(
    f"arm_depthwise_{stem}f16{acc}" for stem in ("conv_", "nhwc_conv_", "conv_wrapper_") for acc in ("", "_acc16"))
_DW_F16_ROUTE_ENTRIES = {
    **{f"arm_depthwise_conv_{r}_nhwc_f16{acc}": r for r in ("direct", "generic", "cin1") for acc in ("", "_acc16")},
    "arm_depthwise_conv_1d_k3_nhwc_f16": "1d_k3",
    "arm_depthwise_conv_2x5_nhwc_f16": "2x5",
}


def depthwise_f16_cin1_ctx_fits(op, kh: int, kw: int, window: Dict[str, int], input_shape, output_shape) -> bool:
    """Whether the ctx the harness passes (arm_depthwise_conv_f16_get_buffer_size's answer, or none under
    force_no_scratch) holds arm_nn_dw_f16_cin1_min_ctx_size: the packed filter plus one patch row. The sizer adds
    arm_convolve_wrapper_f16_get_buffer_size to the packed filter, which is at least one patch row except for a
    1x1 filter with no padding, unit stride and an unchanged spatial size, where it is 0."""
    hint = op.desc.get("hint") or {}
    if isinstance(hint, dict) and hint.get("force_no_scratch"):
        return False
    same_1x1 = (kh == 1 and kw == 1 and window["pad_h"] == 0 and window["pad_w"] == 0 and window["stride_h"] == 1
                and window["stride_w"] == 1 and tuple(output_shape[1:3]) == tuple(input_shape[1:3]))
    return not same_1x1


def depthwise_f16_route(op, kernel_info: Dict[str, Any], input_shape, output_shape, kh: int, kw: int, mult: int,
                        window: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """{route: 'chain' | 'cin1', acc16} for a float16 DepthwiseConv case on a Helium target, else None (a case
    whose entry declines the shape keeps the exact golden: it asserts a status)."""
    from helia_core_tester.generation.ops._shared import kernel_faithful as kf

    if not kf.mve_float16_route(op):
        return None
    fn = str(kernel_info.get("kernel_fn", ""))
    n, h, _, in_c = input_shape
    unit = (window["stride_h"] == 1 and window["stride_w"] == 1 and window["dilation_h"] == 1
            and window["dilation_w"] == 1 and window["pad_h"] == 0 and window["pad_w"] == 0)
    is_1d_k3 = n == 1 and output_shape[0] == 1 and mult == 1 and kw == 3 and kh == 1 and h == 1 and \
        output_shape[1] == 1 and unit
    is_2x5 = kw == 5 and kh == 2 and h == 2 and output_shape[1] == 1 and unit
    # CONVERT_DW_CONV_WITH_ONE_INPUT_CH_AND_OUTPUT_CH_ABOVE_THRESHOLD is 1 under GCC.
    is_cin1 = in_c == 1 and output_shape[3] >= 1 and depthwise_f16_cin1_ctx_fits(op, kh, kw, window, input_shape,
                                                                                  output_shape)
    acc16 = fn.endswith("_acc16")
    if fn in _DW_F16_ROUTERS:
        route = "chain" if (is_1d_k3 or is_2x5 or mult == 1 or not is_cin1) else "cin1"
    elif fn in _DW_F16_ROUTE_ENTRIES:
        r = _DW_F16_ROUTE_ENTRIES[fn]
        gate = {"direct": mult == 1, "generic": True, "cin1": is_cin1, "1d_k3": is_1d_k3, "2x5": is_2x5}[r]
        if not gate:
            return None
        route = "cin1" if r == "cin1" else "chain"
    else:
        return None
    return {"route": route, "acc16": acc16}


def depthwise_taps(x: np.ndarray, kh: int, kw: int, window: Dict[str, int], out_hw) -> np.ndarray:
    """[N, OH, OW, C, KH * KW] input taps of every output position (row by row), zero outside the input."""
    n, h, w, c = x.shape
    oh, ow = out_hw
    ys = (np.arange(oh) * window["stride_h"] - window["pad_h"])[:, None] + window["dilation_h"] * np.arange(kh)
    xs = (np.arange(ow) * window["stride_w"] - window["pad_w"])[:, None] + window["dilation_w"] * np.arange(kw)
    yv = (ys >= 0) & (ys < h)
    xv = (xs >= 0) & (xs < w)
    g = x[:, np.clip(ys, 0, h - 1)][:, :, :, np.clip(xs, 0, w - 1)]  # [N, OH, KH, OW, KW, C]
    mask = yv[:, :, None, None] & xv[None, None]
    g = np.where(mask[None, :, :, :, :, None], g, np.zeros((), x.dtype))
    return np.ascontiguousarray(g.transpose(0, 1, 3, 5, 2, 4).reshape(n, oh, ow, c, kh * kw))


def depthwise_cin1_operands(x: np.ndarray, weights: np.ndarray, kh: int, kw: int, window: Dict[str, int], out_hw):
    """The one-input-channel route's matmul: lhs [N, OH * OW, KH * KW] patch rows, rhs [N, OC, KH * KW]."""
    n = x.shape[0]
    taps = depthwise_taps(x, kh, kw, window, out_hw)
    lhs = taps.reshape(n, out_hw[0] * out_hw[1], kh * kw)
    oc = weights.shape[3]
    rhs = np.broadcast_to(weights.reshape(kh * kw, oc).T[None], (n, oc, kh * kw))
    return lhs, np.ascontiguousarray(rhs)


def depthwise_faithful_case(op, x, weights, bias, kh, kw, mult, window, output_shape, route, activation, dtype):
    """The emulated golden of a float16 DepthwiseConv case on Helium, its bound check, and the nonfinite probe."""
    from helia_core_tester.generation.ops._shared import kernel_faithful as kf

    name = op.desc["name"]
    out_hw = (output_shape[1], output_shape[2])
    if route["route"] == "cin1":
        def call_for(xx, ww, bb):
            lhs, rhs = depthwise_cin1_operands(np.asarray(xx, np.float16), np.asarray(ww, np.float16), kh, kw,
                                               window, out_hw)
            return kf.matmul_nt_n_packed_f16_mve_call(lhs, rhs, bb, policy=kf.NT_N_PACKED_ACC16 if route["acc16"] else kf.NT_N_PACKED_FOLD, activation=activation)
    else:
        def call_for(xx, ww, bb):
            return kf.depthwise_f16_mve_call(xx, ww, bb, window=window, out_hw=out_hw, acc16=route["acc16"],
                                             activation=activation)
    output = op.reference_golden(call_for(x, weights, bias)).reshape(output_shape)

    # Bound: every output is a dot over its KH * KW taps (padded ones zero) plus the bias as one more tap.
    n, oh, ow, oc = output_shape
    taps = depthwise_taps(x, kh, kw, window, out_hw)[:, :, :, np.arange(oc) // mult]  # [N, OH, OW, OC, K]
    lhs = np.concatenate([taps, np.ones(taps.shape[:-1] + (1,), dtype)], axis=-1)
    lhs = lhs.transpose(3, 0, 1, 2, 4).reshape(oc, n * oh * ow, kh * kw + 1)
    b = np.zeros(oc, dtype) if bias is None else np.asarray(bias, dtype)
    rhs = np.concatenate([weights.reshape(kh * kw, oc).T, b[:, None]], axis=-1)[:, None, :]
    with np.errstate(invalid="ignore", over="ignore"):
        kf.check_within_bound(name, output.transpose(3, 0, 1, 2).reshape(oc, -1, 1), lhs, rhs, None, dtype)

    bias_shape = (0,) if bias is None else np.shape(bias)

    def probe(ops):
        bb = None if bias is None else np.asarray(ops[2], np.float16).reshape(bias_shape)
        return call_for(np.asarray(ops[0], np.float16).reshape(x.shape),
                        np.asarray(ops[1], np.float16).reshape(weights.shape), bb).output().reshape(output_shape)

    return output, probe


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
        params = {**window, "input_offset": 0, "output_offset": 0, "activation_min": lo, "activation_max": hi}
        result = {"input_shape": input_shape, "output_shape": output_shape, "depth_multiplier": mult,
                  "input": x, "weights": weights, "biases": bias if use_bias else None,
                  "params": params, "quant_params": None, "filter_operand": weights, "bias_operand": bias}
        route = (depthwise_f16_route(op, kernel_info, input_shape, output_shape, kh, kw, mult, window)
                 if float_dtype == np.float16 else None)
        if route is not None:
            from helia_core_tester.generation.ops._shared import kernel_faithful as kf

            # The kernel's own float16 arithmetic (arm_depthwise_conv_f16's route on Helium), bit for bit.
            output, probe = depthwise_faithful_case(op, x, weights, bias if use_bias else None, kh, kw, mult, window,
                                                    output_shape, route, (lo, hi), float_dtype)
            return {**result, "output": output, "probe": probe,
                    "comparison": kf.comparison_context(op, float_dtype)}
        entry = "depthwise_conv_f16" if float_dtype == np.float16 else "depthwise_conv_f32"
        call = ReferenceCall(entry, {**window, "activation_min": lo, "activation_max": hi},
                             {"input": np.ascontiguousarray(x), "filter": weights, "bias": bias},
                             {"output": output_shape})
        output = op.reference_golden(call)
        # Channels stay independent: each output sums its in-range taps, plus the bias.
        k = window_taps(h, kh, sh, ph, dh, oh) * window_taps(w, kw, sw, pw, dw, ow) + int(use_bias)
        return {**result, "output": output, "comparison": exact_comparison(op, call, k, float_dtype)}

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
        params = {**geometry, "input_offset": 0, "output_offset": 0, "activation_min": lo, "activation_max": hi}
        result = {"input_shape": input_shape, "output_shape": output_shape, "input": x, "weights": weights,
                  "biases": bias if use_bias else None, "params": params, "quant_params": None,
                  "filter_operand": weights, "bias_operand": bias}
        from helia_core_tester.generation.ops._shared import kernel_faithful as kf

        if (float_dtype == np.float16 and kf.mve_float16_route(op)
                and kf.transpose_conv_f16_mve_route(kh, kw, in_c)):
            # The kernel's own float16 arithmetic (arm_transpose_conv_f16's gathered vfmaq_m), bit for bit.
            call = kf.transpose_conv_f16_mve_call(x, weights, bias if use_bias else None, window, output_shape,
                                                  (lo, hi))
            output = op.reference_golden(call)
            patches, rows = kf.transpose_conv_patches(x, weights, window, output_shape)
            kf.check_within_bound(name, kf.unclamped_output(call).reshape(1, -1, out_c), patches, rows,
                                  bias if use_bias else None, float_dtype)
            return {**result, "output": output, "comparison": kf.comparison_context(op, float_dtype)}
        entry = "transpose_conv_f16" if float_dtype == np.float16 else "transpose_conv_f32"
        call = ReferenceCall(entry, {**window, "activation_min": lo, "activation_max": hi},
                             {"input": np.ascontiguousarray(x), "filter": weights, "bias": bias},
                             {"output": output_shape})
        output = op.reference_golden(call)
        # Each output sums the (input, tap) pairs scattered onto it over in_c channels, plus the bias.
        k = (transpose_window_taps(h, kh, sh, window["pad_h"], oh)
             * transpose_window_taps(w, kw, sw, window["pad_w"], ow) * in_c + int(use_bias))
        return {**result, "output": output, "comparison": exact_comparison(op, call, k, float_dtype)}

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
