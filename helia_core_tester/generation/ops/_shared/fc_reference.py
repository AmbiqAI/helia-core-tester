"""Reference case construction for FullyConnected: the draw, the quantization the converter
used to produce, and the golden from the C reference (TFLite's FullyConnected)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from helia_core_tester.generation.ops._shared.conv_reference import (
    _extras,
    _uniform_input,
    float_bounds,
    per_channel_multipliers,
    quantized_bounds,
    weight_rng,
)


def fc_shapes(desc) -> Dict[str, Any]:
    input_shape = tuple(int(d) for d in desc["input_shape"])
    fs = tuple(int(d) for d in desc["filter_shape"])
    if len(input_shape) not in (2, 4) or len(fs) != 2:
        raise ValueError(f"{desc['name']}: FullyConnected needs a 2-D or 4-D input_shape and an [out, in] filter_shape")
    batch = input_shape[0]
    features = int(np.prod(input_shape[1:]))
    # Keras read only the unit count; the feature count comes from the input.
    out_units = fs[0]
    return {"input_shape": input_shape, "batch": batch, "features": features, "out_units": out_units,
            "output_shape": (batch, out_units)}


def fc_reference_case(op, kernel_info: Dict[str, Any], float_kernel: bool, float_dtype) -> Dict[str, Any]:
    from helia_core_tester.generation.ops._shared.quant_knobs import value_range
    from helia_core_tester.generation.reference import policy, weighted
    from helia_core_tester.generation.reference.bindings import get_bindings
    from helia_core_tester.generation.reference.call import ReferenceCall

    desc = op.desc
    name = desc["name"]
    sh = fc_shapes(desc)
    out_units, features, output_shape = sh["out_units"], sh["features"], sh["output_shape"]
    use_bias = bool(desc.get("use_bias", True))
    wrng = weight_rng(op)
    filter_shape = (out_units, features)
    draw = lambda: weighted.glorot_uniform(wrng, filter_shape, features, out_units, desc.get("weight_gain"))  # noqa: E731
    hint = desc.get("hint") or {}

    if float_kernel:
        x = np.asarray(op._sample_uniform(sh["input_shape"]), dtype=float_dtype)
        weights = draw().astype(float_dtype)
        bias = (wrng.uniform(-0.25, 0.25, out_units) if use_bias else np.zeros(0)).astype(float_dtype)
        lo, hi = float_bounds(op)
        entry = "fully_connected_f16" if float_dtype == np.float16 else "fully_connected_f32"
        output = op.reference_golden(ReferenceCall(
            entry, {"activation_min": lo, "activation_max": hi},
            {"input": np.ascontiguousarray(x), "filter": weights, "bias": bias}, {"output": output_shape}))
        params = {"input_offset": 0, "filter_offset": 0, "output_offset": 0, "activation_min": lo, "activation_max": hi}
        return {**sh, "input": x, "weights": weights, "biases": bias if use_bias else None, "output": output,
                "params": params, "multiplier": None, "shift": None, "per_channel": False,
                "filter_operand": weights, "bias_operand": bias}

    kind = "s16" if kernel_info["input_c_type"] == "int16_t" else "s8"
    weight_dtype = str(desc.get("weight_dtype", "S8")).upper()
    extras = _extras(desc)
    filter_offset = int(extras.get("force_filter_offset", 0)) if kind == "s8" else 0
    force_per_tensor = bool(hint.get("force_per_tensor", False))
    in_range = (value_range(desc, "calibration_range", ()) if desc.get("calibration_range")
                else value_range(desc, "input_range", ()) if desc.get("input_range") else (-1.0, 1.0))
    if weight_dtype == "S4":
        # A hand-built int4 model: pinned input and filter scales, output calibrated.
        if "output_scale" in extras or "output_zero_point" in extras:
            raise ValueError(f"{name}: an int4 case's output quantization is calibrated, not pinned in extras")
        in_quant = policy.TensorQuant(float(np.float32(extras.get("input_scale", 4.0))),
                                      int(extras.get("input_zero_point", 3)), kind)
        qfilter = weighted.QuantizedFilter(wrng.integers(-8, 8, size=filter_shape).astype(np.int8),
                                           np.full(out_units, np.float32(extras.get("weight_scale", 1.0))), "S4")
        bias_q = wrng.integers(-128, 128, size=out_units).astype(np.int32) if use_bias else None
        x_float = op.generate_input_data()
        per_channel = False
    else:
        in_quant = op.activation_quant("input", in_range, kind)
        w = draw()
        scales = [float(np.abs(w).max()) / 127.0] if force_per_tensor else None
        qfilter = weighted.quantize_filter(w, "S8", channel_axis=0, scales=scales)
        bias_q = (weighted.quantize_bias(weighted.signed_magnitude(wrng, out_units, 0.125, 0.25), in_quant.scale,
                                         qfilter.scales, wide=kind == "s16") if use_bias else None)
        x_float = _uniform_input(op, in_range)
        per_channel = not force_per_tensor
    if bias_q is None and kind == "s16":
        bias_q = np.zeros(out_units, np.int64)
    x_q = policy.quantize(x_float, in_quant)

    f_lo, f_hi = float_bounds(op)
    real_bias = (bias_q.astype(np.float64) * in_quant.scale * qfilter.scales.astype(np.float64)
                 if bias_q is not None else np.zeros(0))
    real_filter = (qfilter.values.astype(np.float64) + filter_offset) * qfilter.scales.astype(np.float64)[:, None]
    lib = get_bindings()
    real = lib.run("fully_connected_f32", {"activation_min": f_lo, "activation_max": f_hi},
                   {"input": policy.dequantize(x_q, in_quant).astype(np.float32),
                    "filter": real_filter.astype(np.float32), "bias": real_bias.astype(np.float32)},
                   {"output": output_shape})["output"]
    out_quant = op.activation_quant("output", policy.data_range(real), kind)
    multiplier, shift = per_channel_multipliers(in_quant.scale, out_quant.scale, qfilter.scales)
    act_min, act_max = quantized_bounds(op, kind, out_quant.scale, out_quant.zero_point)
    params = {"input_offset": -in_quant.zero_point, "filter_offset": filter_offset,
              "output_offset": out_quant.zero_point, "activation_min": act_min, "activation_max": act_max}
    bias_operand = bias_q if bias_q is not None else np.zeros(0, np.int32)
    output = op.reference_golden(ReferenceCall(
        f"fully_connected_{kind}", params,
        {"input": np.ascontiguousarray(x_q), "filter": np.ascontiguousarray(qfilter.values), "bias": bias_operand,
         "multiplier": multiplier, "shift": shift},
        {"output": output_shape},
        quant={"input": in_quant.to_json(), "output": out_quant.to_json(),
               "filter_scales": [float(v) for v in qfilter.scales]}))
    weights_out = weighted.pack_int4(qfilter.values) if weight_dtype == "S4" else qfilter.values
    return {**sh, "input": x_q, "input_zero_point": in_quant.zero_point, "weights": weights_out,
            "weights_unpacked": qfilter.values, "biases": bias_q, "output": output, "params": params,
            "multiplier": multiplier, "shift": shift, "per_channel": per_channel,
            "filter_operand": qfilter.values, "bias_operand": bias_operand}
