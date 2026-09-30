"""Quantize and Dequantize pools: one kernel call over the input, with TFLite's fused activation
applied as a pre-pass in float (Quantize) or a post-pass over the float output (Dequantize)."""

from __future__ import annotations

from typing import Any, Mapping

from helia_core_tester.generation.harness import ArgumentPool, GuardedBuffer
from helia_core_tester.generation.harness.simple import tensor_case_pool

ACTIVATIONS = ("RELU", "RELU6")


def _activation(context: Mapping[str, Any]) -> str | None:
    if not context.get("has_activation"):
        return None
    kind = str(context.get("activation_type", "NONE")).upper()
    if kind not in ACTIVATIONS:
        raise ValueError(f"{context['name']}: fused activation {kind!r} is not one of {ACTIVATIONS}")
    return kind


def _scale_values(context: Mapping[str, Any]) -> dict[str, str]:
    return {"size": str(context["input_size"]), "zero_point": str(context["zero_point"]),
            "scale": f"{context['scale']}f"}


def quantize_argument_pool(context: Mapping[str, Any]) -> ArgumentPool:
    n, size = context["name"], int(context["input_size"])
    if size < 1:
        raise ValueError(f"{n}: Quantize needs at least one element")
    kind = _activation(context)
    if kind is None:
        return tensor_case_pool(context, _scale_values(context), dims=(), output_count=str(size))
    label = '{{ validation_label | default("Quantize") }}'
    if kind == "RELU":
        loop = f"        {n}_activated_input[i] = (input[i] < 0.0f) ? 0.0f : input[i];"
    else:
        loop = ("        float val = input[i];\n        if (val < 0.0f) val = 0.0f;\n"
                f"        if (val > 6.0f) val = 6.0f;\n        {n}_activated_input[i] = val;")
    pre = (f"    HELIA_GUARD_ARM({n}_activated_input, false /* fully written below: don't poison */);\n"
           f"    for (int i = 0; i < {size}; i++) {{\n{loop}\n    }}")
    guarded = [GuardedBuffer(f"{n}_activated_input", "float", str(size), label="activated_input")]
    checks = [f'    HELIA_GUARD_CHECK({n}_activated_input, "{label} activated_input", failures);']
    post = ""
    probe = context.get("activation_kernel_fn")
    if probe:
        guarded.append(GuardedBuffer(f"{n}_activation_probe", context["output_dtype"], str(size),
                                     label="activation_probe"))
        checks.append(f'    HELIA_GUARD_CHECK({n}_activation_probe, "{label} activation_probe", failures);')
        post = (f"    HELIA_GUARD_ARM({n}_activation_probe, false /* fully written below: don't poison */);\n"
                f"    for (int i = 0; i < {size}; i++) {{\n        {n}_activation_probe[i] = output[i];\n    }}\n"
                f"    {probe}({n}_activation_probe, (uint16_t){size});")
    return tensor_case_pool(context, _scale_values(context), dims=(), output_count=str(size),
                            calls=({"input": f"{n}_activated_input"},), guarded=tuple(guarded), pre_call=pre,
                            post_call=post, extra_checks="\n".join(checks))


def dequantize_argument_pool(context: Mapping[str, Any]) -> ArgumentPool:
    n, size = context["name"], int(context["input_size"])
    if size < 1:
        raise ValueError(f"{n}: Dequantize needs at least one element")
    if context.get("kernel_style") == "widen":
        values = {"block_size": str(size)}
    else:
        values = _scale_values(context)
    kind = _activation(context)
    post = ""
    if kind == "RELU":
        post = f"    for (int i = 0; i < {size}; i++) {{\n        if (output[i] < 0.0f) output[i] = 0.0f;\n    }}"
    elif kind == "RELU6":
        post = (f"    for (int i = 0; i < {size}; i++) {{\n        if (output[i] < 0.0f) output[i] = 0.0f;\n"
                "        if (output[i] > 6.0f) output[i] = 6.0f;\n    }")
    return tensor_case_pool(context, values, dims=(), output_count=str(size), post_call=post)
