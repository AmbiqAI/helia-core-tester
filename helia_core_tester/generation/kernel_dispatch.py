"""
CPU-aware kernel dispatch for generated CMSIS-NN calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from helia_core_tester.core.cpu_targets import get_cpu_profile


def _cpu_buffer_api(base: str, cpu: str) -> str:
    profile = get_cpu_profile(cpu)
    if profile.has_mve:
        return f"{base}_mve"
    if profile.has_dsp:
        return f"{base}_dsp"
    return base


@dataclass(frozen=True)
class DirectEntry:
    """A public ns-cmsis-nn entry that a descriptor can call through `entry:`."""

    operator: str
    activation_dtype: str
    weight_dtype: str
    # The prototype the operator's template renders the call for. An entry of an existing
    # family takes one row below plus its cases; a new family also needs a template branch.
    family: str
    # Scratch query, called with (input_dims, filter_dims).
    buffer_size_fn: str


# Public entries that no wrapper routes to. The depthwise_s8 family takes
# arm_depthwise_conv_wrapper_s8's arguments; convolve_s8 takes arm_convolve_s8's
# (upscale_dims passed as NULL, as arm_convolve_wrapper_s8 does). Both take weight sums.
DIRECT_ENTRIES: Dict[str, DirectEntry] = {
    "arm_depthwise_conv_s8_opt_3x3": DirectEntry(
        "DepthwiseConv", "S8", "S8", "depthwise_s8", "arm_depthwise_conv_s8_opt_get_buffer_size"
    ),
    "arm_depthwise_conv_s8_opt_3x3_c64_s1": DirectEntry(
        "DepthwiseConv", "S8", "S8", "depthwise_s8", "arm_depthwise_conv_s8_opt_get_buffer_size"
    ),
    "arm_depthwise_conv_s8_opt_planar": DirectEntry(
        "DepthwiseConv", "S8", "S8", "depthwise_s8", "arm_depthwise_conv_s8_opt_get_buffer_size"
    ),
    "arm_depthwise_conv_s8_opt_channelwise": DirectEntry(
        "DepthwiseConv", "S8", "S8", "depthwise_s8", "arm_depthwise_conv_s8_opt_get_buffer_size"
    ),
    "arm_convolve_s8_small_cin": DirectEntry(
        "Convolve", "S8", "S8", "convolve_s8", "arm_convolve_s8_get_buffer_size"
    ),
    "arm_convolve_s8_3x3_c16_s1": DirectEntry(
        "Convolve", "S8", "S8", "convolve_s8", "arm_convolve_s8_get_buffer_size"
    ),
}
DEPTHWISE_CONV_S8_DIRECT_ENTRIES = tuple(
    name for name, spec in DIRECT_ENTRIES.items() if spec.operator == "DepthwiseConv"
)
DEPTHWISE_CONV_S8_PLANAR_RULE = "arm_depthwise_conv_s8_opt_planar_supported"

_OPERATOR_LABELS = {"DepthwiseConv": "depthwise", "Convolve": "convolve"}


def resolve_direct_entry(operator: str, entry: str, activation_dtype: str, weight_dtype: str) -> Dict[str, str]:
    """Kernel-info overrides for an `operator` descriptor that calls the named entry."""
    spec = DIRECT_ENTRIES.get(entry)
    if spec is None or spec.operator != operator:
        known = [name for name, candidate in DIRECT_ENTRIES.items() if candidate.operator == operator]
        raise ValueError(f"Unknown {operator} entry {entry!r}; known entries are {known}")
    if (str(activation_dtype).upper(), str(weight_dtype).upper()) != (spec.activation_dtype, spec.weight_dtype):
        raise ValueError(
            f"entry {entry!r} is an {spec.activation_dtype.lower()} {_OPERATOR_LABELS.get(operator, operator)} "
            f"entry; the descriptor is {activation_dtype} x {weight_dtype}"
        )
    return {
        "kernel_fn": entry,
        "kernel_get_buffer_size_fn": spec.buffer_size_fn,
        "entry_family": spec.family,
    }


def resolve_depthwise_conv_entry(entry: str, activation_dtype: str, weight_dtype: str) -> Dict[str, str]:
    """Kernel-info overrides for a descriptor that calls a named depthwise entry."""
    resolved = resolve_direct_entry("DepthwiseConv", entry, activation_dtype, weight_dtype)
    return {key: resolved[key] for key in ("kernel_fn", "kernel_get_buffer_size_fn")}


def resolve_convolve_kernel(activation_dtype: str, weight_dtype: str, cpu: str) -> Dict[str, str]:
    act = str(activation_dtype).upper()
    w = str(weight_dtype).upper()

    if act == "FP32" and w == "FP32":
        return {
            "kernel_fn": "arm_convolve_f32",
            "kernel_get_buffer_size_fn": "arm_convolve_f32_get_buffer_size",
            "input_c_type": "float",
            "output_c_type": "float",
            "weight_c_type": "float",
            "bias_c_type": "float",
            "call_style": "baseline",
            "layout": "ARM_NN_LAYOUT_NHWC",
        }

    if act == "FP16" and w == "FP16":
        return {
            "kernel_fn": "arm_convolve_f16",
            "kernel_get_buffer_size_fn": "arm_convolve_f16_get_buffer_size",
            "input_c_type": "float16_t",
            "output_c_type": "float16_t",
            "weight_c_type": "float16_t",
            "bias_c_type": "float16_t",
            "call_style": "baseline",
            "layout": "ARM_NN_LAYOUT_NHWC",
        }

    if w == "S4":
        if act != "S8":
            raise NotImplementedError(f"Unsupported Convolve dtype combo: {act} x {w}")
        return {
            "kernel_fn": "arm_convolve_wrapper_s4",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_convolve_wrapper_s4_get_buffer_size", cpu),
            "input_c_type": "int8_t",
            "output_c_type": "int8_t",
            "weight_c_type": "int8_t",
            "bias_c_type": "int32_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    if act == "S8" and w == "S8":
        return {
            "kernel_fn": "arm_convolve_wrapper_s8",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_convolve_wrapper_s8_get_buffer_size", cpu),
            "input_c_type": "int8_t",
            "output_c_type": "int8_t",
            "weight_c_type": "int8_t",
            "bias_c_type": "int32_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    if act == "S16" and w == "S8":
        return {
            "kernel_fn": "arm_convolve_wrapper_s16",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_convolve_wrapper_s16_get_buffer_size", cpu),
            "input_c_type": "int16_t",
            "output_c_type": "int16_t",
            "weight_c_type": "int8_t",
            "bias_c_type": "int64_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    raise NotImplementedError(f"Unsupported Convolve dtype combo: {act} x {w}")


def resolve_depthwise_conv_kernel(activation_dtype: str, weight_dtype: str, cpu: str) -> Dict[str, str]:
    act = str(activation_dtype).upper()
    w = str(weight_dtype).upper()

    if act == "FP32" and w == "FP32":
        return {
            "kernel_fn": "arm_depthwise_conv_f32",
            "kernel_get_buffer_size_fn": "arm_depthwise_conv_f32_get_buffer_size",
            "input_c_type": "float",
            "output_c_type": "float",
            "weight_c_type": "float",
            "bias_c_type": "float",
            "call_style": "baseline",
            "layout": "ARM_NN_LAYOUT_NHWC",
        }

    if act == "FP16" and w == "FP16":
        return {
            "kernel_fn": "arm_depthwise_conv_f16",
            "kernel_get_buffer_size_fn": "arm_depthwise_conv_f16_get_buffer_size",
            "input_c_type": "float16_t",
            "output_c_type": "float16_t",
            "weight_c_type": "float16_t",
            "bias_c_type": "float16_t",
            "call_style": "baseline",
            "layout": "ARM_NN_LAYOUT_NHWC",
        }

    if act == "S8" and w == "S4":
        return {
            "kernel_fn": "arm_depthwise_conv_wrapper_s4",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_depthwise_conv_wrapper_s4_get_buffer_size", cpu),
            "input_c_type": "int8_t",
            "output_c_type": "int8_t",
            "weight_c_type": "int8_t",
            "bias_c_type": "int32_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    if act == "S8" and w == "S8":
        return {
            "kernel_fn": "arm_depthwise_conv_wrapper_s8",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_depthwise_conv_wrapper_s8_get_buffer_size", cpu),
            "input_c_type": "int8_t",
            "output_c_type": "int8_t",
            "weight_c_type": "int8_t",
            "bias_c_type": "int32_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    if act == "S16" and w == "S8":
        return {
            "kernel_fn": "arm_depthwise_conv_wrapper_s16",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_depthwise_conv_wrapper_s16_get_buffer_size", cpu),
            "input_c_type": "int16_t",
            "output_c_type": "int16_t",
            "weight_c_type": "int8_t",
            "bias_c_type": "int64_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    raise NotImplementedError(f"Unsupported DepthwiseConv dtype combo: {act} x {w}")


def resolve_fully_connected_kernel(activation_dtype: str, weight_dtype: str, cpu: str) -> Dict[str, str]:
    act = str(activation_dtype).upper()
    w = str(weight_dtype).upper()

    if act == "FP32" and w == "FP32":
        return {
            "kernel_fn": "arm_fully_connected_f32",
            "kernel_get_buffer_size_fn": "arm_fully_connected_f32_get_buffer_size",
            "input_c_type": "float",
            "output_c_type": "float",
            "weight_c_type": "float",
            "bias_c_type": "float",
            "call_style": "baseline",
            "fc_params_type": "cmsis_nn_fc_params_f32",
            "layout": "ARM_NN_LAYOUT_NHWC",
        }

    if act == "FP16" and w == "FP16":
        return {
            "kernel_fn": "arm_fully_connected_f16",
            "kernel_get_buffer_size_fn": "arm_fully_connected_f16_get_buffer_size",
            "input_c_type": "float16_t",
            "output_c_type": "float16_t",
            "weight_c_type": "float16_t",
            "bias_c_type": "float16_t",
            "call_style": "baseline",
            "fc_params_type": "cmsis_nn_fc_params_f16",
            "layout": "ARM_NN_LAYOUT_NHWC",
        }

    if act == "S8" and w == "S4":
        return {
            "kernel_fn": "arm_fully_connected_s4",
            "kernel_get_buffer_size_fn": None,
            "input_c_type": "int8_t",
            "output_c_type": "int8_t",
            "weight_c_type": "int8_t",
            "bias_c_type": "int32_t",
            "call_style": "baseline",
        }

    if act == "S8" and w == "S8":
        return {
            "kernel_fn": "arm_fully_connected_wrapper_s8",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_fully_connected_s8_get_buffer_size", cpu),
            "input_c_type": "int8_t",
            "output_c_type": "int8_t",
            "weight_c_type": "int8_t",
            "bias_c_type": "int32_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    if act == "S16" and w == "S8":
        return {
            "kernel_fn": "arm_fully_connected_wrapper_s16",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_fully_connected_s16_get_buffer_size", cpu),
            "input_c_type": "int16_t",
            "output_c_type": "int16_t",
            "weight_c_type": "int8_t",
            "bias_c_type": "int64_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    raise NotImplementedError(f"Unsupported FullyConnected dtype combo: {act} x {w}")
