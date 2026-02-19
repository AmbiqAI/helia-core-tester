"""
CPU-aware kernel dispatch for generated CMSIS-NN calls.
"""

from __future__ import annotations

from typing import Dict

from helia_core_tester.core.cpu_targets import get_cpu_profile


def _cpu_buffer_api(base: str, cpu: str) -> str:
    profile = get_cpu_profile(cpu)
    if profile.has_mve:
        return f"{base}_mve"
    if profile.has_dsp:
        return f"{base}_dsp"
    return base


def resolve_conv2d_kernel(activation_dtype: str, weight_dtype: str, cpu: str) -> Dict[str, str]:
    act = str(activation_dtype).upper()
    w = str(weight_dtype).upper()

    if w == "S4":
        raise NotImplementedError("Conv2D with S4 weights is not supported by the current generator")

    if act == "S8" and w == "S8":
        return {
            "kernel_fn": "arm_convolve_wrapper_s8",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_convolve_wrapper_s8_get_buffer_size", cpu),
            "input_c_type": "int8_t",
            "output_c_type": "int8_t",
            "bias_c_type": "int32_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    if act == "S16" and w == "S8":
        return {
            "kernel_fn": "arm_convolve_wrapper_s16",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_convolve_wrapper_s16_get_buffer_size", cpu),
            "input_c_type": "int16_t",
            "output_c_type": "int16_t",
            "bias_c_type": "int64_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    raise NotImplementedError(f"Unsupported Conv2D dtype combo: {act} x {w}")


def resolve_depthwise_conv2d_kernel(activation_dtype: str, weight_dtype: str, cpu: str) -> Dict[str, str]:
    act = str(activation_dtype).upper()
    w = str(weight_dtype).upper()

    if act == "S8" and w == "S8":
        return {
            "kernel_fn": "arm_depthwise_conv_wrapper_s8",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_depthwise_conv_wrapper_s8_get_buffer_size", cpu),
            "input_c_type": "int8_t",
            "output_c_type": "int8_t",
            "bias_c_type": "int32_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    if act == "S16" and w == "S8":
        return {
            "kernel_fn": "arm_depthwise_conv_wrapper_s16",
            "kernel_get_buffer_size_fn": _cpu_buffer_api("arm_depthwise_conv_wrapper_s16_get_buffer_size", cpu),
            "input_c_type": "int16_t",
            "output_c_type": "int16_t",
            "bias_c_type": "int64_t",
            "call_style": "m55" if get_cpu_profile(cpu).has_mve else "baseline",
        }

    raise NotImplementedError(f"Unsupported DepthwiseConv2D dtype combo: {act} x {w}")


def resolve_fully_connected_kernel(activation_dtype: str, weight_dtype: str, cpu: str) -> Dict[str, str]:
    act = str(activation_dtype).upper()
    w = str(weight_dtype).upper()

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
