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


DEPTHWISE_CONV_S8_PLANAR_RULE = "arm_depthwise_conv_s8_opt_planar_supported"
DEPTHWISE_CONV_S8_3X3_SIZER = "arm_depthwise_conv_s8_opt_3x3_get_buffer_size"


def depthwise_3x3_scratch_bytes(sizers: object, input_dims: Dict[str, int]) -> int:
    """Mirror the 3x3 depthwise entries' own scratch query when a descriptor lists it beside the
    family's (`entry_sizer: [<family query>, <entry query>]`); zero for every other entry."""
    listed = [sizers] if isinstance(sizers, str) else list(sizers or ())
    if DEPTHWISE_CONV_S8_3X3_SIZER not in listed:
        return 0
    # 16 groups x (3 x 52 + 32), pad row, align.
    return 16 * (3 * 52 + 32) + int(input_dims["w"]) * int(input_dims["c"]) + 16


def autovectorize_declines_if(input_c_type: str) -> str:
    """The preprocessor condition under which an `autovectorize_declines` entry declines.

    Integer entries live on ns-cmsis-nn's MVE integer paths, which an integer coverage build compiles
    out. Float entries live on its MVE float paths, absent without MVE float and compiled out by a
    coverage build that gives the float sources ARM_MATH_AUTOVECTORIZE. CMakeLists.txt sets the
    HELIA_CMSIS_NN_*_AUTOVECTORIZE flags.
    """
    if input_c_type == "float16_t":
        return "!defined(ARM_MATH_MVE_FLOAT16) || defined(HELIA_CMSIS_NN_FLOAT_AUTOVECTORIZE)"
    if input_c_type == "float":
        return "!defined(ARM_MATH_MVEF) || defined(HELIA_CMSIS_NN_FLOAT_AUTOVECTORIZE)"
    return "defined(HELIA_CMSIS_NN_INT_AUTOVECTORIZE)"


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
