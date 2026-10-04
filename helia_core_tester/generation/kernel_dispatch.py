"""
CPU-aware kernel dispatch for generated CMSIS-NN calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
    # Scratch query. depthwise_s8 and convolve_s8 call it with (input_dims, filter_dims),
    # convolve_1x1_s8 with (input_dims); float entries call it like their default entry,
    # with (params, input_dims, filter_dims, output_dims[, layout]). fully_connected_packed_s8
    # takes no scratch: this sizes the weight stream that <entry>_pack builds, from (filter_dims).
    buffer_size_fn: str
    # Float entries: whether the call and the scratch query take a trailing layout argument.
    kernel_needs_layout: bool = False
    buffer_size_needs_layout: bool = False


# Public entries that no wrapper routes to. The depthwise_s8 family takes
# arm_depthwise_conv_wrapper_s8's arguments; convolve_s8 takes arm_convolve_s8's
# (upscale_dims passed as NULL, as arm_convolve_wrapper_s8 does); convolve_1x1_s8 takes
# arm_convolve_1x1_s8_fast's and sizes scratch from the input dims alone. All take weight sums.
# fully_connected_packed_s8 takes one stream in place of the weights, kernel sums and
# quantization; the case builds it once with <entry>_pack, as an ahead-of-time caller does.
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
    # Declines with ARM_CMSIS_NN_NO_IMPL_ERROR outside its shape; callers then run
    # arm_convolve_1x1_s8_fast with the same arguments, so scratch is sized for that.
    "arm_convolve_1x1_s8_short_k": DirectEntry(
        "Convolve", "S8", "S8", "convolve_1x1_s8", "arm_convolve_1x1_s8_fast_get_buffer_size"
    ),
    # Reads a weight stream packed ahead of the call; per-channel quantization only.
    "arm_fully_connected_per_channel_packed_s8": DirectEntry(
        "FullyConnected",
        "S8",
        "S8",
        "fully_connected_packed_s8",
        "arm_fully_connected_per_channel_packed_s8_get_packed_size",
    ),
    # Row broadcast only ([N,H,W,C] x [N|1,H|1,1,C]); arm_add_s8 and arm_mul_s8 never call these.
    # Each takes its router's arguments, needs no scratch, and declines any other shape with
    # ARM_CMSIS_NN_NO_IMPL_ERROR.
    "arm_add_row_broadcast_s8": DirectEntry("Add", "S8", "S8", "add_s8", ""),
    "arm_mul_row_broadcast_s8": DirectEntry("Mul", "S8", "S8", "mul_s8", ""),
}


def _fp16(operator: str, buffer_size_fn: str, kernel_needs_layout: bool, buffer_size_needs_layout: bool) -> DirectEntry:
    return DirectEntry(
        operator, "FP16", "FP16", "float", buffer_size_fn, kernel_needs_layout, buffer_size_needs_layout
    )


def _fp32(operator: str, buffer_size_fn: str, kernel_needs_layout: bool, buffer_size_needs_layout: bool) -> DirectEntry:
    return DirectEntry(
        operator, "FP32", "FP32", "float", buffer_size_fn, kernel_needs_layout, buffer_size_needs_layout
    )


# FP16 entries take their default entry's float call. The _nhwc_ entries have no layout
# argument and size scratch with the layout-taking query at ARM_NN_LAYOUT_NHWC; the wrappers
# keep their own query. The _acc16 entries keep float16 lane accumulation (heliaAOT fast mode).
DIRECT_ENTRIES.update(
    {
        "arm_convolve_f16_acc16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", True, True),
        "arm_convolve_nhwc_f16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_nhwc_f16_acc16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_wrapper_f16_acc16": _fp16("Convolve", "arm_convolve_wrapper_f16_get_buffer_size", False, False),
        "arm_convolve_1x1_f16_acc16": _fp16("Convolve", "arm_convolve_1x1_f16_get_buffer_size", True, True),
        "arm_convolve_1x1_nhwc_f16": _fp16("Convolve", "arm_convolve_1x1_f16_get_buffer_size", False, True),
        "arm_convolve_1x1_nhwc_f16_acc16": _fp16("Convolve", "arm_convolve_1x1_f16_get_buffer_size", False, True),
        "arm_convolve_1_x_n_f16_acc16": _fp16("Convolve", "arm_convolve_1_x_n_f16_get_buffer_size", True, True),
        "arm_convolve_1_x_n_nhwc_f16": _fp16("Convolve", "arm_convolve_1_x_n_f16_get_buffer_size", False, True),
        "arm_convolve_1_x_n_nhwc_f16_acc16": _fp16("Convolve", "arm_convolve_1_x_n_f16_get_buffer_size", False, True),
        "arm_fully_connected_f16_acc16": _fp16("FullyConnected", "arm_fully_connected_f16_get_buffer_size", True, True),
        "arm_fully_connected_nhwc_f16": _fp16("FullyConnected", "arm_fully_connected_f16_get_buffer_size", False, True),
        "arm_fully_connected_nhwc_f16_acc16": _fp16(
            "FullyConnected", "arm_fully_connected_f16_get_buffer_size", False, True
        ),
        "arm_depthwise_conv_f16_acc16": _fp16("DepthwiseConv", "arm_depthwise_conv_f16_get_buffer_size", True, True),
        "arm_depthwise_nhwc_conv_f16": _fp16("DepthwiseConv", "arm_depthwise_conv_f16_get_buffer_size", False, True),
        "arm_depthwise_nhwc_conv_f16_acc16": _fp16(
            "DepthwiseConv", "arm_depthwise_conv_f16_get_buffer_size", False, True
        ),
        "arm_depthwise_conv_wrapper_f16_acc16": _fp16(
            "DepthwiseConv", "arm_depthwise_conv_wrapper_f16_get_buffer_size", False, False
        ),
    }
)

# Per-route float entries (ns-cmsis-nn#674): the argument list of arm_convolve_nhwc_f16/f32 or
# arm_depthwise_nhwc_conv_f16, with no layout argument. _ohwi_ entries take STANDARD filters and
# _packed_ NT_N_PACKED; each declines the other format with ARM_CMSIS_NN_NO_IMPL_ERROR. 1x1, 1xN
# and patch-GEMM size scratch with their own query; the rest need none and get the router's.
DIRECT_ENTRIES.update(
    {
        "arm_convolve_1x1_nhwc_ohwi_f32": _fp32("Convolve", "arm_convolve_1x1_f32_get_buffer_size", False, True),
        "arm_convolve_1_x_n_nhwc_ohwi_f32": _fp32("Convolve", "arm_convolve_1_x_n_f32_get_buffer_size", False, True),
        "arm_convolve_1d_k5_nhwc_ohwi_f32": _fp32("Convolve", "arm_convolve_f32_get_buffer_size", False, True),
        "arm_convolve_1d_k3_nhwc_ohwi_f32": _fp32("Convolve", "arm_convolve_f32_get_buffer_size", False, True),
        "arm_convolve_patch_gemm_nhwc_ohwi_f32": _fp32("Convolve", "arm_convolve_patch_gemm_f32_get_buffer_size", False, False),
        "arm_convolve_direct_nhwc_ohwi_f32": _fp32("Convolve", "arm_convolve_f32_get_buffer_size", False, True),
        "arm_convolve_1x1_nhwc_packed_f32": _fp32("Convolve", "arm_convolve_1x1_f32_get_buffer_size", False, True),
        "arm_convolve_1_x_n_nhwc_packed_f32": _fp32("Convolve", "arm_convolve_1_x_n_f32_get_buffer_size", False, True),
        "arm_convolve_1d_k5_nhwc_packed_f32": _fp32("Convolve", "arm_convolve_f32_get_buffer_size", False, True),
        "arm_convolve_1d_k3_nhwc_packed_f32": _fp32("Convolve", "arm_convolve_f32_get_buffer_size", False, True),
        "arm_convolve_patch_gemm_nhwc_packed_f32": _fp32("Convolve", "arm_convolve_patch_gemm_f32_get_buffer_size", False, False),
        "arm_convolve_direct_nhwc_packed_f32": _fp32("Convolve", "arm_convolve_f32_get_buffer_size", False, True),
        "arm_convolve_small_c_nhwc_f32": _fp32("Convolve", "arm_convolve_f32_get_buffer_size", False, True),
        "arm_convolve_1x1_nhwc_ohwi_f16": _fp16("Convolve", "arm_convolve_1x1_f16_get_buffer_size", False, True),
        "arm_convolve_1_x_n_nhwc_ohwi_f16": _fp16("Convolve", "arm_convolve_1_x_n_f16_get_buffer_size", False, True),
        "arm_convolve_1d_k5_nhwc_ohwi_f16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_1d_k3_nhwc_ohwi_f16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_patch_gemm_nhwc_ohwi_f16": _fp16("Convolve", "arm_convolve_patch_gemm_f16_get_buffer_size", False, False),
        "arm_convolve_direct_nhwc_ohwi_f16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_1x1_nhwc_ohwi_f16_acc16": _fp16("Convolve", "arm_convolve_1x1_f16_get_buffer_size", False, True),
        "arm_convolve_1_x_n_nhwc_ohwi_f16_acc16": _fp16("Convolve", "arm_convolve_1_x_n_f16_get_buffer_size", False, True),
        "arm_convolve_1d_k5_nhwc_ohwi_f16_acc16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_1d_k3_nhwc_ohwi_f16_acc16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_patch_gemm_nhwc_ohwi_f16_acc16": _fp16("Convolve", "arm_convolve_patch_gemm_f16_get_buffer_size", False, False),
        "arm_convolve_direct_nhwc_ohwi_f16_acc16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_1x1_nhwc_packed_f16": _fp16("Convolve", "arm_convolve_1x1_f16_get_buffer_size", False, True),
        "arm_convolve_1_x_n_nhwc_packed_f16": _fp16("Convolve", "arm_convolve_1_x_n_f16_get_buffer_size", False, True),
        "arm_convolve_1d_k5_nhwc_packed_f16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_1d_k3_nhwc_packed_f16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_patch_gemm_nhwc_packed_f16": _fp16("Convolve", "arm_convolve_patch_gemm_f16_get_buffer_size", False, False),
        "arm_convolve_direct_nhwc_packed_f16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_1x1_nhwc_packed_f16_acc16": _fp16("Convolve", "arm_convolve_1x1_f16_get_buffer_size", False, True),
        "arm_convolve_1_x_n_nhwc_packed_f16_acc16": _fp16("Convolve", "arm_convolve_1_x_n_f16_get_buffer_size", False, True),
        "arm_convolve_1d_k5_nhwc_packed_f16_acc16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_1d_k3_nhwc_packed_f16_acc16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_patch_gemm_nhwc_packed_f16_acc16": _fp16("Convolve", "arm_convolve_patch_gemm_f16_get_buffer_size", False, False),
        "arm_convolve_direct_nhwc_packed_f16_acc16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_convolve_small_c_nhwc_f16": _fp16("Convolve", "arm_convolve_f16_get_buffer_size", False, True),
        "arm_depthwise_conv_1d_k3_nhwc_f16": _fp16("DepthwiseConv", "arm_depthwise_conv_f16_get_buffer_size", False, True),
        "arm_depthwise_conv_2x5_nhwc_f16": _fp16("DepthwiseConv", "arm_depthwise_conv_f16_get_buffer_size", False, True),
        "arm_depthwise_conv_cin1_nhwc_f16": _fp16("DepthwiseConv", "arm_depthwise_conv_f16_get_buffer_size", False, True),
        "arm_depthwise_conv_cin1_nhwc_f16_acc16": _fp16("DepthwiseConv", "arm_depthwise_conv_f16_get_buffer_size", False, True),
        "arm_depthwise_conv_direct_nhwc_f16": _fp16("DepthwiseConv", "arm_depthwise_conv_f16_get_buffer_size", False, True),
        "arm_depthwise_conv_direct_nhwc_f16_acc16": _fp16("DepthwiseConv", "arm_depthwise_conv_f16_get_buffer_size", False, True),
        "arm_depthwise_conv_generic_nhwc_f16": _fp16("DepthwiseConv", "arm_depthwise_conv_f16_get_buffer_size", False, True),
        "arm_depthwise_conv_generic_nhwc_f16_acc16": _fp16("DepthwiseConv", "arm_depthwise_conv_f16_get_buffer_size", False, True),
    }
)

DEPTHWISE_CONV_S8_DIRECT_ENTRIES = tuple(
    name for name, spec in DIRECT_ENTRIES.items() if spec.operator == "DepthwiseConv" and spec.activation_dtype == "S8"
)
DEPTHWISE_CONV_S8_PLANAR_RULE = "arm_depthwise_conv_s8_opt_planar_supported"

_OPERATOR_LABELS = {"DepthwiseConv": "depthwise", "Convolve": "convolve", "FullyConnected": "fully connected"}


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
    resolved = {
        "kernel_fn": entry,
        "kernel_get_buffer_size_fn": spec.buffer_size_fn,
        "entry_family": spec.family,
    }
    if spec.family == "float":
        resolved["kernel_needs_layout"] = spec.kernel_needs_layout
        resolved["buffer_size_needs_layout"] = spec.buffer_size_needs_layout
    return resolved


def check_entry_fault(desc: Dict[str, Any], resolved: Dict[str, Any]) -> None:
    """Reject a fault on an entry case, except an invalid layout for a float entry that takes one."""
    fault = desc.get("fault")
    if not fault:
        return
    if fault == "invalid_layout" and resolved.get("entry_family") == "float" and resolved.get("kernel_needs_layout"):
        return
    raise ValueError(
        f"{desc.get('name')}: entry {resolved['kernel_fn']!r} supports only fault: invalid_layout, "
        f"and only as a float entry that takes a layout argument; got fault {fault!r}"
    )


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
