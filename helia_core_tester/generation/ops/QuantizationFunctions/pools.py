"""Quantize and Dequantize pools: one kernel call over the input, with TFLite's fused activation
applied as a pre-pass in float (Quantize) or a post-pass over the float output (Dequantize)."""

from __future__ import annotations

from typing import Any, Mapping

from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, GuardedBuffer, HarnessInput
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


def dequantize_f16_bits_pool(context: Mapping[str, Any]) -> ArgumentPool:
    """A bit-pattern case (arm_dequantize_f16_bits_f32): binary16 inputs and the float32 bits each
    NaN rule gives. The C file picks the rule its build compiled and compares the output bits."""
    n, size = context["name"], int(context["input_size"])
    upper, kernel_fn = n.upper(), context["kernel_fn"]
    if size < 1:
        raise ValueError(f"{n}: a bit-pattern case needs at least one element")
    header = [Declaration(f"{n}_input", "uint16_t", ArrayLiteral(context["input_data_array"]), array=True,
                          comment="Binary16 bit patterns")]
    header_text = "\n".join([
        "// Expected float32 bits: every NaN the default NaN (MVE vector conversion)",
        f"static const uint32_t {n}_expected_vector_bits[] = {{",
        context["expected_vector_bits_array"],
        "};",
        "",
        "// Expected float32 bits: every NaN quiet with its sign and payload (scalar conversion)",
        f"static const uint32_t {n}_expected_scalar_bits[] = {{",
        context["expected_scalar_bits_array"],
        "};",
    ])
    file_scope = "\n".join([
        f"/* The NaN rule this build compiled for {kernel_fn}: the MVE vector conversion where the library",
        " * builds with float16, MVE float16 and no ARM_MATH_AUTOVECTORIZE, unless its assembler verdict needs",
        " * the scalar form (Internal/arm_nn_vcvt_f16.h); the scalar conversion's rule everywhere else. */",
        "#if ARM_NN_ENABLE_F16 && defined(ARM_MATH_MVE_FLOAT16) && !defined(HELIA_CMSIS_NN_FLOAT_AUTOVECTORIZE) && \\",
        "    !defined(ARM_NN_VCVT_F16_SCALAR_FORM)",
        f"    #define {upper}_EXPECTED_BITS {n}_expected_vector_bits",
        f'    #define {upper}_NAN_RULE "vector: default NaN"',
        "#else",
        f"    #define {upper}_EXPECTED_BITS {n}_expected_scalar_bits",
        f'    #define {upper}_NAN_RULE "scalar: quiet, sign and payload kept"',
        "#endif",
    ])
    validation = "\n".join([
        f'    printf("HELIA_F16_NAN_RULE %s ({int(context["nan_count"])} NaN inputs)\\r\\n", {upper}_NAN_RULE);',
        f"    for (int32_t i = 0; i < {upper}_OUTPUT_SIZE; ++i) {{",
        "        uint32_t got;",
        f"        memcpy(&got, &{n}_output[i], sizeof(got));",
        f"        HELIA_VALIDATE_FLOAT_BITS(got, {upper}_EXPECTED_BITS[i], 0x7f800000u, 0, i, 20, failures);",
        "    }",
    ])
    return ArgumentPool(
        name=n, values={"block_size": str(size)}, header=header, header_text=header_text, file_scope=file_scope,
        validation=validation, output_count=str(size), benchmark=False, scratch_buffer=False,
        inputs=(HarnessInput("input", "input", f"{n}_input"),),
        includes=("<string.h>", '"Internal/arm_nn_vcvt_f16.h"'),
    )
