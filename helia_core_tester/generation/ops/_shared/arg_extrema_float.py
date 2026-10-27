"""Focused standalone float-input, exact-index fixtures."""

import numpy as np

from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


def float_arg_kernel(op, kind):
    dtype = op.tensor_dtype("input")
    if op.tensor_dtype("output") != "S32":
        raise ValueError("Float arg extrema requires explicit S32 output")
    if dtype not in ("FP16", "FP32"):
        raise ValueError("Expected FP16 or FP32 input")
    return {
        "kernel_fn": f"arm_arg{kind}_{'f16' if dtype == 'FP16' else 'f32'}",
        "input_c_type": "float16_t" if dtype == "FP16" else "float",
        "output_c_type": "int32_t",
    }


def generate_arg_extrema_float(op, output_dir, kind):
    kernel = float_arg_kernel(op, kind)
    half = op.tensor_dtype("input") == "FP16"
    dtype, word = (np.float16, np.uint16) if half else (np.float32, np.uint32)
    shape, axis = tuple(op.desc["input_shape"]), op.desc.get("axis", -1)
    if (
        len(shape) != 4
        or not isinstance(axis, (int, np.integer))
        or not 0 <= axis < 4
        or any(n <= 0 for n in shape)
    ):
        raise ValueError(
            "Standalone float arg fixtures require positive rank4 and canonical axis"
        )
    raw = op.desc.get("hint", {}).get("extras", {}).get("input_bits")
    if raw is None:
        sign = 1 if kind == "max" else -1
        values = np.full(shape, -sign, dtype=dtype)
        kept = tuple(1 if i == axis else n for i, n in enumerate(shape))
        positions = (0, shape[axis] // 2, shape[axis] - 1)
        for flat, coordinate in enumerate(np.ndindex(kept)):
            index = list(coordinate)
            # Cycle positions AND shift later groups: retained batches must
            # differ in winning indices, not merely in winning magnitudes.
            index[axis] = (positions[flat % 3] + flat // 3) % shape[axis]
            values[tuple(index)] = sign * 2
        bits = values.view(word)
    else:
        if any(not isinstance(x, int) or not 0 <= x <= np.iinfo(word).max for x in raw):
            raise ValueError(
                "input_bits must be unsigned integers of the element width"
            )
        bits = np.asarray(raw, dtype=word).reshape(shape)
    from helia_core_tester.generation.reference.call import ReferenceCall

    expected = op.reference_golden(ReferenceCall(
        f"arg_{kind}_{'f16' if half else 'f32'}", {"axis": int(axis)},
        {"input": np.ascontiguousarray(bits.view(dtype))},
        {"output": tuple(n for i, n in enumerate(shape) if i != axis)}))
    builder = TemplateContextBuilder()
    suffix = f"arg{kind}"
    op._write_op_outputs(
        output_dir,
        suffix,
        f"BasicMathFunctions/{suffix}/{suffix}.h.j2",
        f"BasicMathFunctions/{suffix}/{suffix}.c.j2",
        {
            "name": op.desc["name"],
            "input_dims": builder.nhwc_to_cmsis_dims(shape),
            "axis": axis,
            "input_dtype": kernel["input_c_type"],
            "output_dtype": "int32_t",
            "kernel_fn": kernel["kernel_fn"],
            "output_size": expected.size,
            "expected_output_array": builder.format_array_as_c_literal(expected),
            "float_kernel": True,
            "validation_mode": "exact_int",
            "input_count": bits.size,
            "word_type": "uint16_t" if half else "uint32_t",
            "input_bits": [hex(int(x)) for x in bits.flat],
        },
        {
            "name": op.desc["name"],
            "operator": op.desc["operator"],
            "operator_name": suffix,
        },
    )


from helia_core_tester.generation.harness import ArrayLiteral, Declaration, HarnessInput  # noqa: E402
from helia_core_tester.generation.harness.registry import harness_pool  # noqa: E402
from helia_core_tester.generation.harness.simple import tensor_case_pool  # noqa: E402


def arg_extrema_argument_pool(context):
    """ArgMax/ArgMin: a float case keeps its input as storage bits and copies them into a local
    array in the test, so non-finite patterns reach the kernel unchanged."""
    n = context["name"]
    values = {"axis": context["axis"]}
    if not context.get("float_kernel"):
        return tensor_case_pool(context, values, dims=("input_dims",))
    bits = Declaration(f"{n}_input_bits", context["word_type"], ArrayLiteral("    " + ", ".join(context["input_bits"])),
                       array=True)
    prologue = (f"    {context['input_dtype']} {n}_input[{context['input_count']}];\n"
                f"    memcpy({n}_input, {n}_input_bits, sizeof({n}_input));")
    pool = tensor_case_pool({**context, "input_data_array": "", "expected_output_array": context["expected_output_array"]},
                            values, dims=("input_dims",), includes=("<string.h>",), test_prologue=prologue,
                            inputs=(HarnessInput("input_data", "input", f"{n}_input"),))
    header = [d for d in pool.header if d.name != f"{n}_input"]
    header.insert(1, bits)
    from dataclasses import replace

    return replace(pool, header=header)


harness_pool("BasicMathFunctions/argmax/argmax.c.j2", label="ArgMax")(arg_extrema_argument_pool)
harness_pool("BasicMathFunctions/argmin/argmin.c.j2", label="ArgMin")(arg_extrema_argument_pool)
