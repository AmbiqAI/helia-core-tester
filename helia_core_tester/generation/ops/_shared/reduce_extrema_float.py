"""Raw-bit fixtures for the float reduce-extrema contract (ns-cmsis-nn#498)."""

from pathlib import Path

import numpy as np

from helia_core_tester.generation.ops._shared.reduce_extrema_reference import (
    reduce_extrema_reference,
)
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


def generate_reduce_extrema_float(
    op, output_dir: Path, kind: str, kernel_info: dict
) -> None:
    half = op.tensor_dtype("input") == "FP16"
    word, dtype = (np.uint16, np.float16) if half else (np.uint32, np.float32)
    shape = tuple(op.desc["input_shape"])
    axes = op.desc.get("axes", [1, 2])
    if not isinstance(axes, list):
        axes = [axes]
    raw = op.desc.get("hint", {}).get("extras", {}).get("input_bits")
    if raw is None:
        raise ValueError(
            "Float reduce extrema requires explicit hint.extras.input_bits"
        )
    if any(not isinstance(x, int) or x < 0 or x > np.iinfo(word).max for x in raw):
        raise ValueError(
            "input_bits must contain unsigned integers of the element width"
        )
    bits = np.asarray(raw, dtype=word).reshape(shape)
    expected = reduce_extrema_reference(bits.view(dtype), axes, kind).view(word)
    builder = TemplateContextBuilder()
    context = {
        "name": op.desc["name"],
        **kernel_info,
        "input_dtype": kernel_info["input_c_type"],
        "output_dtype": kernel_info["output_c_type"],
        "input_dims": builder.nhwc_to_cmsis_dims(shape),
        "axis_dims": builder.build_reduce_axis_dims(len(shape), axes),
        "output_dims": builder.build_reduce_output_dims(shape, axes, keepdims=True),
        "float_kernel": True,
        "validation_mode": "float",
        "word_type": "uint16_t" if half else "uint32_t",
        "infinity_bits": "0x7c00u" if half else "0x7f800000u",
        "input_count": bits.size,
        "input_bits": [hex(int(x)) for x in bits.flat] or ["0"],
        "expected_bits": [hex(int(x)) for x in expected.flat] or ["0"],
    }
    suffix = f"reduce_{kind}"
    op._write_op_outputs(
        output_dir,
        suffix,
        f"BasicMathFunctions/{suffix}/{suffix}.h.j2",
        f"BasicMathFunctions/{suffix}/{suffix}.c.j2",
        context,
        {
            "name": op.desc["name"],
            "operator": op.desc["operator"],
            "operator_name": suffix,
        },
    )


from helia_core_tester.generation.harness import ArrayLiteral, Declaration, GuardedBuffer, HarnessInput  # noqa: E402
from helia_core_tester.generation.harness.registry import harness_pool  # noqa: E402
from helia_core_tester.generation.harness.simple import dims_count, dims_declaration, tensor_case_pool  # noqa: E402

_REDUCE_DIMS = ("input_dims", "output_dims", "axis_dims")


def _reduce_extrema_pool(label):
    def pool(context):
        """ReduceMax/ReduceMin: a float case copies storage bits into a guarded input the kernel must
        not write, and checks every output element bit for bit (NaN payloads, signed zeros)."""
        n = context["name"]
        count = dims_count(context["output_dims"])
        if not context.get("float_kernel"):
            return tensor_case_pool(context, {}, dims=_REDUCE_DIMS, output_count=count)
        word, dtype, input_count = context["word_type"], context["input_dtype"], int(context["input_count"])
        header = [dims_declaration(f"{n}_{d}", context[d]) for d in _REDUCE_DIMS]
        header += [
            Declaration(f"{n}_input_bits", word, ArrayLiteral("    " + ", ".join(context["input_bits"])), array=True),
            Declaration(f"{n}_expected_bits", word, ArrayLiteral("    " + ", ".join(context["expected_bits"])), array=True),
        ]
        prologue = (f"    HELIA_GUARD_ARM({n}_input, false /* copied in below; the kernel must not write it */);\n"
                    f"    memcpy({n}_input, {n}_input_bits, {input_count} * sizeof({dtype}));")
        checks = f'    HELIA_GUARD_CHECK({n}_input, "{{{{ validation_label | default("{label}") }}}} input", failures);'
        validation = (f"    if (memcmp({n}_input, {n}_input_bits, {input_count} * sizeof({dtype})) != 0) ++failures;\n"
                      f"    for (int i = 0; i < {n.upper()}_OUTPUT_SIZE; ++i) {{\n"
                      f"        {word} actual;\n"
                      f"        memcpy(&actual, {n}_output + i, sizeof(actual));\n"
                      f"        HELIA_VALIDATE_FLOAT_BITS(actual, {n}_expected_bits[i], {context['infinity_bits']}, 0, i, 20, failures);\n"
                      f"    }}")
        from helia_core_tester.generation.harness import ArgumentPool

        values = {d: f"&{n}_{d}" for d in _REDUCE_DIMS}
        return ArgumentPool(
            name=n, values=values, header=header, output_count=count, benchmark=False, scratch_buffer=False,
            inputs=(HarnessInput("input_data", "input", "NULL" if input_count == 0 else f"{n}_input"),),
            guarded=(GuardedBuffer(f"{n}_input", dtype, str(input_count or 1)),), includes=("<string.h>",),
            test_prologue=prologue, extra_checks=checks, validation=validation, output_poison=True,
        )
    return pool


harness_pool("BasicMathFunctions/reduce_max/reduce_max.c.j2", label="ReduceMax")(_reduce_extrema_pool("ReduceMax"))
harness_pool("BasicMathFunctions/reduce_min/reduce_min.c.j2", label="ReduceMin")(_reduce_extrema_pool("ReduceMin"))
