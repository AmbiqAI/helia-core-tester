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
