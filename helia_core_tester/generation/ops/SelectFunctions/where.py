"""Where operation implementation."""
from pathlib import Path

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpWhere(OperationBase):
    """Where operation - returns coordinates of non-zero elements."""

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            return {"kernel_fn": "arm_where_s16", "c_type": "int16_t", "cond_c_type": "int16_t", "np_dtype": "int16", "qmin": -32768, "qmax": 32767}
        return {"kernel_fn": "arm_where_s8", "c_type": "int8_t", "cond_c_type": "int8_t", "np_dtype": "int8", "qmin": -128, "qmax": 127}

    def generate_c_files(self, output_dir: _Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        ki = self._select_kernel()
        input_shape = list(self.desc["input_shape"])
        rank = len(input_shape)
        total_elements = int(np.prod(input_shape))

        rng = self._seeded_rng()
        np_dtype = np.int16 if ki["np_dtype"] == "int16" else np.int8
        # Generate condition with ~50% non-zero
        condition = rng.integers(-5, 6, size=input_shape, dtype=np_dtype)

        # Row-major coordinates of the non-zero elements, as TFLite WHERE emits them.
        output_data = np.argwhere(condition).astype(np.int64)
        num_true = output_data.shape[0]
        max_output_size = total_elements * rank  # worst case all true

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "rank": rank,
            "input_shape": input_shape,
            "total_elements": total_elements,
            "num_true": num_true,
            "max_output_size": max_output_size,
            "condition_array": builder.format_array_as_c_literal(condition),
            "expected_output_array": builder.format_array_as_c_literal(output_data.flatten()),
            "cond_c_type": ki["cond_c_type"],
            "output_c_type": "int64_t",
            "kernel_fn": ki["kernel_fn"],
        }

        self.render_harness_case(
            Path(output_dir), stem="where", context=context, pool=where_argument_pool(context),
            validation_key="SelectFunctions/where/where.c.j2", label="Where", operator="Where",
        )


from dataclasses import replace as _replace  # noqa: E402

from helia_core_tester.generation.harness import Declaration  # noqa: E402
from helia_core_tester.generation.harness.simple import shaped_case_pool  # noqa: E402

_WHERE_VALIDATION = """    /* Validate num_true matches expected */
    HELIA_VALIDATE_SCALAR_EQ_INT("Where", "num_true", {{ num_true }}, {{ name }}_num_true);

    int output_count = {{ name }}_num_true * {{ rank }};
    HELIA_VALIDATE_OUTPUTS(
        {{ validation_mode_token | default("TOLERANT_INT") }},
        {{ name }}_output,
        {{ name }}_expected_output,
        output_count,
        {{ validation_tolerance | default(0) }},
        {{ validation_atol }}f,
        {{ validation_rtol }}f,
        {{ validation_report_limit | default(20) }},
        failures
    );"""


def where_argument_pool(context):
    """Where writes a variable number of coordinates and reports the count through num_true."""
    n = context["name"]
    pool = shaped_case_pool(
        {**context, "c_type": context["cond_c_type"]}, shapes=(("shape", "input_shape"),),
        params_type="cmsis_nn_where_params", params={"rank": context["rank"], "shape": f"{n}_shape"},
        inputs=(("condition", "condition", "condition_array"),), output_ctype=context["output_c_type"],
        output_count=str(context["max_output_size"]), values={"num_true": f"&{n}_num_true"},
        validation=_WHERE_VALIDATION)
    count = Declaration(f"{n}_num_true", "int32_t", "0", storage="static")
    return _replace(pool, source=(count,))
