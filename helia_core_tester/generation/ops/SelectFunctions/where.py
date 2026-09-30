"""Where operation implementation."""
from pathlib import Path

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpWhere(OperationBase):
    """Where operation - returns coordinates of non-zero elements."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Where uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import (
            LiteRtSingleOpBuilder, TensorSpec, _default_quant,
        )
        import ai_edge_litert.schema_py_generated as litert

        activation_dtype = self.desc.get("activation_dtype", "S8")
        tensor_type = litert.TensorType.INT16 if activation_dtype == "S16" else litert.TensorType.INT8
        input_shape = tuple(self.desc["input_shape"])
        total_elements = int(np.prod(input_shape))
        rank = len(input_shape)

        builder = LiteRtSingleOpBuilder(op_name="WHERE")
        input_idx = builder.add_tensor(TensorSpec(
            name="condition", shape=input_shape, tensor_type=tensor_type, is_input=True,
            quantization=_default_quant(tensor_type),
        ))
        # Output is dynamic: max shape is [total_elements, rank]
        output_idx = builder.add_tensor(TensorSpec(
            name="output", shape=(total_elements, rank), tensor_type=litert.TensorType.INT64, is_output=True,
        ))
        builder.add_operator("WHERE", inputs=[input_idx], outputs=[output_idx],
            options=None, options_type=litert.BuiltinOptions.NONE)
        self._write_tflite_bytes(out_path, builder.build())

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

        # Use TFLite interpreter for reference output; fall back to INT32 model if type unsupported
        tflite_path = str(output_dir / f"{name}.tflite")
        try:
            interpreter = self.load_litert_interpreter(tflite_path)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], condition)
            interpreter.invoke()
            output_data = np.array(interpreter.get_tensor(output_details[0]["index"]), dtype=np.int64)
        except (ValueError, RuntimeError):
            # Rebuild with INT32 condition (WHERE doesn't support INT16)
            from ai_edge_litert.interpreter import Interpreter
            from helia_core_tester.generation.utils.litert_builder import LiteRtSingleOpBuilder, TensorSpec
            import ai_edge_litert.schema_py_generated as litert
            b = LiteRtSingleOpBuilder(op_name="WHERE")
            i_idx = b.add_tensor(TensorSpec(name="condition", shape=tuple(input_shape), tensor_type=litert.TensorType.INT32, is_input=True))
            o_idx = b.add_tensor(TensorSpec(name="output", shape=(total_elements, rank), tensor_type=litert.TensorType.INT64, is_output=True))
            b.add_operator("WHERE", inputs=[i_idx], outputs=[o_idx], options=None, options_type=litert.BuiltinOptions.NONE)
            interp = Interpreter(model_content=bytes(b.build()))
            interp.allocate_tensors()
            inp_d = interp.get_input_details()
            out_d = interp.get_output_details()
            interp.set_tensor(inp_d[0]["index"], condition.astype(np.int32))
            interp.invoke()
            output_data = np.array(interp.get_tensor(out_d[0]["index"]), dtype=np.int64)
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
