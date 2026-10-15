"""Float DIV tests against LiteRT, retaining compact broadcast operands."""

from pathlib import Path

import numpy as np

from helia_core_tester.generation.ops._shared.binary_basic_math_base import BinaryBasicMathBase
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


class OpDiv(BinaryBasicMathBase):
    """Generate flat and NHWC broadcast division for FP32 and FP16 storage."""

    def validate_descriptor(self) -> None:
        self._dtype()
        shapes = [tuple(self.desc[key]) for key in ("input_1_shape", "input_2_shape")]
        if any(not 1 <= len(shape) <= 4 or any(dim <= 0 for dim in shape) for shape in shapes):
            raise ValueError("Div requires positive shapes of rank 1 through 4")
        np.broadcast_shapes(*shapes)
        dtype = self._dtype()
        lo = dtype.type(self.desc.get("act_min", -1000))
        hi = dtype.type(self.desc.get("act_max", 1000))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise ValueError("Div requires finite ordered activation bounds")

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Div uses LiteRT-only model generation.")

    def _dtype(self):
        dtype = self.tensor_dtype("input")
        if dtype not in ("FP16", "FP32") or self.tensor_dtype("output") != dtype:
            raise ValueError("Div requires matching FP16 or FP32 input/output storage")
        return np.dtype("float16" if dtype == "FP16" else "float32")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        import ai_edge_litert.schema_py_generated as schema
        from helia_core_tester.generation.utils.litert_builder import build_binary_broadcast_op

        self.validate_descriptor()
        options = schema.DivOptionsT()
        options.fusedActivationFunction = schema.ActivationFunctionType.NONE
        # LiteRT executes FP32 division. Half tests first round the operands to
        # FP16, widen for the oracle, then round its quotient back to FP16.
        data = build_binary_broadcast_op(
            op_name="DIV",
            input_1_shape=self.desc["input_1_shape"],
            input_2_shape=self.desc["input_2_shape"],
            dtype="float32",
            options=options,
            options_type=schema.BuiltinOptions.DivOptions,
        )
        self._write_tflite_bytes(out_path, data)

    def generate_c_files(self, output_dir: Path) -> None:
        self.validate_descriptor()
        dtype = self._dtype()
        suffix = "f16" if dtype == np.dtype("float16") else "f32"
        c_type = "float16_t" if suffix == "f16" else "float"
        name = self.desc["name"]
        shape_a = tuple(self.desc["input_1_shape"])
        shape_b = tuple(self.desc["input_2_shape"])
        shape_out = np.broadcast_shapes(shape_a, shape_b)
        broadcast = self._float_broadcast_call(auto_on_shape_mismatch=True)
        rng = self._seeded_rng()
        a = rng.uniform(-8, 8, shape_a).astype(dtype)
        b = rng.uniform(0.25, 4, shape_b).astype(dtype)
        b.flat[::2] *= -1
        interpreter = self.load_litert_interpreter(str(output_dir / f"{name}.tflite"))
        inputs = interpreter.get_input_details()
        interpreter.set_tensor(inputs[0]["index"], a.astype(np.float32))
        interpreter.set_tensor(inputs[1]["index"], b.astype(np.float32))
        interpreter.invoke()
        expected = interpreter.get_tensor(interpreter.get_output_details()[0]["index"]).astype(dtype)
        # Check the runtime oracle against independently evaluated binary64
        # quotients; all operands here are finite representable storage values.
        independent = (a.astype(np.float64) / b.astype(np.float64)).astype(dtype)
        np.testing.assert_array_equal(expected, independent)
        lo = dtype.type(self.desc.get("act_min", -1000))
        hi = dtype.type(self.desc.get("act_max", 1000))
        expected = np.clip(expected, lo, hi).astype(dtype)
        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "input1_dims": builder.nhwc_to_cmsis_dims((1,) * (4 - len(shape_a)) + shape_a),
            "input2_dims": builder.nhwc_to_cmsis_dims((1,) * (4 - len(shape_b)) + shape_b),
            "output_dims": builder.nhwc_to_cmsis_dims((1,) * (4 - len(shape_out)) + shape_out),
            "input1_data_array": builder.format_array_as_c_literal(a),
            "input2_data_array": builder.format_array_as_c_literal(b),
            "expected_output_array": builder.format_array_as_c_literal(expected),
            "input_dtype": c_type,
            "output_dtype": c_type,
            "block_size": int(np.prod(shape_out)),
            "kernel_fn": f"arm_elementwise_div_{'broadcast_' if broadcast else ''}{suffix}",
            "float_kernel": True,
            "float_broadcast": broadcast,
            "out_activation_min_literal": builder.format_float_literal(float(lo)),
            "out_activation_max_literal": builder.format_float_literal(float(hi)),
            "validation_label": "Div",
            "validation_mode": "float",
            "validation_atol": 0.0,
            "validation_rtol": 0.0,
            "expected_status": self.expected_status(),
        }
        # Mul's existing template accepts a binary float kernel and its call
        # style; reusing it keeps the guarded invocation and assertion in one place.
        self._write_op_outputs(
            output_dir, "mul", "BasicMathFunctions/mul/mul.h.j2",
            "BasicMathFunctions/mul/mul.c.j2", context,
            {"name": name, "operator": "Div", "operator_name": "mul"},
        )
