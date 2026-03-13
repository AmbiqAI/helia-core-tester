"""
Abs operation implementation.
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpAbs(OperationBase):
    """
    Abs operation.
    """

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Abs uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_abs_op

        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        else:
            raise NotImplementedError(f"Unsupported Abs dtype: {activation_dtype}")

        input_shape = tuple(self.desc["input_shape"])
        model_bytes = build_abs_op(input_shape=input_shape, dtype=dtype)
        with open(out_path, "wb") as f:
            f.write(model_bytes)

    def _select_cmsis_abs_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            return {
                "kernel_fn": "arm_abs_s8",
                "input_c_type": "int8_t",
                "output_c_type": "int8_t",
            }
        if activation_dtype == "S16":
            return {
                "kernel_fn": "arm_abs_s16",
                "input_c_type": "int16_t",
                "output_c_type": "int16_t",
            }
        raise NotImplementedError(f"Unsupported Abs dtype: {activation_dtype}")

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import (
            calculate_multiplier_shift,
            scalar_scale_zp,
            activation_bounds,
        )
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert

        name = self.desc["name"]
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        kernel_info = self._select_cmsis_abs_kernel()

        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)

        input_shape = self._ensure_shape_tuple(op_tensors["inputs"][0]["shape"])
        output_shape = self._ensure_shape_tuple(op_tensors["outputs"][0]["shape"])

        input_quant = op_tensors["inputs"][0]["quantization"]
        output_quant = op_tensors["outputs"][0]["quantization"]

        input_scale, input_zp = scalar_scale_zp(input_quant)
        output_scale, output_zp = scalar_scale_zp(output_quant)

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        activation_dtype = self.desc.get("activation_dtype", "S8")
        activation_min, activation_max = activation_bounds(activation_dtype)

        effective_scale = float(input_scale) / float(output_scale)
        output_mult, output_shift = calculate_multiplier_shift(effective_scale)
        needs_rescale = 0 if abs(effective_scale - 1.0) < 1e-6 else 1
        if bool(self.desc.get("hint", {}).get("force_rescale", False)):
            needs_rescale = 1

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        self.rng.__setstate__(rng_state)

        qmin, qmax = activation_bounds(activation_dtype)
        np_in_dtype = np.int16 if activation_dtype == "S16" else np.int8
        input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)

        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], input_q)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        output_data = np.array(output_data)

        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)

        block_size = int(np.prod(output_shape))

        context = {
            "name": name,
            "prefix": name,
            "input_dims": input_dims,
            "output_dims": output_dims,
            "input_offset": -int(input_zp),
            "output_offset": int(output_zp),
            "out_mult": int(output_mult),
            "out_shift": int(output_shift),
            "needs_rescale": int(needs_rescale),
            "out_activation_min": int(activation_min),
            "out_activation_max": int(activation_max),
            "block_size": int(block_size),
            "input_data_array": input_array_str,
            "expected_output_array": expected_output_array_str,
            "input_dtype": kernel_info["input_c_type"],
            "output_dtype": kernel_info["output_c_type"],
            "kernel_fn": kernel_info["kernel_fn"],
        }

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "Abs"),
            "operator_name": "abs",
        }
        self._write_op_outputs(output_dir, "abs", "abs/abs.h.j2", "abs/abs.c.j2", context, cmake_context)
