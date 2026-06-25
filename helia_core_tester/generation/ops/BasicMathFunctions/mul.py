"""
Multiply (elementwise) operation implementation for Helia-Core Tester.
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.binary_basic_math_base import BinaryBasicMathBase 


class OpMul(BinaryBasicMathBase):
    """
    Mul operation.
    """
    
    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Mul uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_mul_op

        activation_dtype = self.tensor_dtype("input")
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        elif activation_dtype == "FP32":
            dtype = "float32"
        elif activation_dtype == "FP16":
            dtype = "float16"
        else:
            raise NotImplementedError(f"Unsupported Mul dtype: {activation_dtype}")

        input_1_shape = tuple(self.desc["input_1_shape"])
        input_2_shape = tuple(self.desc["input_2_shape"])

        model_bytes = build_mul_op(
            input_1_shape=input_1_shape,
            input_2_shape=input_2_shape,
            dtype=dtype,
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def _select_cmsis_mul_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Mul operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.tensor_dtype("input")
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_mul_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'float_kernel': False,
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_mul_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t',
                'float_kernel': False,
            }
        elif activation_dtype == 'FP32':
            return {
                'kernel_fn': 'arm_elementwise_mul_f32',
                'input_c_type': 'float',
                'output_c_type': 'float',
                'float_kernel': True,
            }
        elif activation_dtype == 'FP16':
            return {
                'kernel_fn': 'arm_elementwise_mul_f16',
                'input_c_type': 'float16_t',
                'output_c_type': 'float16_t',
                'float_kernel': True,
            }
        else:
            raise NotImplementedError(f"Unsupported Mul dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Mul operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import (
    calculate_multiplier_shift,
    scalar_scale_zp,
    activation_bounds,
)
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_mul_kernel()
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        input1_shape = self._ensure_shape_tuple(op_tensors['inputs'][0]['shape'])
        input2_shape = self._ensure_shape_tuple(
            op_tensors['inputs'][1]['shape'] if len(op_tensors['inputs']) > 1 else op_tensors['inputs'][0]['shape']
        )
        output_shape = self._ensure_shape_tuple(op_tensors['outputs'][0]['shape'])
        
        builder = TemplateContextBuilder()
        input1_dims = builder.nhwc_to_cmsis_dims(input1_shape)
        input2_dims = builder.nhwc_to_cmsis_dims(input2_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        activation_dtype = self.tensor_dtype("input")
        if kernel_info["float_kernel"]:
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            input1_q = self._sample_uniform(input1_shape, dtype=float_dtype)
            input2_q = self._sample_uniform(input2_shape, dtype=float_dtype)
            activation_min = float(self.desc.get("act_min", -1.0e30))
            activation_max = float(self.desc.get("act_max", 1.0e30))
            output_data = np.clip(
                input1_q.astype(np.float32) * input2_q.astype(np.float32),
                activation_min,
                activation_max,
            ).astype(float_dtype)
            activation_min_literal = builder.format_float_literal(activation_min)
            activation_max_literal = builder.format_float_literal(activation_max)
            input1_zp = input2_zp = output_zp = output_mult = output_shift = 0
        else:
            input1_quant = op_tensors['inputs'][0]['quantization']
            input2_quant = op_tensors['inputs'][1]['quantization'] if len(op_tensors['inputs']) > 1 else input1_quant
            output_quant = op_tensors['outputs'][0]['quantization']

            input1_scale, input1_zp = scalar_scale_zp(input1_quant)
            input2_scale, input2_zp = scalar_scale_zp(input2_quant)
            output_scale, output_zp = scalar_scale_zp(output_quant)

            effective_scale = (float(input1_scale) * float(input2_scale)) / float(output_scale)
            output_mult, output_shift = calculate_multiplier_shift(effective_scale)
            activation_min, activation_max = activation_bounds(activation_dtype)
            input1_data = self._sample_uniform(input1_shape)
            input2_data = self._sample_uniform(input2_shape)
            qmin, qmax = activation_bounds(activation_dtype)
            np_in_dtype = np.int16 if activation_dtype == "S16" else np.int8
            input1_q = np.round(input1_data / float(input1_scale) + float(input1_zp)).astype(np.int32)
            input1_q = np.clip(input1_q, qmin, qmax).astype(np_in_dtype)

            input2_q = np.round(input2_data / float(input2_scale) + float(input2_zp)).astype(np.int32)
            input2_q = np.clip(input2_q, qmin, qmax).astype(np_in_dtype)

            if input1_shape == input2_shape:
                interpreter = self.load_litert_interpreter(str(tflite_path))
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], input1_q)
                interpreter.set_tensor(input_details[1]['index'], input2_q)
                interpreter.invoke()
                output_data = np.array(interpreter.get_tensor(output_details[0]['index']))
            else:
                output_data = self._simulate_mul_quantized(
                    input1_q,
                    input2_q,
                    input1_offset=-int(input1_zp),
                    input2_offset=-int(input2_zp),
                    out_offset=int(output_zp),
                    out_mult=int(output_mult),
                    out_shift=int(output_shift),
                    out_activation_min=int(activation_min),
                    out_activation_max=int(activation_max),
                    out_dtype=np_in_dtype,
                )
        
        # Format arrays
        input1_array_str = builder.format_array_as_c_literal(input1_q)
        input2_array_str = builder.format_array_as_c_literal(input2_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate block size (total number of elements)
        block_size = int(np.prod(output_shape))
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input1_dims': input1_dims,
            'input2_dims': input2_dims,
            'output_dims': output_dims,
            # For CMSIS-NN, input offsets should be negated (like other operations)
            # Output offset is used as-is (not negated)
            'input1_offset': -int(input1_zp),
            'input2_offset': -int(input2_zp),
            'out_offset': int(output_zp),
            'out_mult': int(output_mult),
            'out_shift': int(output_shift),
            'out_activation_min': int(activation_min),
            'out_activation_max': int(activation_max),
            'block_size': int(block_size),
            'input1_data_array': input1_array_str,
            'input2_data_array': input2_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'float_kernel': kernel_info["float_kernel"],
        }
        if kernel_info["float_kernel"]:
            context["out_activation_min_literal"] = activation_min_literal
            context["out_activation_max_literal"] = activation_max_literal
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Mul'),
            'operator_name': 'mul',
        }
        self._write_op_outputs(output_dir, "mul", "BasicMathFunctions/mul/mul.h.j2", "BasicMathFunctions/mul/mul.c.j2", context, cmake_context)

    @classmethod
    def _simulate_mul_quantized(
        cls,
        input1_q: np.ndarray,
        input2_q: np.ndarray,
        *,
        input1_offset: int,
        input2_offset: int,
        out_offset: int,
        out_mult: int,
        out_shift: int,
        out_activation_min: int,
        out_activation_max: int,
        out_dtype: np.dtype,
    ) -> np.ndarray:
        a = input1_q.astype(np.int32) + int(input1_offset)
        b = input2_q.astype(np.int32) + int(input2_offset)
        prod = a * b
        prod = cls._requantize_np(prod, int(out_mult), int(out_shift))
        prod = prod + int(out_offset)
        prod = np.clip(prod, int(out_activation_min), int(out_activation_max))
        return prod.astype(out_dtype)
