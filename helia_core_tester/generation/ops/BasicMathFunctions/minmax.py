"""Maximum and Minimum operation implementation."""

from pathlib import Path
from typing import Dict

import numpy as np

from helia_core_tester.generation.ops._shared.binary_basic_math_base import BinaryBasicMathBase
from helia_core_tester.generation.utils.litert_builder import build_binary_broadcast_op


def build_minmax_op(*, operator: str, input_1_shape, input_2_shape, dtype: str) -> bytes:
    """Build one LiteRT MINIMUM or MAXIMUM model."""
    op_name = operator.upper()
    if op_name not in {"MINIMUM", "MAXIMUM"}:
        raise ValueError(f"Unsupported operator: {operator}")
    return build_binary_broadcast_op(
        op_name=op_name,
        input_1_shape=input_1_shape,
        input_2_shape=input_2_shape,
        dtype=dtype,
    )


class OpMinMax(BinaryBasicMathBase):
    """Maximum and Minimum operation implementation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("MinMax uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        activation_dtype = self.tensor_dtype("input", default=self.desc.get("activation_dtype", "S8"))
        dtype_map = {
            "S8": "int8",
            "S16": "int16",
            "FP16": "float16",
            "FP32": "float32",
        }
        try:
            dtype = dtype_map[activation_dtype]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported MinMax dtype: {activation_dtype}") from exc

        model_bytes = build_minmax_op(
            operator=self.desc.get("operator", "Maximum"),
            input_1_shape=tuple(self.desc["input_1_shape"]),
            input_2_shape=tuple(self.desc["input_2_shape"]),
            dtype=dtype,
        )
        self._write_tflite_bytes(out_path, model_bytes)
    
    def _select_cmsis_minmax_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for MinMax operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.tensor_dtype("input", default=self.desc.get('activation_dtype', 'S8'))
        op_name = self.desc.get('operator', 'Maximum')
        
        if activation_dtype == 'S8':
            if op_name == 'Minimum':
                kernel_fn = 'arm_minimum_s8'
            elif op_name == 'Maximum':
                kernel_fn = 'arm_maximum_s8'
            else:
                raise ValueError(f"Unsupported operator: {op_name}")
            return {
                'kernel_fn': kernel_fn,
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            if op_name == 'Minimum':
                kernel_fn = 'arm_minimum_s16'
            elif op_name == 'Maximum':
                kernel_fn = 'arm_maximum_s16'
            else:
                raise ValueError(f"Unsupported operator: {op_name}")
            return {
                'kernel_fn': kernel_fn,
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        elif activation_dtype == 'FP32':
            if op_name == 'Minimum':
                kernel_fn = 'arm_minimum_f32'
            elif op_name == 'Maximum':
                kernel_fn = 'arm_maximum_f32'
            else:
                raise ValueError(f"Unsupported operator: {op_name}")
            return {
                'kernel_fn': kernel_fn,
                'input_c_type': 'float',
                'output_c_type': 'float'
            }
        elif activation_dtype == 'FP16':
            if op_name == 'Minimum':
                kernel_fn = 'arm_minimum_f16'
            elif op_name == 'Maximum':
                kernel_fn = 'arm_maximum_f16'
            else:
                raise ValueError(f"Unsupported operator: {op_name}")
            return {
                'kernel_fn': kernel_fn,
                'input_c_type': 'float16_t',
                'output_c_type': 'float16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported MinMax dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for MinMax operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_minmax_kernel()
        op_name = self.desc.get('operator', 'Maximum')
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT (multi-input operator)
        input1_shape = op_tensors['inputs'][0]['shape']
        input2_shape = op_tensors['inputs'][1]['shape'] if len(op_tensors['inputs']) > 1 else input1_shape
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input1_shape is not None:
            input1_shape = tuple(input1_shape)
        if input2_shape is not None:
            input2_shape = tuple(input2_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input1_dims = builder.nhwc_to_cmsis_dims(input1_shape)
        input2_dims = builder.nhwc_to_cmsis_dims(input2_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Draw both operands from one RNG stream: reseeding per call would make
        # input1 == input2 and the golden collapse to min(x,x)/max(x,x) == x.
        input1_data, input2_data = self._sample_dual_uniform_inputs(input1_shape, input2_shape)

        float_kernel = kernel_info["input_c_type"] in {"float", "float16_t"}
        if float_kernel:
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            input1_q = input1_data.astype(float_dtype)
            input2_q = input2_data.astype(float_dtype)
            if op_name == "Maximum":
                output_data = np.maximum(input1_q, input2_q).astype(float_dtype)
            else:
                output_data = np.minimum(input1_q, input2_q).astype(float_dtype)
        elif kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
            input1_quant = op_tensors['inputs'][0]['quantization']
            input2_quant = op_tensors['inputs'][1]['quantization'] if len(op_tensors['inputs']) > 1 else input1_quant
            input1_scale = self._quant_param_scalar(input1_quant, "scale", 1.0)
            input1_zp = self._quant_param_scalar(input1_quant, "zero_point", 0)
            input2_scale = self._quant_param_scalar(input2_quant, "scale", 1.0)
            input2_zp = self._quant_param_scalar(input2_quant, "zero_point", 0)
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
            input1_quant = op_tensors['inputs'][0]['quantization']
            input2_quant = op_tensors['inputs'][1]['quantization'] if len(op_tensors['inputs']) > 1 else input1_quant
            input1_scale = self._quant_param_scalar(input1_quant, "scale", 1.0)
            input1_zp = self._quant_param_scalar(input1_quant, "zero_point", 0)
            input2_scale = self._quant_param_scalar(input2_quant, "scale", 1.0)
            input2_zp = self._quant_param_scalar(input2_quant, "zero_point", 0)
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        if not float_kernel:
            input1_q = np.round(input1_data / float(input1_scale) + float(input1_zp)).astype(np.int32)
            input1_q = np.clip(input1_q, qmin, qmax).astype(np_in_dtype)

            input2_q = np.round(input2_data / float(input2_scale) + float(input2_zp)).astype(np.int32)
            input2_q = np.clip(input2_q, qmin, qmax).astype(np_in_dtype)

            desc_input1_shape = tuple(self.desc.get("input_1_shape", input1_shape))
            desc_input2_shape = tuple(self.desc.get("input_2_shape", input2_shape))
            if input1_shape == input2_shape and desc_input1_shape == desc_input2_shape:
                interpreter = self.load_litert_interpreter(str(tflite_path))
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                in0_shape = tuple(input_details[0].get('shape', input1_q.shape))
                in1_shape = tuple(input_details[1].get('shape', input2_q.shape))

                if in0_shape == input1_q.shape and in1_shape == input2_q.shape:
                    try:
                        interpreter.set_tensor(input_details[0]['index'], input1_q)
                        interpreter.set_tensor(input_details[1]['index'], input2_q)
                        interpreter.invoke()
                        output_data = np.array(interpreter.get_tensor(output_details[0]['index']))
                    except (ValueError, RuntimeError):
                        output_data = np.maximum(input1_q, input2_q) if op_name == "Maximum" else np.minimum(input1_q, input2_q)
                else:
                    output_data = np.maximum(input1_q, input2_q) if op_name == "Maximum" else np.minimum(input1_q, input2_q)
            else:
                output_data = np.maximum(input1_q, input2_q) if op_name == "Maximum" else np.minimum(input1_q, input2_q)
        
        # Format arrays
        output_data, nonfinite_context = self.apply_nonfinite_policy(output_data)
        input1_array_str = builder.format_array_as_c_literal(input1_q)
        input2_array_str = builder.format_array_as_c_literal(input2_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'input1_dims': input1_dims,
            'input2_dims': input2_dims,
            'output_dims': output_dims,
            'input1_data_array': input1_array_str,
            'input2_data_array': input2_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'operator': op_name,
        }
        context.update(nonfinite_context)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'MinMax'),
            'operator_name': 'minmax'
        }
        self._write_op_outputs(output_dir, "minmax", "BasicMathFunctions/minmax/minmax.h.j2", "BasicMathFunctions/minmax/minmax.c.j2", context, cmake_context)
        
