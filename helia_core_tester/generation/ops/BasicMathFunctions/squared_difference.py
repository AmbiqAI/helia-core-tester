"""SquaredDifference operation implementation."""

from pathlib import Path
from helia_core_tester.generation.ops._shared.binary_basic_math_base import BinaryBasicMathBase
from helia_core_tester.generation.utils.litert_builder import build_binary_broadcast_op
from typing import Dict
import numpy as np


SQUARED_DIFFERENCE_QUANT_PRESETS = {
    "int8": {
        "input_1_quant": ([1.0 / 128.0], [-128]),
        "input_2_quant": ([1.0 / 256.0], [-128]),
        "output_quant": ([1.0 / 64.0], [-128]),
    },
    "int16": {
        "input_1_quant": ([1.0 / 32768.0], [0]),
        "input_2_quant": ([1.0 / 65536.0], [0]),
        "output_quant": ([1.0 / 32768.0], [0]),
    },
}


def build_squared_difference_op(*, input_1_shape, input_2_shape, dtype: str = "int8") -> bytes:
    quant = SQUARED_DIFFERENCE_QUANT_PRESETS[dtype]
    return build_binary_broadcast_op(
        op_name="SQUARED_DIFFERENCE",
        input_1_shape=input_1_shape,
        input_2_shape=input_2_shape,
        dtype=dtype,
        input_1_quant=quant["input_1_quant"],
        input_2_quant=quant["input_2_quant"],
        output_quant=quant["output_quant"],
    )


class OpSquaredDifference(BinaryBasicMathBase):
    """SquaredDifference operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("SquaredDifference uses LiteRT-only model generation.")

    def _select_cmsis_squared_difference_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for SquaredDifference operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_squared_difference_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            call_style = self.desc.get("hint", {}).get("call_style", "")
            if str(call_style).lower() == "elementwise":
                return {
                    'kernel_fn': 'arm_elementwise_squared_difference_s16',
                    'input_c_type': 'int16_t',
                    'output_c_type': 'int16_t'
                }
            return {
                'kernel_fn': 'arm_squared_difference_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported SquaredDifference dtype: {activation_dtype}")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        else:
            raise NotImplementedError(f"Unsupported SquaredDifference dtype: {activation_dtype}")
        model_bytes = build_squared_difference_op(
            input_1_shape=tuple(self.desc["input_1_shape"]),
            input_2_shape=tuple(self.desc["input_2_shape"]),
            dtype=dtype,
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for SquaredDifference operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import (
            scalar_scale_zp,
            activation_bounds,
            elementwise_squared_difference_quant_params,
        )
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_squared_difference_kernel()
        
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
        
        # Extract quantization from LiteRT
        input1_quant = op_tensors['inputs'][0]['quantization']
        input2_quant = op_tensors['inputs'][1]['quantization'] if len(op_tensors['inputs']) > 1 else input1_quant
        output_quant = op_tensors['outputs'][0]['quantization']
        
        input1_scale, input1_zp = scalar_scale_zp(input1_quant)
        input2_scale, input2_zp = scalar_scale_zp(input2_quant)
        output_scale, output_zp = scalar_scale_zp(output_quant)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input1_dims = builder.nhwc_to_cmsis_dims(input1_shape)
        input2_dims = builder.nhwc_to_cmsis_dims(input2_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        activation_min, activation_max = activation_bounds(activation_dtype)
        sqdiff_qparams = elementwise_squared_difference_quant_params(
            input1_scale=float(input1_scale),
            input2_scale=float(input2_scale),
            output_scale=float(output_scale),
            activation_dtype=activation_dtype,
        )
        mult1 = sqdiff_qparams["input1_mult"]
        shift1 = sqdiff_qparams["input1_shift"]
        mult2 = sqdiff_qparams["input2_mult"]
        shift2 = sqdiff_qparams["input2_shift"]
        output_mult = sqdiff_qparams["out_mult"]
        output_shift = sqdiff_qparams["out_shift"]
        left_shift = sqdiff_qparams["left_shift"]
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input1_data = self.rng.uniform(-1.0, 1.0, size=input1_shape).astype(np.float32)
        input2_data = self.rng.uniform(-1.0, 1.0, size=input2_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        # Quantize inputs
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        input1_q = np.round(input1_data / float(input1_scale) + float(input1_zp)).astype(np.int32)
        input1_q = np.clip(input1_q, qmin, qmax).astype(np_in_dtype)
        
        input2_q = np.round(input2_data / float(input2_scale) + float(input2_zp)).astype(np.int32)
        input2_q = np.clip(input2_q, qmin, qmax).astype(np_in_dtype)
        
        # Run inference using LiteRT interpreter when shapes match (no broadcast).
        # LiteRT broadcasting can abort in some runtimes, so fall back to a local
        # quantized simulation for broadcasted shapes.
        if input1_shape == input2_shape:
            interpreter = self.load_litert_interpreter(str(tflite_path))
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'], input1_q)
            interpreter.set_tensor(input_details[1]['index'], input2_q)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = np.array(output_data)
        else:
            output_data = self._simulate_squared_difference_quantized(
                input1_q,
                input2_q,
                input1_offset=-int(input1_zp),
                input2_offset=-int(input2_zp),
                input1_mult=int(mult1),
                input1_shift=int(shift1),
                input2_mult=int(mult2),
                input2_shift=int(shift2),
                left_shift=int(left_shift),
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
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input1_dims': input1_dims,
            'input2_dims': input2_dims,
            'output_dims': output_dims,
            'input1_offset': -int(input1_zp),
            'input1_mult': int(mult1),
            'input1_shift': int(shift1),
            'input2_offset': -int(input2_zp),
            'input2_mult': int(mult2),
            'input2_shift': int(shift2),
            'left_shift': int(left_shift),
            'out_offset': int(output_zp),
            'out_mult': int(output_mult),
            'out_shift': int(output_shift),
            'out_activation_min': int(activation_min),
            'out_activation_max': int(activation_max),
            'block_size': int(np.prod(output_shape)),
            'call_style': str(self.desc.get("hint", {}).get("call_style", "")),
            'input1_data_array': input1_array_str,
            'input2_data_array': input2_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'SquaredDifference'),
            'operator_name': 'squared_difference'
        }
        self._write_op_outputs(output_dir, "squared_difference", "BasicMathFunctions/squared_difference/squared_difference.h.j2", "BasicMathFunctions/squared_difference/squared_difference.c.j2", context, cmake_context)

    @classmethod
    def _simulate_squared_difference_quantized(
        cls,
        input1_q: np.ndarray,
        input2_q: np.ndarray,
        *,
        input1_offset: int,
        input2_offset: int,
        input1_mult: int,
        input1_shift: int,
        input2_mult: int,
        input2_shift: int,
        left_shift: int,
        out_offset: int,
        out_mult: int,
        out_shift: int,
        out_activation_min: int,
        out_activation_max: int,
        out_dtype: np.dtype,
    ) -> np.ndarray:
        a = (input1_q.astype(np.int32) + int(input1_offset)) << int(left_shift)
        b = (input2_q.astype(np.int32) + int(input2_offset)) << int(left_shift)
        a = cls._requantize_np(a, int(input1_mult), int(input1_shift))
        b = cls._requantize_np(b, int(input2_mult), int(input2_shift))
        diff = (a - b) ** 2
        diff = cls._requantize_np(diff, int(out_mult), int(out_shift))
        diff = diff + int(out_offset)
        diff = np.clip(diff, int(out_activation_min), int(out_activation_max))
        return diff.astype(out_dtype)
