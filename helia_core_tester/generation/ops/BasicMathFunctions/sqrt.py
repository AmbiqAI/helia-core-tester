"""Sqrt operation implementation."""

from math import sqrt
from pathlib import Path
from typing import Any, Dict

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.litert_builder import build_unary_same_shape_op
from helia_core_tester.generation.utils.litert_utils import (
    get_operator_tensors_from_litert,
    load_litert_model,
)


SQRT_S16_LUT_SIZE = 513
SQRT_S16_SLOT_SHIFT = 7
SQRT_S16_SLOT_HALF_STEP = 1 << (SQRT_S16_SLOT_SHIFT - 1)


def clamp_f32(x, min_val, max_val):
    '''Clamp a float32 value to the specified range.'''
    return max(min(x, max_val), min_val)

def _quant_param_to_scalar(value, name: str, cast):
    """Normalize LiteRT quantization values to a scalar."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Sqrt expects scalar quantization for {name}, got shape {arr.shape}")
    return cast(arr.reshape(-1)[0])


def _sqrt_quantized_real(real_value: float) -> float:
    """Return sqrt(real_value) while clamping the invalid negative domain to zero."""
    if real_value <= 0.0:
        return 0.0
    return float(sqrt(real_value))


def _quantize_s16_sqrt_output(real_value: float, output_scale: float, output_zp: int) -> int:
    """Match LiteRT's int16 sqrt output more closely with truncation-based requantization."""
    quantized_output = int(np.trunc(np.float32(real_value / np.float32(output_scale)))) + int(output_zp)
    return int(np.clip(quantized_output, -32768, 32767))

def make_sqrt_lut_s8(input_scale, input_zp, output_scale, output_zp) -> np.ndarray:
    """Generate the uint8-addressed LUT required by arm_sqrt_s8."""
    input_scale = _quant_param_to_scalar(input_scale, "input_scale", float)
    input_zp = _quant_param_to_scalar(input_zp, "input_zero_point", int)
    output_scale = _quant_param_to_scalar(output_scale, "output_scale", float)
    output_zp = _quant_param_to_scalar(output_zp, "output_zero_point", int)

    lut = np.zeros(256, dtype=np.int8)
    for i in range(-128, 128):
        final_val = output_zp
        x = np.float32(input_scale) * np.float32(i - input_zp)

        if x > np.float32(0.0):
            res = np.float32(sqrt(float(x)))
            quantized_output = int(np.trunc(np.float32(res / np.float32(output_scale)))) + int(
                output_zp
            )
            final_val = min(max(quantized_output, -128), 127)

        # Mimic C's (uint8_t)i indexing with two's complement wrap.
        lut[i & 0xFF] = np.int8(final_val)

    return lut


def make_sqrt_lut_s16(input_scale, input_zp, output_scale, output_zp) -> np.ndarray:
    """Generate the 513-entry interpolated LUT required by arm_sqrt_s16."""
    input_scale = _quant_param_to_scalar(input_scale, "input_scale", float)
    input_zp = _quant_param_to_scalar(input_zp, "input_zero_point", int)
    output_scale = _quant_param_to_scalar(output_scale, "output_scale", float)
    output_zp = _quant_param_to_scalar(output_zp, "output_zero_point", int)

    lut = np.zeros(SQRT_S16_LUT_SIZE, dtype=np.int16)
    for index in range(SQRT_S16_LUT_SIZE):
        q_value = -32768 + (index << SQRT_S16_SLOT_SHIFT)
        real_value = float(np.float32(input_scale) * np.float32(q_value - input_zp))
        compensated_sqrt = _sqrt_quantized_real(real_value)

        # arm_sqrt_s16 linearly interpolates between neighboring LUT anchors.
        # Bias interior positive anchors upward by the local midpoint sag of sqrt(x)
        # so the piecewise-linear approximation tracks the concave curve better.
        if 0 < index < (SQRT_S16_LUT_SIZE - 1) and real_value > 0.0:
            next_q_value = q_value + (1 << SQRT_S16_SLOT_SHIFT)
            midpoint_q_value = q_value + SQRT_S16_SLOT_HALF_STEP
            next_real_value = float(np.float32(input_scale) * np.float32(next_q_value - input_zp))
            midpoint_real_value = float(np.float32(input_scale) * np.float32(midpoint_q_value - input_zp))
            next_sqrt = _sqrt_quantized_real(next_real_value)
            midpoint_sqrt = _sqrt_quantized_real(midpoint_real_value)
            compensated_sqrt += midpoint_sqrt - 0.5 * (compensated_sqrt + next_sqrt)

        lut[index] = np.int16(
            _quantize_s16_sqrt_output(compensated_sqrt, output_scale, output_zp)
        )

    return lut


def make_sqrt_lut(input_scale, input_zp, output_scale, output_zp, activation_dtype: str) -> np.ndarray:
    """Generate a dtype-specific lookup table for the Sqrt operation."""
    if activation_dtype == "S8":
        return make_sqrt_lut_s8(input_scale, input_zp, output_scale, output_zp)
    if activation_dtype == "S16":
        return make_sqrt_lut_s16(input_scale, input_zp, output_scale, output_zp)
    raise NotImplementedError(f"Unsupported Sqrt dtype: {activation_dtype}")


def build_sqrt_op(
    *,
    input_shape,
    dtype: str = "int8",
) -> bytes:
    return build_unary_same_shape_op(
        op_name="SQRT",
        input_shape=input_shape,
        dtype=dtype,
    )

class OpSqrt(OperationBase):
    """
    Sqrt operation.
    """

    def needs_keras_model(self) -> bool:
        return False
    
    def build_keras_model(self):
        raise NotImplementedError("Sqrt uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        else:
            raise NotImplementedError(f"Unsupported Sqrt dtype: {activation_dtype}")

        input_shape = tuple(self.desc["input_shape"])
        model_bytes = build_sqrt_op(
            input_shape=input_shape,
            dtype=dtype,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)
    
    def _select_cmsis_sqrt_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Sqrt operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')

        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_sqrt_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'lut_c_type': 'int8_t',
                'lut_size': 256,
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_sqrt_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t',
                'lut_c_type': 'int16_t',
                'lut_size': SQRT_S16_LUT_SIZE,
            }
        else:
            raise NotImplementedError(f"Unsupported Sqrt dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Sqrt operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_sqrt_kernel()
        
        input_shape = tuple(self.desc["input_shape"])
        
        builder = TemplateContextBuilder()
        comparison = self.comparison_config()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        
        # Generate deterministic integer input data
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        input_q = self.rng.integers(0, qmax + 1, size=input_shape, dtype=np_in_dtype)
        self.rng.__setstate__(rng_state)

        model, subgraph = load_litert_model(str(tflite_path))
        
        # Get operator tensors (first operator)
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")
        
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)

        input_quant = op_tensors['inputs'][0]['quantization']
        output_quant = op_tensors['outputs'][0]['quantization']

        # Compute expected output directly
        output_data = self.run_inference(str(tflite_path), input_q)
        output_shape = tuple(output_data.shape)
        input_scale = input_quant['scale']
        input_zp = input_quant['zero_point']
        output_scale = output_quant['scale']
        output_zp = output_quant['zero_point'] 
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)


        activation_dtype = self.desc.get("activation_dtype", "S8")
        sqrt_lut = make_sqrt_lut(
            input_scale=input_scale,
            input_zp=input_zp,
            output_scale=output_scale,
            output_zp=output_zp,
            activation_dtype=activation_dtype,
        )

        # Build template context
        context = {
            'name': name,
            'input_dims': input_dims,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'output_size': int(np.prod(output_shape)),
            'lut_c_type': kernel_info["lut_c_type"],
            'lut_size': kernel_info["lut_size"],
            'sqrt_lut': sqrt_lut,
        }
        if comparison.get("mode") == "tolerant_int":
            context["validation_mode"] = "tolerant_int"
            context["comparison_tolerance"] = int(comparison.get("tolerance", 1))
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("BasicMathFunctions/sqrt/sqrt.h.j2", context)
        h_path = includes_api_dir / f"{name}_sqrt.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("BasicMathFunctions/sqrt/sqrt.c.j2", context)
        c_path = output_dir / f"{name}_sqrt.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Sqrt'),
            'operator_name': 'sqrt'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
