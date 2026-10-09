"""Maximum and Minimum operation implementation."""

from pathlib import Path
from typing import Dict

import numpy as np

from helia_core_tester.generation.ops._shared.binary_basic_math_base import BinaryBasicMathBase


def _split_scalar(input1_q: np.ndarray, input2_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Set a scalar operand to the other's median."""
    if input1_q.size == 1 and input2_q.size > 1:
        return np.full_like(input1_q, np.median(input2_q)), input2_q
    if input2_q.size == 1 and input1_q.size > 1:
        return input1_q, np.full_like(input2_q, np.median(input1_q))
    return input1_q, input2_q


class OpMinMax(BinaryBasicMathBase):
    """Maximum and Minimum operation implementation."""

    SIGN_SPAN_OPERANDS = ("input_1", "input_2")

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
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_minmax_kernel()
        op_name = self.desc.get('operator', 'Maximum')
        input1_shape, input2_shape, output_shape = self._binary_shapes()

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

            def float_reference(operands, _dtype=float_dtype, _op=op_name):
                combine = np.maximum if _op == "Maximum" else np.minimum
                return combine(operands[0], operands[1]).astype(_dtype)

            output_data = float_reference([input1_q, input2_q])
        elif kernel_info["input_c_type"] in ("int8_t", "int16_t"):
            np_in_dtype = np.int8 if kernel_info["input_c_type"] == "int8_t" else np.int16
            qmin, qmax = np.iinfo(np_in_dtype).min, np.iinfo(np_in_dtype).max
            input1_scale, input1_zp = input2_scale, input2_zp = self._fixed_quant()
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        if not float_kernel:
            input1_data = self._widen_s8(input1_data, input1_scale, kernel_info["input_c_type"])
            input2_data = self._widen_s8(input2_data, input2_scale, kernel_info["input_c_type"])
            input1_q = np.round(input1_data / float(input1_scale) + float(input1_zp)).astype(np.int32)
            input1_q = np.clip(input1_q, qmin, qmax).astype(np_in_dtype)

            input2_q = np.round(input2_data / float(input2_scale) + float(input2_zp)).astype(np.int32)
            input2_q = np.clip(input2_q, qmin, qmax).astype(np_in_dtype)
            input1_q, input2_q = self._enforce_int_operand_sign_span(
                (("input_1", input1_q, input1_zp), ("input_2", input2_q, input2_zp)),
                steerable=("input_1", "input_2"),
            )
            input1_q, input2_q = _split_scalar(input1_q, input2_q)

            # Both operands share one quantization, so no requantization: TFLM's
            # MaximumMinimumBroadcastSlow is exactly the elementwise max/min.
            combine = np.maximum if op_name == "Maximum" else np.minimum
            output_data = combine(input1_q, input2_q).astype(np_in_dtype)

        # Format arrays
        if float_kernel:
            output_data, nonfinite_context = self.apply_nonfinite_policy(
                output_data, reference=float_reference, inputs=[input1_q, input2_q]
            )
        else:
            nonfinite_context = {}
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
        


from dataclasses import replace as _replace  # noqa: E402

from helia_core_tester.generation.harness.model import Declaration as _Declaration  # noqa: E402
from helia_core_tester.generation.harness.registry import harness_pool  # noqa: E402
from helia_core_tester.generation.harness.simple import binary_case_pool  # noqa: E402


@harness_pool("BasicMathFunctions/minmax/minmax.c.j2", label="Minmax")
def minmax_argument_pool(context):
    n = context["name"]
    pool = binary_case_pool(context)
    ctx = _Declaration(f"{n}_ctx", "cmsis_nn_context", {"buf": "NULL", "size": "0"}, storage="static")
    return _replace(pool, values={**pool.values, "ctx": f"&{n}_ctx"}, source=(ctx,), owns_ctx=True)
