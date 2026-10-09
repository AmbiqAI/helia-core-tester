"""
ReduceMin operation implementation.
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.reduce_extrema_integer import (
    boundary_inputs, reduce_extrema_golden,
)
from helia_core_tester.generation.io.dtypes import descriptor_dtype_to_c_type
from helia_core_tester.generation.ops._shared.reduce_extrema_float import generate_reduce_extrema_float


_KERNEL_BY_DTYPE = {
    "S8": "arm_reduce_min_s8",
    "S16": "arm_reduce_min_s16",
    "FP32": "arm_reduce_min_f32",
    "FP16": "arm_reduce_min_f16",
}


class OpReduceMin(OperationBase):
    """
    ReduceMin operation.
    """
    
    def _element_dtype(self) -> str:
        dtype = self.tensor_dtype("input")
        if self.tensor_dtype("output", default=dtype) != dtype:
            raise ValueError("Reduce extrema requires matching input/output dtypes")
        if dtype not in _KERNEL_BY_DTYPE:
            raise NotImplementedError(f"Unsupported reduce extrema dtype: {dtype}")
        return dtype

    def needs_tflite(self) -> bool:
        # Input and output share quantization, so the golden is a selection over
        # the raw codes (numpy); the float path has its own numpy golden.
        return False

    def _select_cmsis_reduce_min_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for ReduceMin operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        dtype = self._element_dtype()
        c_type = descriptor_dtype_to_c_type(dtype)
        return {
            "kernel_fn": _KERNEL_BY_DTYPE[dtype],
            "input_c_type": c_type,
            "output_c_type": c_type,
        }

    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for ReduceMin operation.
        """
        if self._element_dtype() in ("FP32", "FP16"):
            generate_reduce_extrema_float(
                self, output_dir, "min", self._select_cmsis_reduce_min_kernel()
            )
            return

        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        kernel_info = self._select_cmsis_reduce_min_kernel()
        input_shape = tuple(int(d) for d in self.desc["input_shape"])
        np_dtype = np.int16 if self._element_dtype() == "S16" else np.int8

        builder = TemplateContextBuilder()
        
        # Convert input shape to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        
        # Extract axes from descriptor (default to [1, 2] for spatial dimensions)
        axes = self.desc.get('axes', [1, 2])
        if not isinstance(axes, list):
            axes = [axes]
        
        # Build CMSIS reduction dims from input + axis semantics.
        # Keep CMSIS output dims 4D even when TFLite output rank is reduced (keepdims=false).
        axis_dims_cmsis = builder.build_reduce_axis_dims(len(input_shape), axes)
        output_dims = builder.build_reduce_output_dims(
            input_shape=input_shape,
            axes=axes,
            keepdims=bool(self.desc.get('keepdims', True))
        )
        
        # Work in integer codes so boundary separation survives quantization.
        input_q = boundary_inputs(input_shape, axes, np_dtype, "min")
        output_data = reduce_extrema_golden(input_q, axes, "min", keepdims=bool(self.desc.get('keepdims', True)))

        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'axis_dims': axis_dims_cmsis,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'ReduceMin'),
            'operator_name': 'reduce_min'
        }
        self._write_op_outputs(
            output_dir,
            "reduce_min",
            "BasicMathFunctions/reduce_min/reduce_min.h.j2",
            "BasicMathFunctions/reduce_min/reduce_min.c.j2",
            context,
            cmake_context,
        )
        
