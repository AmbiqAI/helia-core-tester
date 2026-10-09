"""
ReduceMax operation implementation.
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.reduce_extrema_integer import (
    boundary_inputs,
)
from helia_core_tester.generation.io.dtypes import descriptor_dtype_to_c_type
from helia_core_tester.generation.ops._shared.reduce_extrema_float import generate_reduce_extrema_float


_KERNEL_BY_DTYPE = {
    "S8": "arm_reduce_max_s8",
    "S16": "arm_reduce_max_s16",
    "FP32": "arm_reduce_max_f32",
    "FP16": "arm_reduce_max_f16",
}


class OpReduceMax(OperationBase):
    """
    ReduceMax operation.
    """
    
    def _element_dtype(self) -> str:
        dtype = self.tensor_dtype("input")
        if self.tensor_dtype("output", default=dtype) != dtype:
            raise ValueError("Reduce extrema requires matching input/output dtypes")
        if dtype not in _KERNEL_BY_DTYPE:
            raise NotImplementedError(f"Unsupported reduce extrema dtype: {dtype}")
        return dtype

    def uses_reference(self) -> bool:
        return True

    def _select_cmsis_reduce_max_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for ReduceMax operation.
        
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
        Generate C and H files from templates for ReduceMax operation.
        """
        if self._element_dtype() in ("FP32", "FP16"):
            generate_reduce_extrema_float(
                self, output_dir, "max", self._select_cmsis_reduce_max_kernel()
            )
            return

        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        from helia_core_tester.generation.reference.call import ReferenceCall

        name = self.desc['name']
        kernel_info = self._select_cmsis_reduce_max_kernel()
        input_shape = tuple(int(d) for d in self.desc["input_shape"])
        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        axes = self.desc.get('axes', [1, 2])
        if not isinstance(axes, list):
            axes = [axes]
        # Keep CMSIS output dims 4D even when the reduced rank drops (keepdims=false).
        axis_dims_cmsis = builder.build_reduce_axis_dims(len(input_shape), axes)
        output_dims = builder.build_reduce_output_dims(
            input_shape=input_shape,
            axes=axes,
            keepdims=bool(self.desc.get('keepdims', True))
        )
        # Selection works on raw codes (input and output share a quantization), so the
        # inputs place a unique winner at a domain boundary.
        np_dtype = np.int16 if kernel_info["input_c_type"] == "int16_t" else np.int8
        input_q = boundary_inputs(input_shape, axes, np_dtype, "max")
        norm = sorted({int(a) % len(input_shape) for a in axes})
        output_data = self.reference_golden(ReferenceCall(
            f"reduce_max_{'s16' if np_dtype == np.int16 else 's8'}", {"axis_mask": sum(1 << a for a in norm)},
            {"input": np.ascontiguousarray(input_q)},
            {"output": tuple(1 if i in norm else n for i, n in enumerate(input_shape))}))

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
            'operator': self.desc.get('operator', 'ReduceMax'),
            'operator_name': 'reduce_max'
        }
        self._write_op_outputs(
            output_dir,
            "reduce_max",
            "BasicMathFunctions/reduce_max/reduce_max.h.j2",
            "BasicMathFunctions/reduce_max/reduce_max.c.j2",
            context,
            cmake_context,
        )
        
