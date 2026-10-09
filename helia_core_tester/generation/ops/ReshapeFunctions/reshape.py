"""
Reshape operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.copy_pool import copy_argument_pool


class OpReshape(OperationBase):
    """
    Reshape operation.
    """

    def _select_cmsis_reshape_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Reshape operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.tensor_dtype("input")
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_reshape_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'FP32':
            return {
                'kernel_fn': 'arm_reshape_f32',
                'input_c_type': 'float',
                'output_c_type': 'float'
            }
        elif activation_dtype == 'FP16':
            return {
                'kernel_fn': 'arm_reshape_f16',
                'input_c_type': 'float16_t',
                'output_c_type': 'float16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Reshape dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Reshape operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_reshape_kernel()
        
        input_shape = tuple(self.desc['input_shape'])
        output_shape = tuple(self.desc.get('target_shape'))
        if output_shape is None:
            raise ValueError("Reshape operation requires 'target_shape' in descriptor")
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Calculate total size (should be same for input and output)
        total_size = int(np.prod(input_shape))
        
        # Generate input data
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        if kernel_info["input_c_type"] in {"float", "float16_t"}:
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            input_q = self._sample_uniform(input_shape, dtype=float_dtype)
        else:
            input_q = self.rng.integers(-128, 128, size=input_shape, dtype=np.int8)
        self.rng.__setstate__(rng_state)

        output_data = input_q.reshape(output_shape)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'total_size': total_size,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        self.render_harness_case(
            output_dir, stem="reshape", context=context, pool=copy_argument_pool(context),
            validation_key="ReshapeFunctions/reshape/reshape.c.j2", label="Reshape", operator="Reshape",
        )
        
