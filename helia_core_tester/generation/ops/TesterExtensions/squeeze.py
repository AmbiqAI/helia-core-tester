"""
Squeeze operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.copy_pool import copy_argument_pool


class OpSqueeze(OperationBase):
    """
    Squeeze operation - removes dimensions of size 1.
    """
    
    def needs_tflite(self) -> bool:
        # A squeeze is a copy: the golden is the reshaped input, quantized from its own range.
        return False

    def _select_cmsis_squeeze_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Squeeze operation.
        
        Squeeze is similar to Reshape - it's just a memory copy with shape change.
        We use arm_reshape_s8 which is essentially a memcpy.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_reshape_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Squeeze dtype: {activation_dtype} (only S8 supported)")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Squeeze operation.
        
        Squeeze removes dimensions of size 1. Since it's just a shape/view change
        and the data layout doesn't change, we can use arm_reshape_s8 (which is
        essentially a memcpy) similar to how Reshape works.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_squeeze_kernel()
        
        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        # tf.squeeze semantics: no axes drops every size-1 axis; explicit axes index the
        # full shape but never the batch axis.
        axes = self.desc.get('axes', None)
        rank = len(input_shape)
        if axes is None:
            squeezed = {i for i in range(rank) if input_shape[i] == 1}
        else:
            squeezed = {int(a) % rank for a in axes}
            if 0 in squeezed or any(input_shape[a] != 1 for a in squeezed):
                raise ValueError(f"{name}: cannot squeeze axes {axes} of {input_shape}")
        output_shape = tuple(d for i, d in enumerate(input_shape) if i not in squeezed) or (1,)

        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Calculate total size (should be same for input and output)
        total_size = int(np.prod(input_shape))
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        from helia_core_tester.generation.reference import policy

        quant = policy.descriptor_quant(self.desc.get("quantization", {}).get("input"), "s8")
        quant = quant or policy.activation_quant(input_data, "s8")
        input_scale, input_zp = quant.scale, quant.zero_point

        # Quantize inputs
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
        
        output_data = np.reshape(input_q, output_shape)

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
            output_dir, stem="squeeze", context=context, pool=copy_argument_pool(context),
            validation_key="TesterExtensions/squeeze/squeeze.c.j2", label="Squeeze", operator="Squeeze",
        )
        
