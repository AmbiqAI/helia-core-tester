"""
ReLU6 operation implementation for Helia-Core Tester.
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.relu_base import ReluFamilyBase
from helia_core_tester.generation.harness.simple import tensor_case_pool


def relu6_values(context: Dict[str, Any]) -> Dict[str, Any]:
    """The kernel's scalars by parameter name: the requantisation and the quantised [0, 6] range."""
    return {"input_offset": context["input_offset"], "output_offset": context["output_offset"],
            "output_multiplier": context["output_mult"], "output_shift": context["output_shift"],
            "act_min": context["act_min"], "act_max": context["act_max"], "output_size": context["output_size"]}


class OpRelu6(ReluFamilyBase):
    """
    Relu6 operation.
    """
    
    ACT_MAX = 6.0
    KERNEL_PREFIX = "relu6"

    def _select_cmsis_relu6_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Relu6 operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_relu_generic_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_relu_generic_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Relu6 dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Relu6 operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_info = self._select_cmsis_relu6_kernel()
        ref = self.relu_reference()
        params = ref["params"]
        input_shape = output_shape = ref["shape"]
        input_q, output_data = ref["input_q"], ref["output"]
        input_zp, output_zp = params["input_zero_point"], params["output_zero_point"]
        output_mult, output_shift = params["output_multiplier"], params["output_shift"]
        act_min, act_max = params["act_min"], params["act_max"]

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate output size (total number of elements)
        output_size = int(np.prod(output_shape))
        
        # Build template context
        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'input_offset': int(input_zp),
            'output_offset': int(output_zp),
            'output_mult': int(output_mult),
            'output_shift': int(output_shift),
            'act_min': int(act_min),
            'act_max': int(act_max),
            'output_size': int(output_size),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        self.render_harness_case(
            output_dir, stem="relu6", context=context, pool=tensor_case_pool(context, relu6_values(context)),
            validation_key="ActivationFunctions/relu6/relu6.c.j2", label="ReLU6", operator="Relu6",
        )
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")