"""
LeakyRelu operation implementation with CMSIS-NN matching golden output.
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness.simple import tensor_case_pool


def leaky_relu_values(context: Dict[str, Any]) -> Dict[str, Any]:
    """The kernel's scalars by parameter name: the alpha and identity requantisation pairs."""
    return {"input_offset": context["input_offset"], "output_offset": context["output_offset"],
            "output_multiplier_alpha": context["output_mult_alpha"], "output_shift_alpha": context["output_shift_alpha"],
            "output_multiplier_identity": context["output_mult_identity"],
            "output_shift_identity": context["output_shift_identity"], "output_size": context["output_size"]}


class OpLeakyRelu(OperationBase):
    """
    LeakyRelu operation.
    """
    
    def uses_reference(self) -> bool:
        return True

    def _select_cmsis_leaky_relu_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for LeakyRelu operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_leaky_relu_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_leaky_relu_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported LeakyRelu dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for LeakyRelu operation.
        """
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference import quant as ref_quant
        from helia_core_tester.generation.reference.bindings import get_bindings
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_info = self._select_cmsis_leaky_relu_kernel()
        alpha = float(self.desc.get('alpha', 0.1))
        input_shape = output_shape = tuple(int(d) for d in self.desc["input_shape"])
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        kind = ref_quant.kind(activation_dtype)

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        # The input covers the [-1, 1] draw; the output covers LeakyRelu of it.
        in_quant = self.activation_quant("input", (-1.0, 1.0), kind)
        out_quant = self.activation_quant("output", (min(0.0, -alpha), max(1.0, -alpha)), kind)
        params = get_bindings().prepare("leaky_relu_prepare", {
            "dtype": ref_quant.hct_dtype(activation_dtype), "alpha": alpha,
            "input_scale": in_quant.scale, "input_zero_point": in_quant.zero_point,
            "output_scale": out_quant.scale, "output_zero_point": out_quant.zero_point,
        })
        input_zp, output_zp = params["input_offset"], params["output_offset"]
        mult_alpha, shift_alpha = params["alpha_multiplier"], params["alpha_shift"]
        mult_identity, shift_identity = params["identity_multiplier"], params["identity_shift"]

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        self.rng.__setstate__(rng_state)
        input_q = policy.quantize(input_data, in_quant)
        output_data = self.reference_golden(ReferenceCall(
            f"leaky_relu_{kind}", params, {"input": np.ascontiguousarray(input_q)}, {"output": output_shape},
            quant={"input": in_quant.to_json(), "output": out_quant.to_json(), "alpha": alpha},
        ))

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
            'output_mult_alpha': int(mult_alpha),
            'output_shift_alpha': int(shift_alpha),
            'output_mult_identity': int(mult_identity),
            'output_shift_identity': int(shift_identity),
            'output_size': int(output_size),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        self.render_harness_case(
            output_dir, stem="leaky_relu", context=context,
            pool=tensor_case_pool(context, leaky_relu_values(context)),
            validation_key="ActivationFunctions/leaky_relu/leaky_relu.c.j2", label="LeakyReLU", operator="LeakyRelu",
        )
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")
