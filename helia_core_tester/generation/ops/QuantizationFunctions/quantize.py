"""
Quantize operation implementation for Helia-Core Tester.

Following the official CMSIS-NN test generator logic from RefactoredTestGen/Lib/op_quantize.py
"""

import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.quantization_base import QuantizationFamilyBase
from helia_core_tester.generation.ops.QuantizationFunctions.pools import quantize_argument_pool


class OpQuantize(QuantizationFamilyBase):
    """
    Quantize operation.
    """

    def uses_reference(self) -> bool:
        return True

    def primary_execution_dtype(self) -> str:
        """Quantize converts an FP32 input to a quantized output, so the
        conversion/quantization target dtype is the *output* dtype, not the
        input. The shared base implementation defaults to the input dtype
        (correct for unary/activation-like ops), which for Quantize is always
        FP32 -- this previously caused the shared TFLite converter to skip
        int8/int16 quantization entirely and emit a fully float32 model.
        """
        resolved = self.resolved_tensor_dtypes()
        if "output" in resolved:
            return resolved["output"]
        return super().primary_execution_dtype()
    
    def _select_cmsis_quantize_kernel(self) -> dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Quantize operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        input_dtype = self.tensor_dtype("input")
        output_dtype = self.tensor_dtype("output")

        if input_dtype != "FP32":
            raise NotImplementedError(f"Quantize currently requires FP32 input, got {input_dtype}")
        
        if output_dtype == 'S8':
            return {
                'kernel_fn': 'arm_quantize_f32_s8',
                'input_c_type': self.tensor_c_type("input"),
                'output_c_type': self.tensor_c_type("output"),
            }
        if output_dtype == 'S16':
            return {
                'kernel_fn': 'arm_quantize_f32_s16',
                'input_c_type': self.tensor_c_type("input"),
                'output_c_type': self.tensor_c_type("output"),
            }
        raise NotImplementedError(f"Unsupported Quantize output dtype: {output_dtype}")

    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Quantize operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        from helia_core_tester.generation.reference.case import ReferenceCall

        name = self.desc['name']
        kernel_info = self._select_cmsis_quantize_kernel()
        kind = "s8" if kernel_info["output_c_type"] == "int8_t" else "s16"

        builder = TemplateContextBuilder()

        # The harness applies the fused activation in float before the kernel quantizes.
        activation_str = self.desc.get('activation', 'NONE')
        if activation_str not in ('NONE', 'RELU', 'RELU6'):
            raise ValueError(f"Unsupported activation: {activation_str}")
        has_activation = activation_str in ['RELU', 'RELU6']
        comparison_tolerance = 1

        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        input_data = self._sample_uniform(input_shape).astype(np.float32)
        act_max = {"RELU": np.inf, "RELU6": 6.0}.get(activation_str)
        activated = input_data if act_max is None else np.clip(input_data, 0.0, act_max).astype(np.float32)
        # The output quantization covers the activated [-1, 1] draw range.
        draw_range = np.array([-1.0, 1.0], dtype=np.float32)
        out_quant = self.activation_quant(
            "output", draw_range if act_max is None else np.clip(draw_range, 0.0, act_max), kind)
        output_scale, output_zp = out_quant.scale, out_quant.zero_point
        call = ReferenceCall(
            f"quantize_{kind}", {"scale": output_scale, "zero_point": output_zp},
            {"input": np.ascontiguousarray(activated)}, input_shape, "int8" if kind == "s8" else "int16",
            quant={"output": out_quant.to_json()},
        )
        output_data = self.reference_golden(call)

        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_data)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate size
        input_size = int(np.prod(input_shape))
        
        # Select activation kernel function if needed
        activation_kernel_fn = None
        if has_activation:
            if activation_str == 'RELU':
                if kernel_info["output_c_type"] == "int8_t":
                    activation_kernel_fn = 'arm_relu_q7'
                else:  # int16_t
                    activation_kernel_fn = 'arm_relu_q15'
            elif activation_str == 'RELU6':
                if kernel_info["output_c_type"] == "int8_t":
                    activation_kernel_fn = 'arm_relu6_q7'
                else:  # int16_t
                    # ReLU6 for int16: use manual clamp
                    # Note: For S16, zero_point should be 0, so we quantize 6.0 as: 6.0 / scale
                    activation_kernel_fn = None
        
        # Build template context
        context = {
            'name': name,
            'input_size': input_size,
            'zero_point': int(output_zp),
            'scale': float(output_scale),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'has_activation': has_activation,
            'activation_kernel_fn': activation_kernel_fn,
            'activation_type': activation_str if has_activation else 'NONE',
            'comparison_tolerance': comparison_tolerance,
            'validation_helpers': ['tolerant_int'],
        }
        
        self.render_harness_case(
            output_dir, stem="quantize", context=context, pool=quantize_argument_pool(context),
            validation_key="QuantizationFunctions/quantize/quantize.c.j2", label="Quantize", operator="Quantize", sidecar=True,
        )
        
