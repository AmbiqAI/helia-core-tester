"""
Quantize operation implementation for Helia-Core Tester.

The golden is TFLite's AffineQuantize on the C reference, after the fused activation the harness
applies in float.
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
        
        from helia_core_tester.generation.reference import quant as ref_quant
        from helia_core_tester.generation.reference.call import ReferenceCall

        name = self.desc['name']
        kernel_info = self._select_cmsis_quantize_kernel()
        builder = TemplateContextBuilder()

        activation_str = self.desc.get('activation', 'NONE')
        has_activation = activation_str in ['RELU', 'RELU6']
        comparison_tolerance = 1
        act_min, act_max = {"RELU": (0.0, float("inf")), "RELU6": (0.0, 6.0)}.get(
            activation_str, (float("-inf"), float("inf")))

        input_shape = output_shape = tuple(int(d) for d in self.desc["input_shape"])
        out_kind = ref_quant.kind(self.tensor_dtype("output"))
        # The [-1, 1] draw after the activation: the range the converter used to calibrate.
        out_quant = self.activation_quant("output", (max(-1.0, act_min), min(1.0, act_max)), out_kind)
        output_scale, output_zp = out_quant.scale, out_quant.zero_point
        input_data = self._sample_uniform(input_shape)
        output_data = self.reference_golden(ReferenceCall(
            f"quantize_f32_{out_kind}",
            {"scale": output_scale, "zero_point": output_zp, "activation_min": act_min, "activation_max": act_max},
            {"input": np.ascontiguousarray(input_data, dtype=np.float32)}, {"output": output_shape},
            quant={"output": out_quant.to_json(), "activation": activation_str},
        ))

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
        
