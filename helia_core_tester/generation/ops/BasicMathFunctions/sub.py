"""
Subtract operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.binary_basic_math_base import BinaryBasicMathBase


class OpSub(BinaryBasicMathBase):
    """
    Subtract operation.
    """

    SIGN_SPAN_OPERANDS = ("input_1", "input_2")
    
    def _select_cmsis_sub_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Sub operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.tensor_dtype("input", default="S8")

        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_sub_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'float_kernel': False,
            }
        elif activation_dtype == 'S16':
            call_style = self.desc.get("hint", {}).get("call_style", "")
            if str(call_style).lower() == "elementwise":
                return {
                    'kernel_fn': 'arm_elementwise_sub_s16',
                    'input_c_type': 'int16_t',
                    'output_c_type': 'int16_t',
                    'float_kernel': False,
                }
            return {
                'kernel_fn': 'arm_sub_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t',
                'float_kernel': False,
            }
        elif activation_dtype in ('FP32', 'FP16'):
            # The flat float kernel has no dims, so two shapes can only reach the
            # dims-taking broadcast entry point (ns-cmsis-nn#415).
            float_broadcast = self._float_broadcast_call(auto_on_shape_mismatch=True)
            suffix = 'f32' if activation_dtype == 'FP32' else 'f16'
            c_type = 'float' if activation_dtype == 'FP32' else 'float16_t'
            return {
                'kernel_fn': f"arm_elementwise_sub_broadcast_{suffix}" if float_broadcast else f"arm_elementwise_sub_{suffix}",
                'input_c_type': c_type,
                'output_c_type': c_type,
                'float_kernel': True,
                'float_broadcast': float_broadcast,
            }
        else:
            raise NotImplementedError(f"Unsupported Sub dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Sub operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import activation_bounds
        
        name = self.desc['name']
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_sub_kernel()
        input1_shape, input2_shape, output_shape = self._binary_shapes()

        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input1_dims = builder.nhwc_to_cmsis_dims(input1_shape)
        input2_dims = builder.nhwc_to_cmsis_dims(input2_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        activation_dtype = self.tensor_dtype("input", default="S8")

        if kernel_info["float_kernel"]:
            # The flat kernels take equal shapes; the broadcast entry point takes the
            # operands as drawn and the golden broadcasts them by NumPy's rules.
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            activation_min = float(self.desc.get("act_min", -1.0e30))
            activation_max = float(self.desc.get("act_max", 1.0e30))
            # Draw both operands from one RNG stream: reseeding per call would
            # make input1 == input2 and the golden identically zero.
            input1_f32, input2_f32 = self._sample_dual_uniform_inputs(input1_shape, input2_shape)
            input1_q = input1_f32.astype(float_dtype)
            input2_q = input2_f32.astype(float_dtype)
            output_data = np.clip(
                input1_q.astype(np.float32) - input2_q.astype(np.float32),
                activation_min,
                activation_max,
            ).astype(float_dtype)
            activation_min_literal = builder.format_float_literal(activation_min)
            activation_max_literal = builder.format_float_literal(activation_max)
            mult1 = shift1 = mult2 = shift2 = output_mult = output_shift = left_shift = 0
            input1_zp = input2_zp = output_zp = 0
        else:
            from helia_core_tester.generation.reference import params as ref_params

            input1_scale, input1_zp = input2_scale, input2_zp = output_scale, output_zp = self._fixed_quant()
            activation_min, activation_max = activation_bounds(activation_dtype)
            p = ref_params.sub_params(
                activation_dtype, input1_scale, input1_zp, input2_scale, input2_zp, output_scale, output_zp
            )
            mult1, shift1 = p.input1_multiplier, p.input1_shift
            mult2, shift2 = p.input2_multiplier, p.input2_shift
            output_mult, output_shift = p.output_multiplier, p.output_shift
            left_shift = p.left_shift

            # Generate input data and quantize
            rng_state = self.rng.__getstate__()
            self.rng = np.random.default_rng(self.seed)

            input1_data = self.rng.uniform(-1.0, 1.0, size=input1_shape).astype(np.float32)
            input2_data = self.rng.uniform(-1.0, 1.0, size=input2_shape).astype(np.float32)

            self.rng.__setstate__(rng_state)
            input1_data = self._widen_s8(input1_data, input1_scale, kernel_info["input_c_type"])
            input2_data = self._widen_s8(input2_data, input2_scale, kernel_info["input_c_type"])

            # Quantize inputs
            if kernel_info["input_c_type"] == "int8_t":
                np_in_dtype = np.int8
                qmin, qmax = -128, 127
            elif kernel_info["input_c_type"] == "int16_t":
                np_in_dtype = np.int16
                qmin, qmax = -32768, 32767
            else:
                raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")

            input1_q = np.round(input1_data / float(input1_scale) + float(input1_zp)).astype(np.int32)
            input1_q = np.clip(input1_q, qmin, qmax).astype(np_in_dtype)

            input2_q = np.round(input2_data / float(input2_scale) + float(input2_zp)).astype(np.int32)
            input2_q = np.clip(input2_q, qmin, qmax).astype(np_in_dtype)
            input1_q, input2_q = self._enforce_int_operand_sign_span(
                (("input_1", input1_q, input1_zp), ("input_2", input2_q, input2_zp)),
                steerable=("input_1", "input_2"),
            )

            output_data = self._reference_binary(
                "sub", input1_q, input2_q, output_shape, p.__dict__, activation_min, activation_max,
                quant={"input1": {"scale": input1_scale, "zero_point": input1_zp},
                       "input2": {"scale": input2_scale, "zero_point": input2_zp},
                       "output": {"scale": output_scale, "zero_point": output_zp}},
            )

        # Format arrays
        input1_array_str = builder.format_array_as_c_literal(input1_q)
        input2_array_str = builder.format_array_as_c_literal(input2_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        # Only the quantized path renders these as integers; the float path passes the
        # bounds as literals instead, and the +/-INFINITY no-clamp idiom has no integer
        # image at all, so casting it would raise rather than produce a dead field.
        if kernel_info["float_kernel"]:
            out_activation_min_ctx = activation_min
            out_activation_max_ctx = activation_max
        else:
            out_activation_min_ctx = int(activation_min)
            out_activation_max_ctx = int(activation_max)

        context = {
            'name': name,
            'input1_dims': input1_dims,
            'input2_dims': input2_dims,
            'output_dims': output_dims,
            'input1_offset': -int(input1_zp),
            'input1_mult': int(mult1),
            'input1_shift': int(shift1),
            'input2_offset': -int(input2_zp),
            'input2_mult': int(mult2),
            'input2_shift': int(shift2),
            'left_shift': int(left_shift),
            'out_offset': int(output_zp),
            'out_mult': int(output_mult),
            'out_shift': int(output_shift),
            'out_activation_min': out_activation_min_ctx,
            'out_activation_max': out_activation_max_ctx,
            'block_size': int(np.prod(output_shape)),
            'call_style': str(self.desc.get("hint", {}).get("call_style", "")),
            'input1_data_array': input1_array_str,
            'input2_data_array': input2_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'float_kernel': kernel_info["float_kernel"],
        }
        if kernel_info["float_kernel"]:
            context["out_activation_min_literal"] = activation_min_literal
            context["out_activation_max_literal"] = activation_max_literal
            context["validation_mode"] = "float"
            if kernel_info.get("float_broadcast"):
                context["float_broadcast"] = True

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Sub'),
            'operator_name': 'sub'
        }
        self._write_op_outputs(output_dir, "sub", "BasicMathFunctions/sub/sub.h.j2", "BasicMathFunctions/sub/sub.c.j2", context, cmake_context)


from helia_core_tester.generation.harness.registry import harness_pool  # noqa: E402
from helia_core_tester.generation.harness.simple import binary_case_pool  # noqa: E402

harness_pool("BasicMathFunctions/sub/sub.c.j2", label="Sub")(binary_case_pool)
