"""
Multiply (elementwise) operation implementation for Helia-Core Tester.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.binary_basic_math_base import BinaryBasicMathBase 


class OpMul(BinaryBasicMathBase):
    """
    Mul operation.
    """

    SIGN_SPAN_OPERANDS = ("input_1", "input_2")
    # Keeps about a fifth saturated.
    S8_REACH = 48
    
    def uses_reference(self) -> bool:
        return True

    def _select_cmsis_mul_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Mul operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        entry_kernel = self._direct_entry_kernel()
        if entry_kernel:
            return entry_kernel
        activation_dtype = self.tensor_dtype("input")
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_mul_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'float_kernel': False,
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_mul_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t',
                'float_kernel': False,
            }
        elif activation_dtype in ('FP32', 'FP16'):
            # Opt-in only: un-hinted mismatched shapes keep the materialised-broadcast
            # flat call that mul_float_{channel,scalar}_broadcast_* already pin.
            float_broadcast = self._float_broadcast_call(auto_on_shape_mismatch=False)
            suffix = 'f32' if activation_dtype == 'FP32' else 'f16'
            c_type = 'float' if activation_dtype == 'FP32' else 'float16_t'
            return {
                'kernel_fn': f"arm_elementwise_mul_broadcast_{suffix}" if float_broadcast else f"arm_elementwise_mul_{suffix}",
                'input_c_type': c_type,
                'output_c_type': c_type,
                'float_kernel': True,
                'float_broadcast': float_broadcast,
            }
        else:
            raise NotImplementedError(f"Unsupported Mul dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Mul operation.
        """
        from helia_core_tester.generation.reference import quant as ref_quant
        from helia_core_tester.generation.reference.abi import activation_code
        from helia_core_tester.generation.reference.bindings import get_bindings, output_shape_for
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_info = self._select_cmsis_mul_kernel()
        input1_shape = tuple(int(d) for d in self.desc["input_1_shape"])
        input2_shape = tuple(int(d) for d in self.desc["input_2_shape"])
        output_shape = output_shape_for("mul", input1_shape, input2_shape)

        builder = TemplateContextBuilder()
        input1_dims = builder.nhwc_to_cmsis_dims(input1_shape)
        input2_dims = builder.nhwc_to_cmsis_dims(input2_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        activation_dtype = self.tensor_dtype("input")
        if kernel_info["float_kernel"]:
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            # Draw both operands from one RNG stream: reseeding per call would
            # make input1 == input2 and the golden collapse to input1**2.
            input1_f32, input2_f32 = self._sample_dual_uniform_inputs(input1_shape, input2_shape)
            input1_q = input1_f32.astype(float_dtype)
            input2_q = input2_f32.astype(float_dtype)
            activation_min = float(self.desc.get("act_min", -1.0e30))
            activation_max = float(self.desc.get("act_max", 1.0e30))
            output_data = self.reference_golden(ReferenceCall(
                f"mul_{ref_quant.kind(activation_dtype)}",
                {"activation_min": activation_min, "activation_max": activation_max},
                {"input1": input1_q, "input2": input2_q},
                {"output": output_shape},
            ))
            if not kernel_info.get("float_broadcast"):
                # The flat kernel sees the broadcast already materialised.
                input1_q = np.broadcast_to(input1_q, output_shape).astype(float_dtype, copy=True)
                input2_q = np.broadcast_to(input2_q, output_shape).astype(float_dtype, copy=True)
            activation_min_literal = builder.format_float_literal(activation_min)
            activation_max_literal = builder.format_float_literal(activation_max)
            input1_zp = input2_zp = output_zp = output_mult = output_shift = 0
        else:
            scale, zero_point = ref_quant.preset_quant(activation_dtype)
            input1_scale = input2_scale = scale
            input1_zp = input2_zp = output_zp = zero_point
            params = get_bindings().prepare("mul_prepare", {
                "dtype": ref_quant.hct_dtype(activation_dtype), "activation": activation_code("NONE"),
                "input1_scale": scale, "input1_zero_point": zero_point,
                "input2_scale": scale, "input2_zero_point": zero_point,
                "output_scale": scale, "output_zero_point": zero_point,
            })
            output_mult, output_shift = params["output_multiplier"], params["output_shift"]
            activation_min, activation_max = params["activation_min"], params["activation_max"]
            input1_data, input2_data = self._sample_dual_uniform_inputs(input1_shape, input2_shape)
            input1_data = self._widen_s8(input1_data, input1_scale, kernel_info["input_c_type"])
            input2_data = self._widen_s8(input2_data, input2_scale, kernel_info["input_c_type"])
            qmin, qmax = (-32768, 32767) if activation_dtype == "S16" else (-128, 127)
            np_in_dtype = np.int16 if activation_dtype == "S16" else np.int8
            input1_q = np.round(input1_data / float(input1_scale) + float(input1_zp)).astype(np.int32)
            input1_q = np.clip(input1_q, qmin, qmax).astype(np_in_dtype)

            input2_q = np.round(input2_data / float(input2_scale) + float(input2_zp)).astype(np.int32)
            input2_q = np.clip(input2_q, qmin, qmax).astype(np_in_dtype)
            input1_q, input2_q = self._enforce_int_operand_sign_span(
                (("input_1", input1_q, input1_zp), ("input_2", input2_q, input2_zp)),
                steerable=("input_1", "input_2"),
            )

            # The reference requantizes with double rounding (SRDHM, then a rounding divide
            # by a power of two, halves away from zero), as CMSIS-NN's arm_nn_requantize does;
            # the converter-era golden disagreed with both at exact ties (the
            # mul_default_s8_hw_generated 19/160 mismatch investigation).
            output_data = self.reference_golden(ReferenceCall(
                f"mul_{ref_quant.kind(activation_dtype)}",
                params,
                {"input1": np.ascontiguousarray(input1_q), "input2": np.ascontiguousarray(input2_q)},
                {"output": output_shape},
                quant={role: {"scale": scale, "zero_point": zero_point} for role in ("input1", "input2", "output")},
            ))

        # Format arrays
        input1_array_str = builder.format_array_as_c_literal(input1_q)
        input2_array_str = builder.format_array_as_c_literal(input2_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate block size (total number of elements)
        block_size = int(np.prod(output_shape))
        
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
            # For CMSIS-NN, input offsets should be negated (like other operations)
            # Output offset is used as-is (not negated)
            'input1_offset': -int(input1_zp),
            'input2_offset': -int(input2_zp),
            'out_offset': int(output_zp),
            'out_mult': int(output_mult),
            'out_shift': int(output_shift),
            'out_activation_min': out_activation_min_ctx,
            'out_activation_max': out_activation_max_ctx,
            'block_size': int(block_size),
            'input1_data_array': input1_array_str,
            'input2_data_array': input2_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'expected_status': self.expected_status(),
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
            'operator': self.desc.get('operator', 'Mul'),
            'operator_name': 'mul',
        }
        self._write_op_outputs(output_dir, "mul", "BasicMathFunctions/mul/mul.h.j2", "BasicMathFunctions/mul/mul.c.j2", context, cmake_context)



from helia_core_tester.generation.harness.registry import harness_pool  # noqa: E402
from helia_core_tester.generation.harness.simple import binary_case_pool  # noqa: E402

harness_pool("BasicMathFunctions/mul/mul.c.j2", label="Mul")(binary_case_pool)
