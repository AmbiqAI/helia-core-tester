"""
Add operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.binary_basic_math_base import BinaryBasicMathBase


class OpAdd(BinaryBasicMathBase):
    """
    Add operation.
    """

    SIGN_SPAN_OPERANDS = ("input_1", "input_2")
    
    def _select_cmsis_add_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Add operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        entry_kernel = self._direct_entry_kernel()
        if entry_kernel:
            return entry_kernel
        activation_dtype = self.tensor_dtype("input")
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_add_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'float_kernel': False,
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_add_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t',
                'float_kernel': False,
            }
        elif activation_dtype in ('FP32', 'FP16'):
            hint = self.desc.get("hint", {})
            legacy_fp16 = activation_dtype == 'FP16' and str(hint.get("kernel_variant", "")).lower() == "legacy_fp16"
            # Opt-in only: un-hinted mismatched shapes keep the materialised-broadcast
            # flat call that add_float_{channel,scalar}_broadcast_* already pin.
            float_broadcast = self._float_broadcast_call(auto_on_shape_mismatch=False)
            if legacy_fp16 and float_broadcast:
                raise ValueError("hint.kernel_variant legacy_fp16 has no broadcast entry point")
            if legacy_fp16:
                return {
                    'kernel_fn': 'arm_elementwise_add_fp16',
                    'input_c_type': 'float16_t',
                    'output_c_type': 'float16_t',
                    'float_kernel': True,
                    'legacy_fp16_kernel': True,
                }
            suffix = 'f32' if activation_dtype == 'FP32' else 'f16'
            c_type = 'float' if activation_dtype == 'FP32' else 'float16_t'
            return {
                'kernel_fn': f"arm_elementwise_add_broadcast_{suffix}" if float_broadcast else f"arm_elementwise_add_{suffix}",
                'input_c_type': c_type,
                'output_c_type': c_type,
                'float_kernel': True,
                'legacy_fp16_kernel': False,
                'float_broadcast': float_broadcast,
            }
        else:
            raise NotImplementedError(f"Unsupported Add dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Add operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import activation_bounds
        
        name = self.desc['name']
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_add_kernel()
        input1_shape, input2_shape, output_shape = self._binary_shapes()

        builder = TemplateContextBuilder()
        input1_dims = builder.nhwc_to_cmsis_dims(input1_shape)
        input2_dims = builder.nhwc_to_cmsis_dims(input2_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        activation_dtype = self.tensor_dtype("input")

        if kernel_info["float_kernel"]:
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            # Draw both operands from one RNG stream: reseeding per call would
            # make input1 == input2 and the golden collapse to 2*input1.
            input1_f32, input2_f32 = self._sample_dual_uniform_inputs(input1_shape, input2_shape)
            input1_q = input1_f32.astype(float_dtype)
            input2_q = input2_f32.astype(float_dtype)
            activation_min = float(self.desc.get("act_min", -1.0e30))
            activation_max = float(self.desc.get("act_max", 1.0e30))
            output_data = np.clip(
                input1_q.astype(np.float32) + input2_q.astype(np.float32),
                activation_min,
                activation_max,
            ).astype(float_dtype)
            if not kernel_info.get("float_broadcast"):
                # The flat kernel sees the broadcast already materialised.
                input1_q = np.broadcast_to(input1_q, output_shape).astype(float_dtype, copy=True)
                input2_q = np.broadcast_to(input2_q, output_shape).astype(float_dtype, copy=True)
            activation_min_literal = builder.format_float_literal(activation_min)
            activation_max_literal = builder.format_float_literal(activation_max)
            mult1 = shift1 = mult2 = shift2 = output_mult = output_shift = left_shift = 0
            input1_zp = input2_zp = output_zp = 0
        else:
            from helia_core_tester.generation.reference import params as ref_params

            scale, zero_point = self._fixed_quant()
            input1_scale = input2_scale = output_scale = scale
            input1_zp = input2_zp = output_zp = zero_point

            activation_min, activation_max = activation_bounds(activation_dtype)
            p = ref_params.addsub_params(
                activation_dtype, input1_scale, input1_zp, input2_scale, input2_zp, output_scale, output_zp
            )
            mult1, shift1 = p.input1_multiplier, p.input1_shift
            mult2, shift2 = p.input2_multiplier, p.input2_shift
            output_mult, output_shift = p.output_multiplier, p.output_shift
            left_shift = p.left_shift

            # Draw both operands from one RNG stream: reseeding per call would
            # make input1 == input2 and weaken/vacuously pass the golden.
            input1_data, input2_data = self._sample_dual_uniform_inputs(input1_shape, input2_shape)
            input1_data = self._widen_s8(input1_data, input1_scale, kernel_info["input_c_type"])
            input2_data = self._widen_s8(input2_data, input2_scale, kernel_info["input_c_type"])
            qmin, qmax = activation_bounds(activation_dtype)
            np_in_dtype = np.int16 if activation_dtype == "S16" else np.int8
            input1_q = np.round(input1_data / float(input1_scale) + float(input1_zp)).astype(np.int32)
            input1_q = np.clip(input1_q, qmin, qmax).astype(np_in_dtype)

            input2_q = np.round(input2_data / float(input2_scale) + float(input2_zp)).astype(np.int32)
            input2_q = np.clip(input2_q, qmin, qmax).astype(np_in_dtype)
            input1_q, input2_q = self._enforce_int_operand_sign_span(
                (("input_1", input1_q, input1_zp), ("input_2", input2_q, input2_zp)),
                steerable=("input_1", "input_2"),
            )
            output_data = self._reference_binary(
                "add", input1_q, input2_q, output_shape, p.__dict__, activation_min, activation_max,
                quant={"input1": {"scale": input1_scale, "zero_point": input1_zp},
                       "input2": {"scale": input2_scale, "zero_point": input2_zp},
                       "output": {"scale": output_scale, "zero_point": output_zp}},
            )

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
            'block_size': int(block_size),
            'input1_data_array': input1_array_str,
            'input2_data_array': input2_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'expected_status': self.expected_status(),
            'float_kernel': kernel_info["float_kernel"],
            'legacy_fp16_kernel': kernel_info.get("legacy_fp16_kernel", False),
        }
        if kernel_info["float_kernel"]:
            context["out_activation_min_literal"] = activation_min_literal
            context["out_activation_max_literal"] = activation_max_literal
            context["validation_mode"] = "float"
            if kernel_info.get("float_broadcast"):
                context["float_broadcast"] = True
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Add'),
            'operator_name': 'add',
        }
        self._write_op_outputs(output_dir, "add", "BasicMathFunctions/add/add.h.j2", "BasicMathFunctions/add/add.c.j2", context, cmake_context)


from helia_core_tester.generation.harness.registry import harness_pool  # noqa: E402
from helia_core_tester.generation.harness.simple import binary_case_pool  # noqa: E402

harness_pool("BasicMathFunctions/add/add.c.j2", label="Add")(binary_case_pool)
