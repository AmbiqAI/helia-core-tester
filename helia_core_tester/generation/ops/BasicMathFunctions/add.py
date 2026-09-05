"""
Add operation implementation.
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.binary_basic_math_base import BinaryBasicMathBase
from helia_core_tester.generation.utils.litert_builder import build_binary_broadcast_op


def build_add_op(
    *,
    input_1_shape,
    input_2_shape,
    dtype: str = "int8",
) -> bytes:
    from ai_edge_litert import schema_py_generated as litert

    options = litert.AddOptionsT()
    options.fusedActivationFunction = litert.ActivationFunctionType.NONE
    return build_binary_broadcast_op(
        op_name="ADD",
        input_1_shape=input_1_shape,
        input_2_shape=input_2_shape,
        dtype=dtype,
        options=options,
        options_type=litert.BuiltinOptions.AddOptions,
    )


class OpAdd(BinaryBasicMathBase):
    """
    Add operation.
    """
    
    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Add uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        activation_dtype = self.tensor_dtype("input")
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        elif activation_dtype == "FP32":
            dtype = "float32"
        elif activation_dtype == "FP16":
            dtype = "float16"
        else:
            raise NotImplementedError(f"Unsupported Add dtype: {activation_dtype}")

        input_1_shape = tuple(self.desc["input_1_shape"])
        input_2_shape = tuple(self.desc["input_2_shape"])

        model_bytes = build_add_op(
            input_1_shape=input_1_shape,
            input_2_shape=input_2_shape,
            dtype=dtype,
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def _select_cmsis_add_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Add operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
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
        from helia_core_tester.generation.utils.tflite_utils import (
            scalar_scale_zp,
            activation_bounds,
            elementwise_addsub_quant_params,
        )
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_add_kernel()
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        input1_shape = self._ensure_shape_tuple(op_tensors['inputs'][0]['shape'])
        input2_shape = self._ensure_shape_tuple(
            op_tensors['inputs'][1]['shape'] if len(op_tensors['inputs']) > 1 else op_tensors['inputs'][0]['shape']
        )
        output_shape = self._ensure_shape_tuple(op_tensors['outputs'][0]['shape'])
        
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
            input1_quant = op_tensors['inputs'][0]['quantization']
            input2_quant = op_tensors['inputs'][1]['quantization'] if len(op_tensors['inputs']) > 1 else input1_quant
            output_quant = op_tensors['outputs'][0]['quantization']

            input1_scale, input1_zp = scalar_scale_zp(input1_quant)
            input2_scale, input2_zp = scalar_scale_zp(input2_quant)
            output_scale, output_zp = scalar_scale_zp(output_quant)

            activation_min, activation_max = activation_bounds(activation_dtype)
            addsub_qparams = elementwise_addsub_quant_params(
                input1_scale=float(input1_scale),
                input2_scale=float(input2_scale),
                output_scale=float(output_scale),
                activation_dtype=activation_dtype,
            )
            mult1 = addsub_qparams["input1_mult"]
            shift1 = addsub_qparams["input1_shift"]
            mult2 = addsub_qparams["input2_mult"]
            shift2 = addsub_qparams["input2_shift"]
            output_mult = addsub_qparams["out_mult"]
            output_shift = addsub_qparams["out_shift"]
            left_shift = addsub_qparams["left_shift"]

            # Draw both operands from one RNG stream: reseeding per call would
            # make input1 == input2 and weaken/vacuously pass the golden.
            input1_data, input2_data = self._sample_dual_uniform_inputs(input1_shape, input2_shape)
            qmin, qmax = activation_bounds(activation_dtype)
            np_in_dtype = np.int16 if activation_dtype == "S16" else np.int8
            input1_q = np.round(input1_data / float(input1_scale) + float(input1_zp)).astype(np.int32)
            input1_q = np.clip(input1_q, qmin, qmax).astype(np_in_dtype)

            input2_q = np.round(input2_data / float(input2_scale) + float(input2_zp)).astype(np.int32)
            input2_q = np.clip(input2_q, qmin, qmax).astype(np_in_dtype)

            if input1_shape == input2_shape:
                interpreter = self.load_litert_interpreter(str(tflite_path))
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], input1_q)
                interpreter.set_tensor(input_details[1]['index'], input2_q)
                interpreter.invoke()
                output_data = np.array(interpreter.get_tensor(output_details[0]['index']))
            else:
                output_data = self._simulate_add_quantized(
                    input1_q,
                    input2_q,
                    input1_offset=-int(input1_zp),
                    input2_offset=-int(input2_zp),
                    input1_mult=int(mult1),
                    input1_shift=int(shift1),
                    input2_mult=int(mult2),
                    input2_shift=int(shift2),
                    left_shift=int(left_shift),
                    out_offset=int(output_zp),
                    out_mult=int(output_mult),
                    out_shift=int(output_shift),
                    out_activation_min=int(activation_min),
                    out_activation_max=int(activation_max),
                    out_dtype=np_in_dtype,
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

    @classmethod
    def _simulate_add_quantized(
        cls,
        input1_q: np.ndarray,
        input2_q: np.ndarray,
        *,
        input1_offset: int,
        input2_offset: int,
        input1_mult: int,
        input1_shift: int,
        input2_mult: int,
        input2_shift: int,
        left_shift: int,
        out_offset: int,
        out_mult: int,
        out_shift: int,
        out_activation_min: int,
        out_activation_max: int,
        out_dtype: np.dtype,
    ) -> np.ndarray:
        a = (input1_q.astype(np.int32) + int(input1_offset)) << int(left_shift)
        b = (input2_q.astype(np.int32) + int(input2_offset)) << int(left_shift)
        a = cls._requantize_np(a, int(input1_mult), int(input1_shift))
        b = cls._requantize_np(b, int(input2_mult), int(input2_shift))
        s = a + b
        s = cls._requantize_np(s, int(out_mult), int(out_shift))
        s = s + int(out_offset)
        s = np.clip(s, int(out_activation_min), int(out_activation_max))
        return s.astype(out_dtype)
