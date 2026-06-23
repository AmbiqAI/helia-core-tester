"""
Softmax operation implementation for Helia-Core Tester.
"""

import math
from typing import Dict, Any, Tuple
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase  


class OpSoftmax(OperationBase):
    """
    Softmax operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Softmax operation."""
        input_shape = self.desc['input_shape']
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Softmax operation
        output = tf.keras.layers.Softmax()(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)
    
    def _select_cmsis_softmax_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Softmax operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.tensor_dtype("input")
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_softmax_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_softmax_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        elif activation_dtype == 'FP32':
            return {
                'kernel_fn': 'arm_softmax_f32',
                'input_c_type': 'float',
                'output_c_type': 'float'
            }
        else:
            raise NotImplementedError(f"Unsupported Softmax dtype: {activation_dtype}")

    def needs_keras_model(self) -> bool:
        if self.desc.get("hint", {}).get("force_cmsis", False):
            return False
        return True

    def allow_no_tflite(self) -> bool:
        if self.desc.get("hint", {}).get("force_cmsis", False):
            return True
        return False

    @staticmethod
    def _to_int32(value: int) -> int:
        value &= 0xFFFFFFFF
        if value & 0x80000000:
            return value - 0x100000000
        return value

    @staticmethod
    def _clz32(value: int) -> int:
        value &= 0xFFFFFFFF
        if value == 0:
            return 32
        return 32 - value.bit_length()

    @staticmethod
    def _doubling_high_mult(m1: int, m2: int) -> int:
        nn_q31_min = -0x80000000
        nn_q31_max = 0x7FFFFFFF
        mult = 1 << 30
        if (m1 < 0) ^ (m2 < 0):
            mult = 1 - mult
        mult = mult + (int(m1) * int(m2))
        result = int(mult // (1 << 31))
        if (m1 == m2) and (m1 == nn_q31_min):
            result = nn_q31_max
        return OpSoftmax._to_int32(result)

    @staticmethod
    def _divide_by_power_of_two(dividend: int, exponent: int) -> int:
        if exponent == 0:
            return OpSoftmax._to_int32(dividend)
        remainder_mask = (1 << exponent) - 1
        remainder = dividend & remainder_mask
        result = dividend >> exponent
        threshold = remainder_mask >> 1
        if result < 0:
            threshold += 1
        if remainder > threshold:
            result += 1
        return OpSoftmax._to_int32(result)

    @staticmethod
    def _mult_by_power_of_two(val: int, exp: int) -> int:
        nn_q31_min = -0x80000000
        nn_q31_max = 0x7FFFFFFF
        thresh = (1 << (31 - exp)) - 1
        result = int(val) << exp
        if val > thresh:
            result = nn_q31_max
        if val < -thresh:
            result = nn_q31_min
        return OpSoftmax._to_int32(result)

    @staticmethod
    def _exp_on_negative_values(val: int) -> int:
        nn_q31_max = 0x7FFFFFFF
        mask = 0
        shift = 24

        val_mod_minus_quarter = (val & ((1 << shift) - 1)) - (1 << shift)
        remainder = val_mod_minus_quarter - val
        x = (val_mod_minus_quarter << 5) + (1 << 28)
        x2 = OpSoftmax._doubling_high_mult(x, x)

        t1 = OpSoftmax._divide_by_power_of_two(OpSoftmax._doubling_high_mult(x2, x2), 2)
        t2 = OpSoftmax._doubling_high_mult(x2, x)
        t3 = t1 + t2
        t4 = OpSoftmax._doubling_high_mult(t3, 715827883)
        t5 = t4 + x2
        t6 = OpSoftmax._divide_by_power_of_two(t5, 1)
        t7 = x + t6
        result = 1895147668 + OpSoftmax._doubling_high_mult(1895147668, t7)

        def select_if_non_zero(const_val: int) -> Tuple[int, int]:
            nonlocal mask, shift, remainder, result
            mask = 1 if (remainder & (1 << shift)) != 0 else 0
            shift += 1
            if mask:
                result = OpSoftmax._doubling_high_mult(result, const_val)
            return mask, result

        select_if_non_zero(1672461947)
        select_if_non_zero(1302514674)
        select_if_non_zero(790015084)
        select_if_non_zero(290630308)
        select_if_non_zero(39332535)
        select_if_non_zero(720401)
        select_if_non_zero(242)

        if val == 0:
            return nn_q31_max
        return OpSoftmax._to_int32(result)

    @staticmethod
    def _one_over_one_plus_x(val: int) -> int:
        nn_q31_max = 0x7FFFFFFF
        sum_val = int(val) + nn_q31_max
        half_denominator = int((sum_val + (1 if sum_val >= 0 else -1)) // 2)
        x = 1515870810 + OpSoftmax._doubling_high_mult(half_denominator, -1010580540)

        shift = 1 << 29
        x = x + OpSoftmax._mult_by_power_of_two(
            OpSoftmax._doubling_high_mult(x, shift - OpSoftmax._doubling_high_mult(half_denominator, x)), 2)
        x = x + OpSoftmax._mult_by_power_of_two(
            OpSoftmax._doubling_high_mult(x, shift - OpSoftmax._doubling_high_mult(half_denominator, x)), 2)
        x = x + OpSoftmax._mult_by_power_of_two(
            OpSoftmax._doubling_high_mult(x, shift - OpSoftmax._doubling_high_mult(half_denominator, x)), 2)

        return OpSoftmax._mult_by_power_of_two(x, 1)

    @staticmethod
    def _softmax_common_s8(
        input_data: np.ndarray,
        num_rows: int,
        row_size: int,
        mult: int,
        shift: int,
        diff_min: int,
        int16_output: bool,
    ) -> np.ndarray:
        nn_q7_min, nn_q7_max = -128, 127
        nn_q15_min, nn_q15_max = -32768, 32767
        accum_bits = 12
        mask = 1 << shift

        output = np.zeros((num_rows, row_size), dtype=np.int16 if int16_output else np.int8)
        idx = 0
        for row_idx in range(num_rows):
            row = input_data[idx:idx + row_size]
            idx += row_size

            max_val = int(row[0])
            for col in range(1, row_size):
                max_val = max(max_val, int(row[col]))

            sum_val = 0
            for col in range(row_size):
                diff = int(row[col]) - max_val
                if diff >= diff_min:
                    exp_res = OpSoftmax._exp_on_negative_values(
                        OpSoftmax._doubling_high_mult(diff * mask, mult)
                    )
                    sum_val += OpSoftmax._divide_by_power_of_two(exp_res, accum_bits)

            headroom = OpSoftmax._clz32(sum_val)
            shifted = OpSoftmax._to_int32(sum_val << headroom) if sum_val > 0 else 0
            shifted_scale = OpSoftmax._one_over_one_plus_x(
                OpSoftmax._to_int32(shifted - (1 << 31))
            )

            if int16_output:
                bits_over_unit = accum_bits - headroom + 15
                for col in range(row_size):
                    diff = int(row[col]) - max_val
                    if diff >= diff_min:
                        exp_res = OpSoftmax._exp_on_negative_values(
                            OpSoftmax._doubling_high_mult(diff * mask, mult)
                        )
                        res = OpSoftmax._divide_by_power_of_two(
                            OpSoftmax._doubling_high_mult(shifted_scale, exp_res), bits_over_unit
                        ) + nn_q15_min
                        res = max(nn_q15_min, min(nn_q15_max, res))
                        output[row_idx, col] = np.int16(res)
                    else:
                        output[row_idx, col] = np.int16(nn_q15_min)
            else:
                bits_over_unit = accum_bits - headroom + 23
                for col in range(row_size):
                    diff = int(row[col]) - max_val
                    if diff >= diff_min:
                        exp_res = OpSoftmax._exp_on_negative_values(
                            OpSoftmax._doubling_high_mult(diff * mask, mult)
                        )
                        res = OpSoftmax._divide_by_power_of_two(
                            OpSoftmax._doubling_high_mult(shifted_scale, exp_res), bits_over_unit
                        ) + nn_q7_min
                        res = max(nn_q7_min, min(nn_q7_max, res))
                        output[row_idx, col] = np.int8(res)
                    else:
                        output[row_idx, col] = np.int8(nn_q7_min)

        return output
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Softmax operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
        
        name = self.desc['name']
        force_cmsis = self.desc.get("hint", {}).get("force_cmsis", False)
        tflite_path = output_dir / f"{name}.tflite"
        if not force_cmsis and not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_softmax_kernel()
        if force_cmsis and kernel_info["input_c_type"] != "int8_t":
            raise ValueError("CMSIS-only softmax currently supports int8 input only.")
        
        if force_cmsis:
            input_shape = tuple(self.desc["input_shape"])
            output_shape = input_shape
            input_scale = float(self.desc.get("hint", {}).get("input_scale", 1.0 / 128.0))
            input_zp = 0
            output_scale = input_scale
            output_zp = 0
        else:
            # Load LiteRT model for shape and quantization extraction
            from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
            model, subgraph = self.load_litert_model(str(tflite_path))
            op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
            
            # Extract shapes from LiteRT
            input_shape = op_tensors['inputs'][0]['shape']
            output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        if not force_cmsis and kernel_info["input_c_type"] != "float":
            # Extract quantization from LiteRT
            input_quant = op_tensors['inputs'][0]['quantization']
            output_quant = op_tensors['outputs'][0]['quantization']
            
            input_scale = input_quant.get('scale', 1.0)
            input_zp = input_quant.get('zero_point', 0)
            output_scale = output_quant.get('scale', 1.0)
            output_zp = output_quant.get('zero_point', 0)
        
        if kernel_info["input_c_type"] != "float":
            if isinstance(input_scale, (list, np.ndarray)):
                input_scale = float(input_scale[0])
            if isinstance(input_zp, (list, np.ndarray)):
                input_zp = int(input_zp[0])
            if isinstance(output_scale, (list, np.ndarray)):
                output_scale = float(output_scale[0])
            if isinstance(output_zp, (list, np.ndarray)):
                output_zp = int(output_zp[0])

            input_scale = float(input_scale)
            input_zp = int(input_zp)
            output_scale = float(output_scale)
            output_zp = int(output_zp)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Calculate multipliers and shifts for softmax
        # - For S8: use preprocess_softmax_scaling: beta * input_scale * (1 << 26)
        # - For S16: use input_scale_beta_rescale = beta * input_scale / (10.0 / 65535.0)
        softmax_input_integer_bits = 5  # scaled_diff_integer_bits, matches ns-cmsis-nn
        beta = 1.0  # softmax beta parameter (typically 1.0)
        
        if kernel_info["input_c_type"] == "float":
            mult = shift = diff_min = 0
        elif kernel_info["input_c_type"] == "int8_t":
            # S8: preprocess_softmax_scaling
            # input_beta_real_multiplier = min(beta * input_scale * (1 << (31 - scaled_diff_integer_bits)), max)
            max_real_multiplier = (1 << 31) - 1
            input_real_multiplier = min(beta * input_scale * (1 << (31 - softmax_input_integer_bits)), max_real_multiplier)
        else:
            # S16: input_scale_beta_rescale
            # input_scale_beta_rescale = beta * input_scale / (10.0 / 65535.0)
            input_scale_beta_rescale = beta * input_scale / (10.0 / 65535.0)
            input_real_multiplier = input_scale_beta_rescale
        
        mult, shift = calculate_multiplier_shift(input_real_multiplier)
        
        # Calculate diff_min for s8 softmax
        # diff_min = -1.0 * calculate_input_radius(input_integer_bits, input_left_shift, total_signed_bits=31)
        # where calculate_input_radius = floor(max_val * (1 << (31 - input_integer_bits)) / (1 << input_left_shift))
        # Note: input_left_shift can be negative, so we handle division properly
        if kernel_info["input_c_type"] == "int8_t":
            # calculate_input_radius equivalent
            max_val = (1 << softmax_input_integer_bits) - 1
            if shift >= 0:
                max_input_rescaled = max_val * (1 << (31 - softmax_input_integer_bits)) / (1 << shift)
            else:
                # When shift is negative, (1 << shift) would be fractional, so we multiply instead
                max_input_rescaled = max_val * (1 << (31 - softmax_input_integer_bits)) * (1 << (-shift))
            diff_min = -int(math.floor(max_input_rescaled))
        else:
            diff_min = 0  # Not used for s16
        # Calculate num_rows and row_size
        # Softmax operates on the last dimension (row_size)
        # num_rows is the product of all dimensions except the last
        if len(input_shape) >= 2:
            num_rows = int(np.prod(input_shape[:-1]))
            row_size = int(input_shape[-1])
        else:
            # 1D case
            num_rows = 1
            row_size = int(input_shape[0])
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        if kernel_info["input_c_type"] == "float":
            input_q = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        elif kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
            input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
            input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")

        self.rng.__setstate__(rng_state)

        if kernel_info["input_c_type"] == "float":
            interpreter = self.load_litert_interpreter(str(tflite_path))
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], input_q)
            interpreter.invoke()
            output_data = np.array(interpreter.get_tensor(output_details[0]['index']), dtype=np.float32)
        elif force_cmsis:
            hint = self.desc.get("hint", {})
            if isinstance(hint, dict) and "diff_min" in hint:
                diff_min = int(hint["diff_min"])
            hint = self.desc.get("hint", {})
            extras = hint.get("extras", {}) if isinstance(hint, dict) else {}
            output_dtype_hint = str(hint.get("output_dtype", extras.get("output_dtype", ""))).upper()
            int16_output = output_dtype_hint == "S16"
            output_data = self._softmax_common_s8(
                input_q.flatten().astype(np.int8),
                num_rows,
                row_size,
                int(mult),
                int(shift),
                int(diff_min),
                int16_output,
            )
        else:
            # Run inference using LiteRT interpreter
            interpreter = self.load_litert_interpreter(str(tflite_path))
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            interpreter.set_tensor(input_details[0]['index'], input_q)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = np.array(output_data)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        hint = self.desc.get("hint", {})
        extras = hint.get("extras", {}) if isinstance(hint, dict) else {}
        output_dtype_hint = str(hint.get("output_dtype", extras.get("output_dtype", ""))).upper()
        is_s8_s16 = force_cmsis and output_dtype_hint == "S16" and kernel_info["input_c_type"] == "int8_t"
        if kernel_info["input_c_type"] == "float":
            kernel_fn = kernel_info["kernel_fn"]
            output_c_type = kernel_info["output_c_type"]
            returns_status = True
            uses_lut = False
            float_kernel = True
        elif is_s8_s16:
            kernel_fn = "arm_softmax_s8_s16"
            output_c_type = "int16_t"
            returns_status = False
            uses_lut = False
            float_kernel = False
        else:
            kernel_fn = kernel_info["kernel_fn"]
            output_c_type = kernel_info["output_c_type"]
            returns_status = kernel_info["input_c_type"] == "int16_t"
            uses_lut = kernel_info["input_c_type"] == "int16_t"
            float_kernel = False

        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'num_rows': num_rows,
            'row_size': row_size,
            'mult': int(mult),
            'shift': int(shift),
            'diff_min': int(diff_min),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': output_c_type,
            'kernel_fn': kernel_fn,
            'returns_status': returns_status,
            'uses_lut': uses_lut,
            'float_kernel': float_kernel,
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("SoftmaxFunctions/softmax/softmax.h.j2", context)
        h_path = includes_api_dir / f"{name}_softmax.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("SoftmaxFunctions/softmax/softmax.c.j2", context)
        c_path = output_dir / f"{name}_softmax.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Softmax'),
            'operator_name': 'softmax'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
