"""
HardSwish operation implementation.
"""

from typing import Dict, Any, Tuple
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpHardSwish(OperationBase):
    """
    HardSwish operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for HardSwish operation."""
        input_shape = self.desc['input_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # HardSwish operation
        output = tf.keras.layers.Activation('hard_swish')(inputs)
            
        model = tf.keras.Model(inputs=[inputs], outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if str(activation_dtype).upper() == "S16":
            # TFLite quantization for HARD_SWISH int16 is not supported.
            # Skip TFLite generation and rely on descriptor-provided scales.
            return
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply quantization based on activation_dtype
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif activation_dtype == 'S16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16

        
        # Generate representative dataset
        def representative_data_gen():
            rng = np.random.default_rng(rep_seed)
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = rng.uniform(-8.0, 8.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = rng.uniform(-8.0, 8.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = rng.uniform(-8.0, 8.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def _select_cmsis_hard_swish_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for HardSwish operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        variant = self._variant()
        
        if activation_dtype == 'S8':
            if variant == "compat":
                return {
                    'kernel_fn': 'arm_hard_swish_compat_s8',
                    'input_c_type': 'int8_t',
                    'output_c_type': 'int8_t'
                }
            return {
                'kernel_fn': 'arm_hard_swish_precise_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_hard_swish_precise_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported HardSwish dtype: {activation_dtype}")

    @staticmethod
    def _clamp(val: int, vmin: int, vmax: int) -> int:
        return max(vmin, min(vmax, val))

    @classmethod
    def _doubling_high_mult_no_sat(cls, m1: int, m2: int) -> int:
        # matches arm_nn_doubling_high_mult_no_sat
        mult = (1 << 30) + (m1 * m2)
        return int(mult >> 31)

    @classmethod
    def _divide_by_power_of_two(cls, dividend: int, exponent: int) -> int:
        if exponent == 0:
            return int(dividend)
        remainder_mask = (1 << exponent) - 1
        remainder = remainder_mask & dividend
        result = dividend >> exponent
        threshold = remainder_mask >> 1
        if result < 0:
            threshold += 1
        if remainder > threshold:
            result += 1
        return int(result)

    @classmethod
    def _nonneg_divide_by_pot_s32(cls, dividend: int, exponent: int) -> int:
        if exponent == 0:
            return int(dividend)
        t = int(dividend) + (1 << (exponent - 1))
        return int(t >> exponent)

    @classmethod
    def _requantize(cls, val: int, multiplier: int, shift: int) -> int:
        left_shift = shift if shift > 0 else 0
        right_shift = -shift if shift < 0 else 0
        prod = val * (1 << left_shift)
        return cls._divide_by_power_of_two(cls._doubling_high_mult_no_sat(prod, multiplier), right_shift)

    @classmethod
    def _sat_lshift_s16(cls, x: int, shift: int) -> int:
        if shift <= 0:
            return int(x)
        v = int(x) << shift
        return cls._clamp(v, -32768, 32767)

    @classmethod
    def _sqrdmulh_s16(cls, a: int, b: int) -> int:
        if a == -32768 and b == -32768:
            return 32767
        ab = int(a) * int(b)
        r = (ab << 1) + (1 << 15)
        r >>= 16
        return cls._clamp(r, -32768, 32767)

    @classmethod
    def _sqdmulh_s16(cls, a: int, b: int) -> int:
        overflow = (a == -32768 and b == -32768)
        ab = int(a) * int(b)
        q15 = int(ab / (1 << 15))
        if overflow:
            q15 = 32767
        return cls._clamp(q15, -32768, 32767)

    @classmethod
    def _divide_by_power_of_two_s16(cls, x: int, exponent: int) -> int:
        v = cls._divide_by_power_of_two(int(x), int(exponent))
        return cls._clamp(v, -32768, 32767)

    @classmethod
    def _simulate_precise_s8(
        cls,
        input_q: np.ndarray,
        input_offset: int,
        output_offset: int,
        output_multiplier: int,
        output_shift: int,
        relu_q3: int,
        relu_q6: int,
        prescale: int,
    ) -> np.ndarray:
        flat = input_q.flatten()
        out = np.empty_like(flat, dtype=np.int8)
        for i, v in enumerate(flat):
            x = int(v) - int(input_offset)
            xr = cls._clamp(x + int(relu_q3), 0, int(relu_q6))
            xr = cls._nonneg_divide_by_pot_s32(xr, int(prescale))
            y = x * xr
            y = cls._requantize(y, int(output_multiplier), int(output_shift))
            y += int(output_offset)
            y = cls._clamp(y, -128, 127)
            out[i] = np.int8(y)
        return out.reshape(input_q.shape)

    @classmethod
    def _simulate_precise_s16(
        cls,
        input_q: np.ndarray,
        input_offset: int,
        output_offset: int,
        output_multiplier: int,
        output_shift: int,
        relu_q3: int,
        relu_q6: int,
        prescale: int,
    ) -> np.ndarray:
        flat = input_q.flatten()
        out = np.empty_like(flat, dtype=np.int16)
        for i, v in enumerate(flat):
            x = int(v) - int(input_offset)
            xr = cls._clamp(x + int(relu_q3), 0, int(relu_q6))
            xr = cls._nonneg_divide_by_pot_s32(xr, int(prescale))
            y = x * xr
            y = cls._requantize(y, int(output_multiplier), int(output_shift))
            y += int(output_offset)
            y = cls._clamp(y, -32768, 32767)
            out[i] = np.int16(y)
        return out.reshape(input_q.shape)

    @classmethod
    def _simulate_compat_s8(
        cls,
        input_q: np.ndarray,
        input_offset: int,
        output_offset: int,
        output_multiplier_fp: int,
        output_multiplier_exp: int,
        relu_multiplier_fp: int,
        relu_multiplier_exp: int,
    ) -> np.ndarray:
        flat = input_q.flatten()
        out = np.empty_like(flat, dtype=np.int8)
        for i, v in enumerate(flat):
            x = int(v) - int(input_offset)
            hires = cls._sat_lshift_s16(x, 7)
            y_pre = cls._sqrdmulh_s16(hires, int(output_multiplier_fp))
            rel = hires
            if relu_multiplier_exp > 0:
                rel = cls._sat_lshift_s16(rel, int(relu_multiplier_exp) - 1)
                rel = cls._sqrdmulh_s16(rel, int(relu_multiplier_fp))
                rel = cls._sat_lshift_s16(rel, 1)
            elif relu_multiplier_exp < 0:
                rel = cls._sqrdmulh_s16(rel, int(relu_multiplier_fp))
                rel = cls._divide_by_power_of_two_s16(rel, -int(relu_multiplier_exp))
            else:
                rel = cls._sqrdmulh_s16(rel, int(relu_multiplier_fp))
            rel = int((int(rel) + 32768) >> 1)
            y = cls._sqdmulh_s16(rel, y_pre)
            if output_multiplier_exp < 0:
                y = cls._divide_by_power_of_two_s16(y, -int(output_multiplier_exp))
            y += int(output_offset)
            y = cls._clamp(y, -128, 127)
            out[i] = np.int8(y)
        return out.reshape(input_q.shape)

    def _variant(self) -> str:
        extras = self.desc.get("hint", {}).get("extras", {})
        variant = str(extras.get("variant", "precise")).strip().lower()
        return variant

    @staticmethod
    def _quantize_multiplier_q31(real_scale: float) -> Tuple[int, int]:
        if real_scale == 0.0:
            return 0, 0
        import math
        significand, exponent = math.frexp(real_scale)
        q31 = int(math.floor(significand * (1 << 31) + 0.5))
        if q31 == (1 << 31):
            q31 //= 2
            exponent += 1
        return q31, exponent

    @staticmethod
    def _downscale_q31_to_q15(q31: int) -> int:
        q15 = int((q31 + (1 << 15)) >> 16)
        if q15 == (1 << 15):
            q15 = (1 << 15) - 1
        return q15

    @classmethod
    def _to_q15_exp(cls, real_scale: float) -> Tuple[int, int]:
        q31, exp = cls._quantize_multiplier_q31(real_scale)
        q15 = cls._downscale_q31_to_q15(q31)
        return q15, exp
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for HardSwish operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
        import math
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_hard_swish_kernel()
        variant = self._variant()
        
        input_shape = tuple(self.desc['input_shape'])
        output_shape = input_shape
        input_scale = None
        input_zp = None
        output_scale = None
        output_zp = None

        if tflite_path.exists():
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

            # Extract quantization from LiteRT
            input_quant = op_tensors['inputs'][0]['quantization']
            output_quant = op_tensors['outputs'][0]['quantization']

            input_scale = input_quant.get('scale', 1.0)
            input_zp = input_quant.get('zero_point', 0)
            output_scale = output_quant.get('scale', 1.0)
            output_zp = output_quant.get('zero_point', 0)

            # Handle per-channel quantization (convert to scalar)
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
        else:
            extras = self.desc.get("hint", {}).get("extras", {})
            input_scale = float(extras.get("input_scale", 1.0))
            output_scale = float(extras.get("output_scale", input_scale))
            input_zp = int(extras.get("input_zero_point", 0))
            output_zp = int(extras.get("output_zero_point", 0))
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        # Select input dtype before prescale computation
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        # HardSwish formula: x * ReLU6(x + 3) / 6
        # For quantization:
        # relu_q3 = tflite_round(3 / input_scale)
        # relu_q6 = tflite_round(6 / input_scale)
        # prescale = compute_prescale(relu_q6, dtype) to avoid overflow
        # M = ((input_scale^2) / (6.0 * output_scale)) * (1 << prescale)
        
        def tflite_round(x: float) -> int:
            return int(np.floor(x + 0.5) if x >= 0.0 else np.ceil(x - 0.5))

        relu_q3 = tflite_round(3.0 / float(input_scale))
        relu_q6 = tflite_round(6.0 / float(input_scale))
        
        # Determine prescale to avoid overflow in the precise variant
        prescale = 0
        prod_max = 32767 * int(relu_q6)
        dtype_max = np.iinfo(np_in_dtype).max
        while prod_max > dtype_max:
            prescale += 1
            prod_max >>= 1
        
        # Calculate effective scale
        # Formula: real_multiplier = (input_scale^2) / (6.0 * output_scale)
        real_multiplier = (float(input_scale) * float(input_scale)) / (6.0 * float(output_scale))
        real_multiplier_adj = real_multiplier * (1 << prescale)
        
        # Calculate multiplier and shift
        output_mult, output_shift = calculate_multiplier_shift(real_multiplier_adj)
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_data = self.rng.uniform(-8.0, 8.0, size=input_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        # Quantize inputs
        input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
        
        compat_out_fp = None
        compat_out_exp = None
        compat_relu_fp = None
        compat_relu_exp = None
        if variant == "compat":
            if kernel_info["input_c_type"] != "int8_t":
                raise NotImplementedError("HardSwish compat is only supported for S8.")
            hires_input_scale = (1.0 / 128.0) * float(input_scale)
            out_mul_real = hires_input_scale / float(output_scale)
            reluish_scale = 3.0 / 32768.0
            relu_mul_real = hires_input_scale / reluish_scale
            out_q15, out_exp = self._to_q15_exp(out_mul_real)
            relu_q15, relu_exp = self._to_q15_exp(relu_mul_real)
            if out_exp > 0:
                raise ValueError(f"Unexpected positive output exponent ({out_exp}) for HardSwish compat.")
            compat_out_fp = int(out_q15)
            compat_out_exp = int(out_exp)
            compat_relu_fp = int(relu_q15)
            compat_relu_exp = int(relu_exp)

        # Compute expected output using CMSIS-NN-compatible emulation
        if variant == "compat":
            output_data = self._simulate_compat_s8(
                input_q,
                int(input_zp),
                int(output_zp),
                int(compat_out_fp),
                int(compat_out_exp),
                int(compat_relu_fp),
                int(compat_relu_exp),
            )
        else:
            if np_in_dtype == np.int16:
                output_data = self._simulate_precise_s16(
                    input_q,
                    int(input_zp),
                    int(output_zp),
                    int(output_mult),
                    int(output_shift),
                    int(relu_q3),
                    int(relu_q6),
                    int(prescale),
                )
            else:
                output_data = self._simulate_precise_s8(
                    input_q,
                    int(input_zp),
                    int(output_zp),
                    int(output_mult),
                    int(output_shift),
                    int(relu_q3),
                    int(relu_q6),
                    int(prescale),
                )
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate output size (total number of elements)
        output_size = int(np.prod(output_shape))
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'input_offset': int(input_zp),
            'output_offset': int(output_zp),
            'output_mult': int(output_mult),
            'output_shift': int(output_shift),
            'relu_q3': int(relu_q3),
            'relu_q6': int(relu_q6),
            'prescale': int(prescale),
            'output_size': int(output_size),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }

        if variant == "compat":
            context.update({
                'output_multiplier_fp': int(compat_out_fp),
                'output_multiplier_exp': int(compat_out_exp),
                'relu_multiplier_fp': int(compat_relu_fp),
                'relu_multiplier_exp': int(compat_relu_exp),
            })
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("hardswish/hardswish.h.j2", context)
        h_path = includes_api_dir / f"{name}_hardswish.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        template = "hardswish/hardswish.c.j2"
        if variant == "compat":
            template = "hardswish/hardswish_compat.c.j2"
        c_content = self.render_template(template, context)
        c_path = output_dir / f"{name}_hardswish.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'HardSwish'),
            'operator_name': 'hardswish'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")
