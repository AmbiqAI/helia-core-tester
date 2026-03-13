"""Shared implementation for CMSIS-NN hard-swish operators."""

from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class HardSwishFamilyBase(OperationBase):
    """Shared implementation for precise and compat hard-swish generation."""

    VARIANT = "precise"
    OPERATOR_NAME = "HardSwishPrecise"
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for HardSwish operation."""
        input_shape = self.desc['input_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        output = tf.keras.layers.Activation('hard_swish')(inputs)
            
        model = tf.keras.Model(inputs=[inputs], outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if self.variant_name() == "compat" and str(activation_dtype).upper() != "S8":
            raise NotImplementedError("HardSwishCompat is only supported for S8.")
        if self.variant_name() == "precise" and str(activation_dtype).upper() == "S16":
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
        self._write_tflite_bytes(out_path, tflite_model)

    def variant_name(self) -> str:
        return self.VARIANT
    
    def _select_cmsis_hard_swish_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for HardSwish operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        variant = self.variant_name()
        
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
        variant = self.variant_name()
        
        input_shape = tuple(self.desc['input_shape'])
        output_shape = input_shape
        input_scale = None
        input_zp = None
        output_scale = None
        output_zp = None

        if tflite_path.exists():
            op_tensors = self.load_primary_operator_tensors(str(tflite_path))

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

            input_scale = float(self._quant_param_scalar(input_quant, 'scale', 1.0))
            input_zp = int(self._quant_param_scalar(input_quant, 'zero_point', 0))
            output_scale = float(self._quant_param_scalar(output_quant, 'scale', 1.0))
            output_zp = int(self._quant_param_scalar(output_quant, 'zero_point', 0))

            extras = self.desc.get("hint", {}).get("extras", {})
            if variant == "compat" and extras:
                if "input_scale" in extras:
                    input_scale = float(extras["input_scale"])
                if "output_scale" in extras:
                    output_scale = float(extras["output_scale"])
                if "input_zero_point" in extras:
                    input_zp = int(extras["input_zero_point"])
                if "output_zero_point" in extras:
                    output_zp = int(extras["output_zero_point"])
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
        input_data = self._sample_uniform(input_shape, low=-8.0, high=8.0)
        
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
        
        template = "hard_swish/hard_swish.c.j2"
        if variant == "compat":
            template = "hard_swish/hard_swish_compat.c.j2"
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', self.OPERATOR_NAME),
            'operator_name': 'hard_swish',
        }
        self._write_op_outputs(output_dir, "hard_swish", "hard_swish/hard_swish.h.j2", template, context, cmake_context)
        
