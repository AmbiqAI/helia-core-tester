"""
Dequantize operation implementation.
"""

import numpy as np
from pathlib import Path
from helia_core_tester.generation.entry import check_entry_fault, resolve_entry
from helia_core_tester.generation.ops._shared.quantization_base import QuantizationFamilyBase
from helia_core_tester.generation.ops.QuantizationFunctions.pools import dequantize_argument_pool, dequantize_f16_bits_pool

# Binary16 patterns a bit-pattern case starts with, as many as fit before its NaN tail, so that even
# a short case meets the NaN rule: a signalling NaN, a negative quiet NaN, a payload NaN, +Inf, the
# smallest subnormal, -0 and 1.0, then the remaining zeros, subnormals, normal-range ends, -Inf and
# NaNs of both signs. Uniform random patterns fill the rest of a case up to its tail.
_F16_BIT_CLASSES = (
    0x7C01, 0xFE00, 0x7FFF, 0x7C00, 0x0001, 0x8000, 0x3C00,
    0x0000, 0x8001, 0x0200, 0x03FF, 0x83FF, 0x0400, 0x8400, 0xC000, 0x3555,
    0x7BFF, 0xFBFF, 0xFC00, 0x7E00, 0x7E01, 0xFFFF, 0xFC01, 0x7D55, 0x7DFF, 0xFD00,
)
# The last elements of every case: a signalling NaN with payload, a negative signalling NaN and a
# quiet NaN with payload.
_F16_TAIL_NANS = (0x7D55, 0xFC01, 0x7E01)
# What a bit-pattern descriptor may carry (keys the loader adds start with "_" or "resolved_").
_F16_BITS_KEYS = frozenset(
    {"operator", "name", "suite", "hint", "entry", "tensor_dtypes", "activation_dtype", "activation"}
    | {"input_shape", "comparison"}
)


class OpDequantize(QuantizationFamilyBase):
    """
    Dequantize operation.
    """

    def needs_tflite(self) -> bool:
        # The golden is the dequantize formula (or an exact widening) in numpy.
        return False

    def _widens_f16_bits(self) -> bool:
        """An `entry:` case: arm_dequantize_f16_bits_f32 on binary16 bit patterns, checked bit for bit
        against each NaN rule rather than against a converted model."""
        return bool(self.desc.get("entry"))

    def _widens_f16(self) -> bool:
        """FP16 -> FP32 is arm_dequantize_f16_f32 (ns-cmsis-nn#475): a bit-exact widening
        with no scale or zero point, built as a LiteRT DEQUANTIZE rather than a Keras model."""
        return self.tensor_dtype("input") == "FP16"

    def _select_cmsis_dequantize_kernel(self) -> dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Dequantize operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        input_dtype = self.tensor_dtype("input")
        output_dtype = self.tensor_dtype("output")

        entry = self.desc.get("entry")
        if entry:
            # The entry reads binary16 storage: an FP16 tag runs it on the f16 legs, U16 on the f32 legs.
            storage = "U16" if input_dtype in ("FP16", "U16") else input_dtype
            # The entry widens to float32; _generate_f16_bits refuses any other output tag by name.
            resolved = resolve_entry("Dequantize", str(entry), activation_dtype=storage, weight_dtype=storage,
                                     cpu=self.target_cpu, desc=self.desc, extra_roles={"output": "FP32"})
            check_entry_fault(self.desc, resolved)
            return {
                'kernel_fn': resolved["kernel_fn"],
                'input_c_type': 'uint16_t',
                'output_c_type': 'float',
                'kernel_style': 'f16_bits',
            }

        if output_dtype != "FP32":
            raise NotImplementedError(f"Dequantize currently requires FP32 output, got {output_dtype}")
        
        if input_dtype == 'S8':
            return {
                'kernel_fn': 'arm_dequantize_s8_f32',
                'input_c_type': self.tensor_c_type("input"),
                'output_c_type': self.tensor_c_type("output"),
            }
        if input_dtype == 'S16':
            return {
                'kernel_fn': 'arm_dequantize_s16_f32',
                'input_c_type': self.tensor_c_type("input"),
                'output_c_type': self.tensor_c_type("output"),
            }
        if input_dtype == 'FP16':
            return {
                'kernel_fn': 'arm_dequantize_f16_f32',
                'input_c_type': self.tensor_c_type("input"),
                'output_c_type': self.tensor_c_type("output"),
                'kernel_style': 'widen',
            }
        raise NotImplementedError(f"Unsupported Dequantize input dtype: {input_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Dequantize operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_dequantize_kernel()
        
        builder = TemplateContextBuilder()
        comparison = self.comparison_config()

        # The CMSIS-NN arm_dequantize_* kernels only dequantize, so apply any
        # descriptor activation in C to match TFLite behavior.
        activation_str = self.activation_name()
        has_activation = activation_str in ['RELU', 'RELU6']

        if kernel_info.get("kernel_style") == "f16_bits":
            self._generate_f16_bits(output_dir, kernel_info)
            return

        if kernel_info.get("kernel_style") == "widen":
            # No quantization parameters: the kernel widens every float16 bit
            # pattern exactly, so numpy's astype is the reference. input_min/
            # input_max bound the uniform draw; a range inside +/-6.1e-5 keeps
            # every draw a float16 subnormal, and input_mode nonfinite_sweep
            # puts NaN/+Inf/-Inf in the leading lanes.
            input_shape = tuple(int(dim) for dim in self.desc["input_shape"])
            low = float(self.desc.get("input_min", -1.0))
            high = float(self.desc.get("input_max", 1.0))
            input_q = self._sample_uniform(input_shape, low=low, high=high, dtype=np.float16)
            output_data = input_q.astype(np.float32)
            input_scale = 0.0
            input_zp = 0
            if has_activation:
                if activation_str == 'RELU':
                    output_data = np.maximum(output_data, 0.0)
                else:
                    output_data = np.clip(output_data, 0.0, 6.0)
        else:
            input_shape = tuple(int(d) for d in self.desc["input_shape"])
            kind = "s16" if kernel_info["input_c_type"] == "int16_t" else "s8"
            # The [-1, 1] calibration range the converter quantized the input over.
            quant = self.activation_quant("input", np.array([-1.0, 1.0], dtype=np.float32), kind)
            input_scale, input_zp = quant.scale, quant.zero_point

            # For S16, generate input data that better utilizes the quantization range
            # Calculate the effective float range based on scale and zero_point
            if kernel_info["input_c_type"] == "int8_t":
                np_in_dtype = np.int8
                qmin, qmax = -128, 127
                # For S8, use standard range
                input_data_float = self._sample_uniform(input_shape)
            elif kernel_info["input_c_type"] == "int16_t":
                np_in_dtype = np.int16
                qmin, qmax = -32768, 32767
                # For S16, generate data that will quantize to a good range of int16 values
                # Calculate float range that maps to a reasonable subset of int16 range
                # Use approximately 80% of the int16 range to avoid edge cases
                range_fraction = 0.8
                effective_qmin = int(qmin + (1.0 - range_fraction) * (qmax - qmin) / 2)
                effective_qmax = int(qmax - (1.0 - range_fraction) * (qmax - qmin) / 2)
                float_min = (effective_qmin - input_zp) * input_scale
                float_max = (effective_qmax - input_zp) * input_scale
                # Ensure we have a valid range
                if float_max > float_min:
                    input_data_float = self._sample_uniform(input_shape, low=float_min, high=float_max)
                else:
                    # Fallback to standard range if calculated range is invalid
                    input_data_float = self._sample_uniform(input_shape)
            else:
                raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
            
            # Quantize inputs: quantized = round(float_value / scale + zero_point)
            input_q = np.round(input_data_float / float(input_scale) + float(input_zp)).astype(np.int32)
            input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
            
            output_data = (input_q.astype(np.float32) - float(input_zp)) * float(input_scale)

            if has_activation:
                if activation_str == 'RELU':
                    output_data = np.maximum(output_data, 0.0)
                else:
                    output_data = np.clip(output_data, 0.0, 6.0)

        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate size
        input_size = int(np.prod(input_shape))
        
        # Build template context
        context = {
            'name': name,
            'input_size': input_size,
            'zero_point': int(input_zp),
            'scale': float(input_scale),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_style': kernel_info.get("kernel_style", "scale_offset"),
            'has_activation': has_activation,
            'activation_type': activation_str if has_activation else 'NONE',
            'comparison_atol': float(comparison.get("atol", 0.0)),
            'comparison_rtol': float(comparison.get("rtol", 0.0)),
            'validation_helpers': ['float'],
        }
        
        self.render_harness_case(
            output_dir, stem="dequantize", context=context, pool=dequantize_argument_pool(context),
            validation_key="QuantizationFunctions/dequantize/dequantize.c.j2", label="Dequantize", operator="Dequantize", sidecar=True,
        )
        

    def _generate_f16_bits(self, output_dir: Path, kernel_info: dict) -> None:
        """A bit-pattern case: binary16 inputs and the float32 bits each NaN rule gives.

        Finite values and infinities widen exactly. A NaN becomes the default NaN 0x7FC00000 on the
        MVE vector conversion; the scalar conversion, and the integer widening that copies it, keep
        its sign and payload and set the quiet bit. The C file picks the rule its build compiled.
        """
        name = self.desc["name"]
        unknown = sorted(k for k in self.desc if k not in _F16_BITS_KEYS and not k.startswith(("_", "resolved_")))
        if self.activation_name() != "NONE":
            unknown.append("activation")
        if self.tensor_dtype("output") != "FP32":
            unknown.append("tensor_dtypes.output (must be FP32)")
        if any(role not in ("input", "output") for role in (self.desc.get("tensor_dtypes") or {})):
            unknown.append("tensor_dtypes beyond input and output")
        comparison = self.desc.get("comparison")
        if comparison is not None and comparison != {"atol": 0.0, "rtol": 0.0}:
            unknown.append("comparison (the check is bit-exact)")
        if unknown:
            raise ValueError(f"{name}: a bit-pattern case takes none of {unknown}")
        shape = self.desc["input_shape"]
        dims_ok = isinstance(shape, list) and all(type(dim) is int and dim > 0 for dim in shape)
        size = int(np.prod(shape, dtype=object)) if dims_ok else 0
        if not 1 <= size <= 1 << 16:
            raise ValueError(f"{name}: a bit-pattern case holds 1 to 65536 halves in positive int dims, got {shape}")
        tail = min(3, size)
        # The MVE path converts four halves per iteration and the FPU path two, so the last elements
        # sit in the predicated or single-half tail; ending on NaNs puts the NaN rule there too.
        classes = np.array(_F16_BIT_CLASSES[: size - tail], dtype=np.uint16)
        rest = self.rng.integers(0, 1 << 16, size=size - tail - classes.size).astype(np.uint16)
        bits = np.concatenate([classes, rest, np.array(_F16_TAIL_NANS[-tail:], dtype=np.uint16)])

        frac = (bits & 0x3FF).astype(np.uint32)
        sign = (bits >> 15).astype(np.uint32)
        is_nan = ((bits >> 10) & 0x1F == 0x1F) & (frac != 0)
        widened = bits.view(np.float16).astype(np.float32).view(np.uint32)
        scalar = np.where(is_nan, (sign << 31) | 0x7FC00000 | (frac << 13), widened).astype(np.uint32)
        vector = np.where(is_nan, np.uint32(0x7FC00000), widened).astype(np.uint32)

        def hex_rows(values, digits):
            items = [f"0x{int(v):0{digits}X}u" for v in values]
            return ",\n".join("    " + ", ".join(items[i:i + 8]) for i in range(0, len(items), 8))

        context = {
            'name': name,
            'input_size': size,
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_style': 'f16_bits',
            'input_dtype': 'uint16_t',
            'output_dtype': 'float',
            'input_data_array': hex_rows(bits, 4),
            'expected_vector_bits_array': hex_rows(vector, 8),
            'expected_scalar_bits_array': hex_rows(scalar, 8),
            'nan_count': int(is_nan.sum()),
            'has_activation': False,
            'activation_type': 'NONE',
            'zero_point': 0,
            'scale': 0.0,
            'comparison_atol': 0.0,
            'comparison_rtol': 0.0,
            'validation_helpers': ['float'],
        }
        self.render_harness_case(
            output_dir, stem="dequantize", context=context, pool=dequantize_f16_bits_pool(context),
            validation_key="QuantizationFunctions/dequantize/dequantize.c.j2", label="Dequantize", operator="Dequantize",
            sidecar=True,
        )
