"""
GELU operation implementation.

ns-cmsis-nn provides arm_nn_gelu_f32 (ns-cmsis-nn #743) and arm_nn_gelu_f16
(ns-cmsis-nn #761), the exact GELU 0.5 * x * erfc(-x / sqrt(2)), not the tanh
approximation; the f16 kernel evaluates it in float32 and rounds once. The
reference is that expression in float64 at the exactly widened input.

A descriptor either compares against the reference rounded once to the output
type under its `comparison` tolerance, or names a `contract_interval`, in which
case every element is checked against the exact set of outputs the contract
allows (see _shared/contract_interval.py).
"""

import math
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.contract_interval import contract_interval

_ERFC = np.vectorize(math.erfc, otypes=[np.float64])
_KERNELS = {
    'FP32': ('arm_nn_gelu_f32', 'float', np.float32, np.uint32),
    'FP16': ('arm_nn_gelu_f16', 'float16_t', np.float16, np.uint16),
}
# FP16 results below the smallest normal may flush to a signed zero under FZ16.
_FP16_MIN_NORMAL = 2.0**-14


def gelu_exact_reference(x: np.ndarray) -> np.ndarray:
    """Exact GELU in float64.

    The kernel's own ordering, so the non-finite lanes follow IEEE 754:
    +Inf -> +Inf, NaN -> NaN, and -Inf -> NaN because erfc(+Inf) is 0 and
    (-Inf) * 0 is NaN.
    """
    x64 = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return 0.5 * x64 * _ERFC(-x64 / math.sqrt(2.0))


def gelu_contract_reference(x: np.ndarray) -> np.ndarray:
    """gelu_exact_reference, with deep-underflow lanes kept decidable for contract intervals.

    Below x of about -37.2 the float64 result is under 2^-1000; it is subnormal
    from about -37.6 and zero below about -38.5, while the true value is a tiny
    number with the sign of x. With the contract's atol far above 2^-1000,
    every such value gives the same interval, so those lanes use 2^-1000 with
    the sign of x, which float64 resolves. Only an input zero keeps a zero reference, which
    pins the zero of its own sign.
    """
    x64 = np.asarray(x, dtype=np.float64).ravel()
    ref = gelu_exact_reference(x64)
    deep = (np.abs(ref) < 2.0**-1000) & (x64 != 0)
    ref[deep] = np.copysign(2.0**-1000, x64[deep])
    return ref


class OpGelu(OperationBase):
    """
    GELU operation (FP32, FP16).
    """

    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for exact GELU."""
        input_shape = self.desc['input_shape']
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        x = tf.keras.layers.Lambda(
            lambda t: tf.nn.gelu(t, approximate=False),
            name='gelu'
        )(inputs)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert to a plain float32 TFLite model (no quantization)."""
        self._kernel()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        self._write_tflite_bytes(out_path, converter.convert())

    def comparison_config(self):
        """The contract bound for contract_interval cases, recorded as such in the sidecar."""
        bound = self.desc.get('contract_interval')
        if bound is None:
            return super().comparison_config()
        return {'mode': 'interval', 'rtol': float(bound['rtol']), 'atol': float(bound['atol'])}

    def _kernel(self):
        activation_dtype = self.tensor_dtype("input", default="FP32")
        if activation_dtype not in _KERNELS:
            raise NotImplementedError(f"Unsupported GELU dtype: {activation_dtype}")
        return _KERNELS[activation_dtype]

    def _inputs(self, dtype, bits_dtype) -> np.ndarray:
        """Uniform draws, or every bit pattern in the descriptor's input_bits ranges."""
        input_shape = tuple(self.desc['input_shape'])
        ranges = self.desc.get('input_bits')
        if ranges is None:
            return self._sample_uniform(
                input_shape,
                low=float(self.desc.get("input_min", -8.0)),
                high=float(self.desc.get("input_max", 8.0)),
                dtype=dtype,
            )
        limit = 1 << (8 * np.dtype(bits_dtype).itemsize)
        name = self.desc['name']
        if any(key in self.desc for key in ('input_mode', 'input_min', 'input_max')):
            raise ValueError(f"Descriptor {name!r}: input_bits replaces input_mode, input_min and input_max")
        if not ranges or any(not 0 <= start < stop <= limit for start, stop in ranges):
            raise ValueError(f"Descriptor {name!r}: input_bits needs non-empty [start, stop) ranges within [0, {limit:#x}]")
        bits = np.concatenate([np.arange(start, stop, dtype=np.uint64) for start, stop in ranges])
        if bits.size != int(np.prod(input_shape)):
            raise ValueError(
                f"Descriptor {self.desc['name']!r}: input_bits names {bits.size} inputs but "
                f"input_shape {list(input_shape)} holds {int(np.prod(input_shape))}"
            )
        return bits.astype(bits_dtype).view(dtype).reshape(input_shape)

    def generate_c_files(self, output_dir: Path) -> None:
        """Generate C and H files from templates for GELU operation."""
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_fn, c_type, dtype, bits_dtype = self._kernel()
        builder = TemplateContextBuilder()
        input_data = self._inputs(dtype, bits_dtype)

        def reference(operands):
            with np.errstate(over="ignore", invalid="ignore"):
                return gelu_exact_reference(operands[0]).astype(dtype)

        output_data = reference([input_data])
        output_data, nonfinite_context = self.apply_nonfinite_policy(
            output_data, reference=reference, inputs=[input_data]
        )

        context = {
            'name': name,
            'output_size': int(input_data.size),
            'input_data_array': builder.format_array_as_c_literal(input_data),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': c_type,
            'output_dtype': c_type,
            'kernel_fn': kernel_fn,
            'validation_mode': 'float',
            'fpscr_fz16': bool(self.desc.get('fpscr_fz16', False)),
        }
        if context['fpscr_fz16'] and (c_type != 'float16_t' or 'contract_interval' not in self.desc):
            raise ValueError(f"Descriptor {name!r}: fpscr_fz16 applies to float16 cases with a contract_interval")
        context.update(nonfinite_context)

        bound = self.desc.get('contract_interval')
        if bound is not None:
            ref = gelu_contract_reference(input_data)
            lo, hi = contract_interval(ref, dtype, float(bound['rtol']), float(bound['atol']))
            context.update({
                'validation_rtol': float(bound['rtol']),
                'validation_atol': float(bound['atol']),
                'interval_width': np.dtype(dtype).itemsize,
                'interval_bits_type': f'uint{8 * np.dtype(dtype).itemsize}_t',
                'interval_lo_array': builder.format_array_as_c_literal(lo),
                'interval_hi_array': builder.format_array_as_c_literal(hi),
            })
            if context['fpscr_fz16']:
                # Under FZ16 a subnormal result may be a zero with the sign of ref.
                flush = np.isfinite(ref) & (ref != 0) & (np.abs(ref) < _FP16_MIN_NORMAL)
                zero_ok = np.where(flush, np.where(ref < 0, 2, 1), 0).astype(np.uint8)
                context['interval_zero_ok_array'] = builder.format_array_as_c_literal(zero_ok)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Gelu'),
            'operator_name': 'gelu',
        }
        self._write_op_outputs(
            output_dir,
            "gelu",
            "ActivationFunctions/gelu/gelu.h.j2",
            "ActivationFunctions/gelu/gelu.c.j2",
            context,
            cmake_context,
        )
