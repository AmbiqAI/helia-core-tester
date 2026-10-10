"""
GELU operation implementation.

Float32-only: ns-cmsis-nn provides arm_nn_gelu_f32 (ns-cmsis-nn #743), the
exact GELU 0.5 * x * erfc(-x / sqrt(2)), not the tanh approximation. Goldens
come from that expression in float64, rounded once to float32.
"""

import math
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from helia_core_tester.generation.ops._shared.base import OperationBase

_ERFC = np.vectorize(math.erfc, otypes=[np.float64])


def gelu_exact_reference(x: np.ndarray) -> np.ndarray:
    """Exact GELU in float64.

    The kernel's own ordering, so the non-finite lanes follow IEEE 754:
    +Inf -> +Inf, NaN -> NaN, and -Inf -> NaN because erfc(+Inf) is 0 and
    (-Inf) * 0 is NaN.
    """
    x64 = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return 0.5 * x64 * _ERFC(-x64 / math.sqrt(2.0))


class OpGelu(OperationBase):
    """
    GELU operation (FP32).
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
        activation_dtype = self.tensor_dtype("input", default="FP32")
        if activation_dtype != 'FP32':
            raise NotImplementedError(f"Unsupported GELU dtype: {activation_dtype} (FP32 kernel only)")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        self._write_tflite_bytes(out_path, converter.convert())

    def _select_cmsis_gelu_kernel(self) -> Dict[str, str]:
        activation_dtype = self.tensor_dtype("input", default="FP32")
        if activation_dtype != 'FP32':
            raise NotImplementedError(f"Unsupported GELU dtype: {activation_dtype} (FP32 kernel only)")
        return {
            'kernel_fn': 'arm_nn_gelu_f32',
            'input_c_type': 'float',
            'output_c_type': 'float',
        }

    def generate_c_files(self, output_dir: Path) -> None:
        """Generate C and H files from templates for GELU operation."""
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_info = self._select_cmsis_gelu_kernel()
        input_shape = tuple(self.desc['input_shape'])
        builder = TemplateContextBuilder()

        input_data = self._sample_uniform(
            input_shape,
            low=float(self.desc.get("input_min", -8.0)),
            high=float(self.desc.get("input_max", 8.0)),
            dtype=np.float32,
        )

        def reference(operands):
            return gelu_exact_reference(operands[0]).astype(np.float32)

        output_data = reference([input_data])
        output_data, nonfinite_context = self.apply_nonfinite_policy(
            output_data, reference=reference, inputs=[input_data]
        )

        context = {
            'name': name,
            'output_size': int(np.prod(input_shape)),
            'input_data_array': builder.format_array_as_c_literal(input_data),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'validation_mode': 'float',
        }
        context.update(nonfinite_context)

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
