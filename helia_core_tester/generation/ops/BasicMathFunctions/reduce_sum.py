"""
ReduceSum operation implementation.

Float-only: ns-cmsis-nn provides arm_reduce_sum_f32/f16 (no quantized sum
kernels exist; quantized SUM is lowered onto arm_mean_* by the compiler).
Goldens are computed with numpy using float32 accumulation for both dtypes,
matching the kernels' documented accumulation semantics.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpReduceSum(OperationBase):
    """
    ReduceSum operation (FP32/FP16).
    """

    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for ReduceSum operation."""
        input_shape = self.desc['input_shape']
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')

        axes = self.desc.get('axes', [1, 2])
        keepdims = self.desc.get('keepdims', True)

        x = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=axes, keepdims=keepdims),
            name='reduce_sum'
        )(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert to a plain float32 TFLite model (no quantization)."""
        activation_dtype = self.tensor_dtype("input", default="FP32")
        if activation_dtype not in ('FP32', 'FP16'):
            raise NotImplementedError(
                f"Unsupported ReduceSum dtype: {activation_dtype} (float-only kernels)")

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        self._write_tflite_bytes(out_path, tflite_model)

    def _select_cmsis_reduce_sum_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for ReduceSum operation.
        """
        activation_dtype = self.tensor_dtype("input", default="FP32")

        if activation_dtype == 'FP32':
            return {
                'kernel_fn': 'arm_reduce_sum_f32',
                'input_c_type': 'float',
                'output_c_type': 'float',
                'float_kernel': True,
            }
        elif activation_dtype == 'FP16':
            return {
                'kernel_fn': 'arm_reduce_sum_f16',
                'input_c_type': 'float16_t',
                'output_c_type': 'float16_t',
                'float_kernel': True,
            }
        else:
            raise NotImplementedError(
                f"Unsupported ReduceSum dtype: {activation_dtype} (float-only kernels)")

    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for ReduceSum operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        kernel_info = self._select_cmsis_reduce_sum_kernel()
        float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32

        input_shape = tuple(self.desc['input_shape'])

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)

        axes = self.desc.get('axes', [1, 2])
        if not isinstance(axes, list):
            axes = [axes]

        # Keep CMSIS output dims 4D even when TFLite output rank is reduced
        axis_dims_cmsis = builder.build_reduce_axis_dims(len(input_shape), axes)
        output_dims = builder.build_reduce_output_dims(
            input_shape=input_shape,
            axes=axes,
            keepdims=bool(self.desc.get('keepdims', True))
        )

        input_q = self._sample_uniform(input_shape, dtype=float_dtype)

        # Golden with float32 accumulation for both dtypes, matching the
        # kernels' documented semantics (single final rounding for f16).
        output_data = np.sum(
            input_q.astype(np.float32), axis=tuple(axes), keepdims=True
        ).astype(float_dtype)

        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'axis_dims': axis_dims_cmsis,
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'float_kernel': True,
            'validation_mode': 'float',
        }

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'ReduceSum'),
            'operator_name': 'reduce_sum',
        }
        self._write_op_outputs(
            output_dir,
            "reduce_sum",
            "BasicMathFunctions/reduce_sum/reduce_sum.h.j2",
            "BasicMathFunctions/reduce_sum/reduce_sum.c.j2",
            context,
            cmake_context,
        )
