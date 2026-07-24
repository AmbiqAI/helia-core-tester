"""Direct scalar-input PReLU test generation for arm_prelu_scalar_s8/arm_prelu_scalar_s16."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf
from pathlib import Path

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift, requantize_np

# Per-dtype quantization parameters: C type name, numpy dtype, and clamp range.
_DTYPE_INFO = {
    "S8": {"c_type": "int8_t", "np_dtype": np.int8, "qmin": -128, "qmax": 127, "kernel_fn": "arm_prelu_scalar_s8"},
    "S16": {
        "c_type": "int16_t",
        "np_dtype": np.int16,
        "qmin": -32768,
        "qmax": 32767,
        "kernel_fn": "arm_prelu_scalar_s16",
    },
}


class _ScalarInputPreluReference(tf.keras.layers.Layer):
    """Keras reference layer that enforces explicit float casting before PReLU math."""

    def call(self, inputs):
        scalar_tensor, alpha_tensor = inputs
        scalar_f32 = tf.cast(scalar_tensor, tf.float32)
        alpha_f32 = tf.cast(alpha_tensor, tf.float32)
        return tf.where(scalar_f32 >= 0.0, scalar_f32, scalar_f32 * alpha_f32)


class OpPReLUScalar(OperationBase):
    """Generate direct scalar-input arm_prelu_scalar_s8/arm_prelu_scalar_s16 tests."""

    def allow_no_tflite(self) -> bool:
        return True

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("PReLUScalar uses direct-kernel generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        raise NotImplementedError("PReLUScalar does not require a .tflite model.")

    @staticmethod
    def _resolve_alpha_values(alpha_shape: tuple[int, ...], values: Iterable[float] | None) -> np.ndarray:
        total = int(np.prod(alpha_shape))
        if total <= 0:
            raise ValueError("alpha_shape must contain positive dimensions.")

        if values is None:
            data = np.linspace(0.05, 0.25, num=total, dtype=np.float32)
        else:
            data = np.asarray(list(values), dtype=np.float32)
            if data.size == 1:
                data = np.full(total, float(data[0]), dtype=np.float32)
            elif data.size != total:
                raise ValueError(
                    f"alpha_values has {data.size} entries, expected {total} for alpha_shape {alpha_shape}."
                )
        return data.reshape(alpha_shape)

    @staticmethod
    def _quantize(values: np.ndarray, scale: float, zero_point: int, qmin: int, qmax: int, np_dtype) -> np.ndarray:
        quant = np.round(values / float(scale) + int(zero_point)).astype(np.int32)
        quant = np.clip(quant, qmin, qmax)
        return quant.astype(np_dtype)

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]

        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype not in _DTYPE_INFO:
            raise NotImplementedError(
                f"Unsupported PReLUScalar dtype: {activation_dtype} (only S8/S16 supported)"
            )
        dtype_info = _DTYPE_INFO[activation_dtype]
        np_dtype = dtype_info["np_dtype"]
        qmin, qmax = dtype_info["qmin"], dtype_info["qmax"]

        input_shape = tuple(int(dim) for dim in self.desc.get("input_shape", [1, 1, 1, 1]))
        num_pixels = int(np.prod(input_shape))
        if num_pixels <= 0:
            raise ValueError("input_shape must contain positive dimensions.")

        alpha_shape = tuple(int(dim) for dim in self.desc["alpha_shape"])
        total_alpha = int(np.prod(alpha_shape))
        if total_alpha <= 0:
            raise ValueError("alpha_shape must contain positive dimensions.")
        if total_alpha % num_pixels != 0:
            raise ValueError(
                f"alpha_shape total element count ({total_alpha}) must be a multiple of "
                f"num_pixels ({num_pixels}) derived from input_shape {input_shape}."
            )
        block_size = total_alpha // num_pixels

        # Scalar input values: a single value per pixel. Single-pixel descriptors
        # (num_pixels == 1) use the original 'scalar_input_value' field; multi-pixel
        # descriptors provide one value per pixel via hint.extras.input_values.
        if "scalar_input_value" in self.desc:
            scalar_values = [float(self.desc["scalar_input_value"])]
        else:
            extras = self.desc.get("hint", {}).get("extras", {})
            input_values = extras.get("input_values")
            if input_values is None:
                raise ValueError(
                    "PReLUScalar requires 'scalar_input_value' (single pixel) or "
                    "hint.extras.input_values (one value per pixel, multi-pixel)."
                )
            scalar_values = [float(v) for v in input_values]

        if len(scalar_values) != num_pixels:
            raise ValueError(
                f"Number of scalar input values ({len(scalar_values)}) must match "
                f"num_pixels ({num_pixels}) derived from input_shape {input_shape}."
            )
        scalar_float = np.asarray(scalar_values, dtype=np.float32)

        alpha_values = self.desc.get("alpha_values")
        if alpha_values is None:
            extras = self.desc.get("hint", {}).get("extras", {})
            alpha_values = extras.get("alpha_values")

        alpha_float = self._resolve_alpha_values(alpha_shape, alpha_values).reshape(num_pixels, block_size)

        input_scale = float(self.desc.get("input_scale", 0.125))
        alpha_scale = float(self.desc.get("alpha_scale", 0.125))
        output_scale = float(self.desc.get("output_scale", 0.125))

        input_zero_point = int(self.desc.get("input_zero_point", 0))
        alpha_zero_point = int(self.desc.get("alpha_zero_point", 0))
        output_zero_point = int(self.desc.get("output_zero_point", 0))

        scalar_q = self._quantize(scalar_float, input_scale, input_zero_point, qmin, qmax, np_dtype)
        alpha_q = self._quantize(alpha_float, alpha_scale, alpha_zero_point, qmin, qmax, np_dtype).reshape(
            num_pixels, block_size
        )

        output_multiplier_identity, output_shift_identity = calculate_multiplier_shift(input_scale / output_scale)
        output_multiplier_alpha, output_shift_alpha = calculate_multiplier_shift((input_scale * alpha_scale) / output_scale)

        input_offset = -input_zero_point
        alpha_offset = -alpha_zero_point
        output_offset = output_zero_point

        expected_q = np.zeros((num_pixels, block_size), dtype=np_dtype)
        for pixel in range(num_pixels):
            input_value = int(scalar_q[pixel]) + input_offset
            if input_value >= 0:
                out = requantize_np(
                    np.array([input_value], dtype=np.int32),
                    output_multiplier_identity,
                    output_shift_identity,
                )[0] + output_offset
                pixel_expected = np.full(block_size, int(out), dtype=np.int32)
            else:
                alpha_centered = alpha_q[pixel].astype(np.int32) + alpha_offset
                prod = alpha_centered * int(input_value)
                pixel_expected = requantize_np(prod, output_multiplier_alpha, output_shift_alpha) + output_offset
            expected_q[pixel] = np.clip(pixel_expected, qmin, qmax).astype(np_dtype)

        # Keep Keras-based reference generation in place for parity diagnostics.
        scalar_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name="scalar")
        alpha_input = tf.keras.Input(shape=(block_size,), dtype=tf.float32, name="alpha")
        reference_layer = _ScalarInputPreluReference(name="prelu_scalar_reference")
        ref_model = tf.keras.Model([scalar_input, alpha_input], reference_layer([scalar_input, alpha_input]))
        ref_float = ref_model(
            [
                tf.constant(scalar_float.reshape(num_pixels, 1), dtype=tf.float32),
                tf.constant(alpha_float, dtype=tf.float32),
            ],
            training=False,
        ).numpy()
        # NOTE: for S16 this Keras/float reference is a diagnostic-only signal, since the
        # LiteRT int16 PReLU reference (and its float-emulation path here) is known to be
        # inaccurate on the negative branch (see PR description); expected_q above (computed
        # directly from the CMSIS-NN fixed-point math) is the authoritative golden output.
        ref_q = self._quantize(ref_float, output_scale, output_zero_point, qmin, qmax, np_dtype).reshape(-1)
        reference_delta = (
            int(np.max(np.abs(ref_q.astype(np.int32) - expected_q.reshape(-1).astype(np.int32))))
            if ref_q.size
            else 0
        )

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "prefix": name,
            "num_pixels": num_pixels,
            "scalar_array": builder.format_array_as_c_literal(scalar_q.reshape(-1)),
            "alpha_array": builder.format_array_as_c_literal(alpha_q.reshape(-1)),
            "expected_output_array": builder.format_array_as_c_literal(expected_q.reshape(-1)),
            "input_dtype": dtype_info["c_type"],
            "output_dtype": dtype_info["c_type"],
            "kernel_fn": dtype_info["kernel_fn"],
            "block_size": block_size,
            "input_offset": int(input_offset),
            "alpha_offset": int(alpha_offset),
            "output_offset": int(output_offset),
            "output_mult_identity": int(output_multiplier_identity),
            "output_shift_identity": int(output_shift_identity),
            "output_mult_alpha": int(output_multiplier_alpha),
            "output_shift_alpha": int(output_shift_alpha),
            "reference_delta": int(reference_delta),
        }

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "PReLUScalar"),
            "operator_name": "prelu_scalar",
        }
        self._write_op_outputs(
            output_dir,
            "prelu_scalar",
            "ActivationFunctions/prelu_scalar/prelu_scalar.h.j2",
            "ActivationFunctions/prelu_scalar/prelu_scalar.c.j2",
            context,
            cmake_context,
        )
