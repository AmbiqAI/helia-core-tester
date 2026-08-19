"""Float activation operation implementation."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from helia_core_tester.core.cpu_targets import get_cpu_profile
from helia_core_tester.generation.ops._shared.base import OperationBase


_ACTIVATION_LAYERS = {
    "ARM_NN_FLT_ACT_SIGMOID": tf.keras.activations.sigmoid,
    "ARM_NN_FLT_ACT_TANH": tf.keras.activations.tanh,
    "ARM_NN_FLT_ACT_NONE": tf.keras.activations.linear,
}


def _fp16(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float16)


# float16 tanh LUT sampled over x in [0, 4] with 256 intervals, matching
# arm_nn_tanh_lut256_f16 in Source/NNSupportFunctions/arm_nntables_flt.c:
#   arm_nn_tanh_lut256_f16[i] = float16(tanh(4 * i / 256))
_TANH_LUT256_F16 = np.tanh(4.0 * np.arange(257, dtype=np.float64) / 256.0).astype(np.float16)


def _tanh_reference_f16(input_data: np.ndarray) -> np.ndarray:
    """Mirror the scalar FP16 tanh fallback used when MVE intrinsics are disabled."""
    data = _fp16(input_data)
    x2 = _fp16(_fp16(data) * _fp16(data))
    num = _fp16(_fp16(data) * _fp16(_fp16(27.0) + x2))
    den = _fp16(_fp16(27.0) + _fp16(_fp16(9.0) * x2))
    approx = _fp16(num / den)

    saturated = np.where(data < _fp16(0.0), _fp16(-1.0), _fp16(1.0))
    return np.where(np.abs(data) > _fp16(3.0), saturated, approx).astype(np.float16)


def _tanh_reference_f16_mve(input_data: np.ndarray) -> np.ndarray:
    """Mirror the float16 MVE tanh path (arm_nn_vtanh_lut_direct_mve_f16).

    Reproduces the Helium kernel's LUT + linear interpolation and float16 rounding
    so generated expectations match Cortex-M55 MVE output bit-for-bit.
    """
    x = _fp16(input_data)
    ax = _fp16(np.abs(x))
    saturate = ax > _fp16(4.0)
    ax = _fp16(np.minimum(ax, _fp16(4.0)))
    # t = ax * (256 / 4); idx = floor(t) clamped to [0, 255]; frac = t - idx
    t = _fp16(ax * _fp16(64.0))
    idx = np.minimum(t.astype(np.uint16), np.uint16(255))
    frac = _fp16(t - idx.astype(np.float16))
    y0 = _TANH_LUT256_F16[idx]
    y1 = _TANH_LUT256_F16[idx + 1]
    diff = _fp16(y1 - y0)
    # vfmaq performs a single-rounded multiply-add; evaluate exactly then round once.
    interp = _fp16(y0.astype(np.float64) + diff.astype(np.float64) * frac.astype(np.float64))
    magnitude = np.where(saturate, _fp16(1.0), interp)
    result = np.where(x < _fp16(0.0), _fp16(-magnitude), magnitude)
    return result.astype(np.float16)


def _activation_reference(
    input_data: np.ndarray,
    activation_type: str,
    act_param: float,
    activation_dtype: str = "FP32",
    *,
    use_mve_tanh: bool = False,
) -> np.ndarray:
    if activation_type == "ARM_NN_FLT_ACT_TANH" and str(activation_dtype).upper() == "FP16":
        if use_mve_tanh:
            return _tanh_reference_f16_mve(input_data)
        return _tanh_reference_f16(input_data)

    data = input_data.astype(np.float32)
    if activation_type == "ARM_NN_FLT_ACT_SIGMOID":
        return 1.0 / (1.0 + np.exp(-data))
    if activation_type == "ARM_NN_FLT_ACT_TANH":
        return np.tanh(data)
    if activation_type == "ARM_NN_FLT_ACT_HARDSWISH":
        return data * np.clip(data + 3.0, 0.0, 6.0) / 6.0
    if activation_type == "ARM_NN_FLT_ACT_LEAKY_RELU":
        return np.where(data >= 0.0, data, data * float(act_param))
    if activation_type == "ARM_NN_FLT_ACT_RELU":
        return np.maximum(data, 0.0)
    if activation_type == "ARM_NN_FLT_ACT_RELU6":
        return np.clip(data, 0.0, 6.0)
    if activation_type == "ARM_NN_FLT_ACT_NONE":
        return data
    raise ValueError(f"Unsupported float activation type: {activation_type}")


class OpNNActivationFloat(OperationBase):
    """Generate float activation parity tests."""

    def build_keras_model(self) -> tf.keras.Model:
        input_shape = tuple(self.desc["input_shape"])
        activation_type = str(self.desc["activation_type"]).upper()
        act_param = float(self.desc.get("act_param", 0.0))

        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name="input")
        if activation_type in _ACTIVATION_LAYERS:
            output = tf.keras.layers.Activation(_ACTIVATION_LAYERS[activation_type])(inputs)
        elif activation_type == "ARM_NN_FLT_ACT_HARDSWISH":
            output = tf.keras.layers.Lambda(lambda x: x * tf.nn.relu6(x + 3.0) / 6.0)(inputs)
        elif activation_type == "ARM_NN_FLT_ACT_LEAKY_RELU":
            output = tf.keras.layers.LeakyReLU(negative_slope=act_param)(inputs)
        elif activation_type == "ARM_NN_FLT_ACT_RELU":
            output = tf.keras.layers.Lambda(tf.nn.relu)(inputs)
        elif activation_type == "ARM_NN_FLT_ACT_RELU6":
            output = tf.keras.layers.Lambda(tf.nn.relu6)(inputs)
        else:
            raise ValueError(f"Unsupported float activation type: {activation_type}")

        return tf.keras.Model(inputs=inputs, outputs=output)

    def _activation_symbol(self) -> str:
        return str(self.desc["activation_type"]).upper()

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        input_shape = tuple(self.desc["input_shape"])
        activation_dtype = self.tensor_dtype("input", default="FP32")
        if activation_dtype == "FP16":
            float_dtype = np.float16
            input_dtype = output_dtype = "float16_t"
            kernel_fn = "arm_nn_activation_f16"
        elif activation_dtype == "FP32":
            float_dtype = np.float32
            input_dtype = output_dtype = "float"
            kernel_fn = "arm_nn_activation_f32"
        else:
            raise NotImplementedError(f"Unsupported NNActivationFloat dtype: {activation_dtype}")

        input_data = self._sample_uniform(
            input_shape,
            low=float(self.desc.get("input_min", -1.0)),
            high=float(self.desc.get("input_max", 1.0)),
            dtype=float_dtype,
        )
        activation_type = self._activation_symbol()
        # On MVE targets the float16 tanh kernel takes the LUT-based Helium path
        # (arm_nn_vtanh_lut_direct_mve_f16), which differs from the scalar rational
        # fallback used on non-MVE targets; mirror the path the device will execute.
        use_mve_tanh = get_cpu_profile(self.target_cpu).has_mve
        output_data = _activation_reference(
            input_data,
            activation_type,
            float(self.desc.get("act_param", 0.0)),
            activation_dtype,
            use_mve_tanh=use_mve_tanh,
        ).astype(float_dtype)

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "size": int(np.prod(input_shape)),
            "input_data_array": builder.format_array_as_c_literal(input_data),
            "expected_output_array": builder.format_array_as_c_literal(output_data),
            "activation_symbol": activation_type,
            "act_param_literal": builder.format_float_literal(self.desc.get("act_param", 0.0)),
            "input_dtype": input_dtype,
            "output_dtype": output_dtype,
            "kernel_fn": kernel_fn,
        }

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "NNActivationFloat"),
            "operator_name": "nn_activation_float",
        }
        self._write_op_outputs(
            output_dir,
            "nn_activation_float",
            "ActivationFunctions/nn_activation_float/nn_activation_float.h.j2",
            "ActivationFunctions/nn_activation_float/nn_activation_float.c.j2",
            context,
            cmake_context,
        )
