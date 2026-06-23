"""Float activation operation implementation."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from helia_core_tester.generation.ops._shared.base import OperationBase


_ACTIVATION_LAYERS = {
    "ARM_NN_FLT_ACT_SIGMOID": tf.keras.activations.sigmoid,
    "ARM_NN_FLT_ACT_TANH": tf.keras.activations.tanh,
}


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
            output = tf.keras.layers.Lambda(tf.nn.hard_swish)(inputs)
        elif activation_type == "ARM_NN_FLT_ACT_LEAKY_RELU":
            output = tf.keras.layers.LeakyReLU(negative_slope=act_param)(inputs)
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
        input_data = self._sample_uniform(input_shape)

        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = np.array(interpreter.get_tensor(output_details[0]["index"]), dtype=np.float32)

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "prefix": name,
            "size": int(np.prod(input_shape)),
            "input_data_array": builder.format_array_as_c_literal(input_data),
            "expected_output_array": builder.format_array_as_c_literal(output_data),
            "activation_symbol": self._activation_symbol(),
            "act_param_literal": builder.format_float_literal(self.desc.get("act_param", 0.0)),
            "input_dtype": "float",
            "output_dtype": "float",
            "kernel_fn": "arm_nn_activation_f32",
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
