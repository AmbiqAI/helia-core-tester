"""LSTMUnidirectional operation implementation."""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops.catalog import get_operator_spec


class OpLSTMUnidirectional(OperationBase):
    """LSTMUnidirectional operation."""
    
    def build_keras_model(self) -> tf.keras.Model:
        """
        Build Keras model for LSTMUnidirectional.
        Mirrors the flow from ns-cmsis-nn reference implementation.
        """
        # Extract parameters from descriptor
        time_steps = int(self.desc.get('time_steps', 50))
        input_size = int(self.desc.get('feature_size', 24))  # feature_size maps to input_size
        hidden_size = int(self.desc.get('units', 64))  # units maps to hidden_size
        batch_size = int(self.desc.get('batch_size', 1))
        time_major = bool(self.desc.get('time_major', False))
        
        # Build input layer with batch_size specified
        input_layer = tf.keras.Input(
            shape=(time_steps, input_size),
            batch_size=batch_size,
            dtype=tf.float32,
            name='input'
        )
        
        # Handle time_major: transpose if needed
        # Note: time_major parameter is only passed when True (for older TF versions compatibility)
        if time_major:
            input_layer_transposed = tf.transpose(input_layer, perm=[1, 0, 2])
            lstm_layer = tf.keras.layers.LSTM(
                units=hidden_size,
                time_major=True,  # Only pass when True
                return_sequences=bool(self.desc.get('return_sequences', True)),
                name="lstm"
            )(input_layer_transposed)
        else:
            # When time_major=False, don't pass the parameter (use default)
            lstm_layer = tf.keras.layers.LSTM(
                units=hidden_size,
                return_sequences=bool(self.desc.get('return_sequences', True)),
                name="lstm"
            )(input_layer)
        
        model = tf.keras.Model(input_layer, lstm_layer, name="LSTMUnidirectional")
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        if self.desc.get("hint", {}).get("force_cmsis", False):
            raise RuntimeError("LSTM CMSIS-only test; skip TFLite generation.")
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # LSTM requires Select TF ops (flex ops) to handle dynamic tensor lists
        # Disable experimental lower tensor list ops to avoid conversion errors
        converter._experimental_lower_tensor_list_ops = False
        
        # Apply quantization based on activation_dtype
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            # For LSTM: SELECT_TF_OPS is required for TensorList ops
            # However, quantization + SELECT_TF_OPS may not work together
            # Try without quantization first, or skip quantization for LSTM
            # Note: LSTM in TFLite often uses unquantized float32 due to complexity
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            # For now, skip quantization for LSTM as it conflicts with SELECT_TF_OPS
            # The model will be float32
            # TODO: Investigate if post-training quantization can work with SELECT_TF_OPS
        elif activation_dtype == 'S16':
            # Similar to S8: quantization with SELECT_TF_OPS is problematic
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            # Skip quantization for now
        else:
            # For float32 or other types, still need SELECT_TF_OPS for LSTM
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        
        # Generate representative dataset
        def representative_data_gen():
            for _ in range(100):
                if 'time_steps' in self.desc and 'feature_size' in self.desc:
                    # LSTM uses [batch, time_steps, feature_size]
                    batch_size = int(self.desc.get('batch_size', 1))
                    time_steps = int(self.desc['time_steps'])
                    feature_size = int(self.desc['feature_size'])
                    inputs = self.rng.uniform(-1.0, 1.0, size=(batch_size, time_steps, feature_size)).astype(np.float32)
                    yield [inputs]
                elif 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        # Only set representative dataset if quantization is enabled
        # (Currently disabled for LSTM when using SELECT_TF_OPS)
        if activation_dtype in ['S8', 'S16']:
            # Quantization is currently skipped for LSTM, but keep dataset ready
            # for future implementation
            converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)

    def needs_keras_model(self) -> bool:
        return not self.desc.get("hint", {}).get("force_cmsis", False)

    def allow_no_tflite(self) -> bool:
        return self.desc.get("hint", {}).get("force_cmsis", False)

    @staticmethod
    def _dataset_prefixes(dataset: str) -> tuple[str, str]:
        macro_prefix = dataset.upper() + "_"
        data_prefix = dataset.lower() + "_"
        return macro_prefix, data_prefix

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _generate_lstm_expected_f32(
        self,
        input_tensor: np.ndarray,
        forget_w_in: np.ndarray,
        input_w_in: np.ndarray,
        cell_w_in: np.ndarray,
        output_w_in: np.ndarray,
        forget_w_hidden: np.ndarray,
        input_w_hidden: np.ndarray,
        cell_w_hidden: np.ndarray,
        output_w_hidden: np.ndarray,
        forget_bias: np.ndarray,
        input_bias: np.ndarray,
        cell_bias: np.ndarray,
        output_bias: np.ndarray,
        *,
        batch_size: int,
        time_steps: int,
        input_size: int,
        hidden_size: int,
        time_major: bool,
        cell_clip: float,
    ) -> np.ndarray:
        hidden = np.zeros((batch_size, hidden_size), dtype=np.float32)
        cell = np.zeros((batch_size, hidden_size), dtype=np.float32)
        output = np.zeros((time_steps, batch_size, hidden_size), dtype=np.float32)

        if time_major:
            sequence = np.asarray(input_tensor, dtype=np.float32).reshape(time_steps, batch_size, input_size)
        else:
            sequence = np.transpose(
                np.asarray(input_tensor, dtype=np.float32).reshape(batch_size, time_steps, input_size),
                (1, 0, 2),
            )

        for t in range(time_steps):
            x = sequence[t]
            forget_gate = self._sigmoid(x @ forget_w_in.T + hidden @ forget_w_hidden.T + forget_bias)
            input_gate = self._sigmoid(x @ input_w_in.T + hidden @ input_w_hidden.T + input_bias)
            cell_gate = np.tanh(x @ cell_w_in.T + hidden @ cell_w_hidden.T + cell_bias)
            output_gate = self._sigmoid(x @ output_w_in.T + hidden @ output_w_hidden.T + output_bias)
            cell = forget_gate * cell + input_gate * cell_gate
            if cell_clip > 0.0:
                cell = np.clip(cell, -cell_clip, cell_clip)
            hidden = output_gate * np.tanh(cell)
            output[t] = hidden

        if time_major:
            return output.astype(np.float32).flatten()
        return np.transpose(output, (1, 0, 2)).astype(np.float32).flatten()

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files for CMSIS-only LSTM tests.
        """
        if (
            self.desc.get("hint", {}).get("force_cmsis", False)
            and str(self.desc.get("activation_dtype", "S8")).upper() in {"FP32", "FP16"}
        ):
            from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

            name = self.desc['name']
            activation_dtype = str(self.desc.get("activation_dtype", "S8")).upper()
            float_dtype = np.float16 if activation_dtype == "FP16" else np.float32
            data_dtype = "float16_t" if activation_dtype == "FP16" else "float"
            kernel_fn = "arm_lstm_unidirectional_f16" if activation_dtype == "FP16" else "arm_lstm_unidirectional_f32"
            lstm_params_type = "cmsis_nn_lstm_params_f16" if activation_dtype == "FP16" else "cmsis_nn_lstm_params_f32"
            lstm_context_type = "cmsis_nn_lstm_context_f16" if activation_dtype == "FP16" else "cmsis_nn_lstm_context_f32"
            batch_size = int(self.desc.get("batch_size", 1))
            time_steps = int(self.desc.get("time_steps", 1))
            input_size = int(self.desc.get("feature_size", self.desc.get("input_size", 1)))
            hidden_size = int(self.desc.get("units", self.desc.get("hidden_size", 1)))
            time_major = bool(self.desc.get("time_major", False))
            cell_clip = float(self.desc.get("cell_clip", 0.0))

            rng_state = self.rng.__getstate__()
            self.rng = np.random.default_rng(self.seed)
            if time_major:
                input_tensor = self.rng.uniform(-1.0, 1.0, size=(time_steps, batch_size, input_size)).astype(float_dtype)
            else:
                input_tensor = self.rng.uniform(-1.0, 1.0, size=(batch_size, time_steps, input_size)).astype(float_dtype)
            forget_w_in = self.rng.uniform(-0.5, 0.5, size=(hidden_size, input_size)).astype(float_dtype)
            input_w_in = self.rng.uniform(-0.5, 0.5, size=(hidden_size, input_size)).astype(float_dtype)
            cell_w_in = self.rng.uniform(-0.5, 0.5, size=(hidden_size, input_size)).astype(float_dtype)
            output_w_in = self.rng.uniform(-0.5, 0.5, size=(hidden_size, input_size)).astype(float_dtype)
            forget_w_hidden = self.rng.uniform(-0.5, 0.5, size=(hidden_size, hidden_size)).astype(float_dtype)
            input_w_hidden = self.rng.uniform(-0.5, 0.5, size=(hidden_size, hidden_size)).astype(float_dtype)
            cell_w_hidden = self.rng.uniform(-0.5, 0.5, size=(hidden_size, hidden_size)).astype(float_dtype)
            output_w_hidden = self.rng.uniform(-0.5, 0.5, size=(hidden_size, hidden_size)).astype(float_dtype)
            forget_bias = self.rng.uniform(-0.25, 0.25, size=(hidden_size,)).astype(float_dtype)
            input_bias = self.rng.uniform(-0.25, 0.25, size=(hidden_size,)).astype(float_dtype)
            cell_bias = self.rng.uniform(-0.25, 0.25, size=(hidden_size,)).astype(float_dtype)
            output_bias = self.rng.uniform(-0.25, 0.25, size=(hidden_size,)).astype(float_dtype)
            self.rng.__setstate__(rng_state)

            output_ref = self._generate_lstm_expected_f32(
                input_tensor,
                forget_w_in,
                input_w_in,
                cell_w_in,
                output_w_in,
                forget_w_hidden,
                input_w_hidden,
                cell_w_hidden,
                output_w_hidden,
                forget_bias,
                input_bias,
                cell_bias,
                output_bias,
                batch_size=batch_size,
                time_steps=time_steps,
                input_size=input_size,
                hidden_size=hidden_size,
                time_major=time_major,
                cell_clip=cell_clip,
            )

            builder = TemplateContextBuilder()
            context = {
                "name": name,
                "prefix": name,
                "data_dtype": data_dtype,
                "kernel_fn": kernel_fn,
                "lstm_params_type": lstm_params_type,
                "lstm_context_type": lstm_context_type,
                "time_major_literal": "1" if time_major else "0",
                "batch_size": batch_size,
                "time_steps": time_steps,
                "input_size": input_size,
                "hidden_size": hidden_size,
                "cell_clip_literal": builder.format_float_literal(cell_clip),
                "input_tensor_array": builder.format_array_as_c_literal(input_tensor),
                "output_array": builder.format_array_as_c_literal(output_ref.astype(float_dtype)),
                "forget_gate_input_weights_array": builder.format_array_as_c_literal(forget_w_in),
                "input_gate_input_weights_array": builder.format_array_as_c_literal(input_w_in),
                "cell_gate_input_weights_array": builder.format_array_as_c_literal(cell_w_in),
                "output_gate_input_weights_array": builder.format_array_as_c_literal(output_w_in),
                "forget_gate_hidden_weights_array": builder.format_array_as_c_literal(forget_w_hidden),
                "input_gate_hidden_weights_array": builder.format_array_as_c_literal(input_w_hidden),
                "cell_gate_hidden_weights_array": builder.format_array_as_c_literal(cell_w_hidden),
                "output_gate_hidden_weights_array": builder.format_array_as_c_literal(output_w_hidden),
                "forget_gate_bias_array": builder.format_array_as_c_literal(forget_bias),
                "input_gate_bias_array": builder.format_array_as_c_literal(input_bias),
                "cell_gate_bias_array": builder.format_array_as_c_literal(cell_bias),
                "output_gate_bias_array": builder.format_array_as_c_literal(output_bias),
                "cell_state_size": batch_size * hidden_size,
                "dst_size": batch_size * time_steps * hidden_size,
            }

            self._write_op_outputs(
                Path(output_dir),
                "lstm_unidirectional",
                "LSTMFunctions/lstm_unidirectional/lstm_unidirectional_f32.h.j2",
                "LSTMFunctions/lstm_unidirectional/lstm_unidirectional_f32.c.j2",
                context,
                {
                    "name": name,
                    "operator": self.desc.get("operator", "LSTMUnidirectional"),
                    "operator_name": "lstm_unidirectional",
                },
            )
            return

        if not self.desc.get("hint", {}).get("force_cmsis", False):
            return

        from pathlib import Path
        from helia_core_tester.generation.utils.lstm_data import generate_lstm_data, build_lstm_context
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.core.discovery import find_tester_templates_dir

        name = self.desc['name']
        dataset = self.desc.get("dataset")
        if not dataset:
            raise ValueError("LSTM CMSIS test requires 'dataset' in descriptor.")

        activation_dtype = self.desc.get("activation_dtype", "S8").upper()
        batch_size = int(self.desc.get("batch_size", 1))
        time_steps = int(self.desc.get("time_steps", 1))
        input_size = int(self.desc.get("feature_size", self.desc.get("input_size", 1)))
        hidden_size = int(self.desc.get("units", self.desc.get("hidden_size", 1)))
        time_major = bool(self.desc.get("time_major", False))
        extras = self.desc.get("hint", {}).get("extras", {})
        input_zero_point_override = extras.get("input_zero_point")
        output_zero_point_override = extras.get("output_zero_point")
        if input_zero_point_override is not None:
            input_zero_point_override = int(input_zero_point_override)
        if output_zero_point_override is not None:
            output_zero_point_override = int(output_zero_point_override)

        templates_dir = Path(find_tester_templates_dir()) / get_operator_spec("LSTMUnidirectional").template_relpath / "json"
        schema_path = Path(__file__).resolve().parents[4] / "UnitTest" / "RefactoredTestGen" / "schema.fbs"
        work_dir = Path(output_dir) / "_lstm_tmp"
        work_dir.mkdir(parents=True, exist_ok=True)

        data = generate_lstm_data(
            rng=self.rng,
            activation_dtype=activation_dtype,
            batch_size=batch_size,
            time_steps=time_steps,
            input_size=input_size,
            hidden_size=hidden_size,
            time_major=time_major,
            templates_dir=templates_dir,
            schema_path=schema_path,
            work_dir=work_dir,
            dataset=dataset,
            input_zero_point_override=input_zero_point_override,
            output_zero_point_override=output_zero_point_override,
        )

        context = build_lstm_context(
            name=name,
            dataset=dataset,
            activation_dtype=activation_dtype,
            batch_size=batch_size,
            time_steps=time_steps,
            input_size=input_size,
            hidden_size=hidden_size,
            time_major=time_major,
            data=data,
        )

        builder = TemplateContextBuilder()

        def fmt(arr, *, dtype=None):
            if dtype is not None:
                return builder.format_array_as_c_literal(np.asarray(arr, dtype=dtype))
            return builder.format_array_as_c_literal(np.asarray(arr))

        input_dtype = "int16_t" if activation_dtype == "S16" else "int8_t"
        output_dtype = "int16_t" if activation_dtype == "S16" else "int8_t"
        bias_dtype = "int64_t" if activation_dtype == "S16" else "int32_t"
        weight_dtype = "int8_t"

        bias_cast = np.int64 if activation_dtype == "S16" else None
        context.update({
            "time_major_literal": "true" if time_major else "false",
            "input_dtype": input_dtype,
            "output_dtype": output_dtype,
            "bias_dtype": bias_dtype,
            "weight_dtype": weight_dtype,
            "input_tensor_array": fmt(data.tensors["input_tensor"]),
            "output_array": fmt(data.tensors["output"]),
            "input_gate_input_weights_array": fmt(data.tensors["input_gate_input_weights"]),
            "forget_gate_input_weights_array": fmt(data.tensors["forget_gate_input_weights"]),
            "cell_gate_input_weights_array": fmt(data.tensors["cell_gate_input_weights"]),
            "output_gate_input_weights_array": fmt(data.tensors["output_gate_input_weights"]),
            "input_gate_hidden_weights_array": fmt(data.tensors["input_gate_hidden_weights"]),
            "forget_gate_hidden_weights_array": fmt(data.tensors["forget_gate_hidden_weights"]),
            "cell_gate_hidden_weights_array": fmt(data.tensors["cell_gate_hidden_weights"]),
            "output_gate_hidden_weights_array": fmt(data.tensors["output_gate_hidden_weights"]),
            "input_gate_bias_array": fmt(data.tensors["input_gate_bias"], dtype=bias_cast),
            "forget_gate_bias_array": fmt(data.tensors["forget_gate_bias"], dtype=bias_cast),
            "cell_gate_bias_array": fmt(data.tensors["cell_gate_bias"], dtype=bias_cast),
            "output_gate_bias_array": fmt(data.tensors["output_gate_bias"], dtype=bias_cast),
        })

        output_dir = Path(output_dir)
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("LSTMFunctions/lstm_unidirectional/lstm_unidirectional.h.j2", context)
        (includes_api_dir / f"{name}_lstm_unidirectional.h").write_text(h_content)
        c_content = self.render_template("LSTMFunctions/lstm_unidirectional/lstm_unidirectional.c.j2", context)
        (output_dir / f"{name}_lstm_unidirectional.c").write_text(c_content)

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "LSTMUnidirectional"),
            "operator_name": "lstm_unidirectional",
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
