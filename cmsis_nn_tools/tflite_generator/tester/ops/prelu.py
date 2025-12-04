"""
PReLU operation implementation.
"""

from typing import Dict, Any, Iterable
import numpy as np
import tensorflow as tf
from .base import OperationBase


class OpPReLU(OperationBase):
    """
    PReLU (Parametric ReLU) operation.
    """
    
    def _prepare_alpha_values(self, input_shape: tuple, alpha_values: Iterable[float] | None = None) -> np.ndarray:
        """Prepare alpha values for PReLU layer."""
        value_shape = input_shape[1:]  # Remove batch dimension
        if not value_shape:
            raise ValueError("Input shape must include at least one non-batch dimension for PReLU.")
        num_values = int(np.prod(value_shape))
        
        if alpha_values is None:
            # Default: linear spacing from 0.05 to 0.25
            data = np.linspace(0.05, 0.25, num=num_values, dtype=np.float32)
        else:
            data = np.asarray(alpha_values, dtype=np.float32)
            if data.size != num_values:
                raise ValueError(
                    f"alpha_values has {data.size} entries, but expected {num_values} "
                    f"to match input shape {value_shape}."
                )
        return data.reshape(value_shape)
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for PReLU operation."""
        input_shape = self.desc['input_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Get alpha values from hint if provided, otherwise use default
        alpha_values = None
        if 'hint' in self.desc and 'extras' in self.desc.get('hint', {}):
            extras = self.desc['hint']['extras']
            if 'alpha_values' in extras:
                # Flatten the nested list if present
                alpha_list = extras['alpha_values']
                if isinstance(alpha_list, list) and len(alpha_list) > 0:
                    if isinstance(alpha_list[0], list):
                        # Flatten nested list
                        alpha_values = [item for sublist in alpha_list for item in sublist]
                    else:
                        alpha_values = alpha_list
        
        alpha_array = self._prepare_alpha_values(tuple(input_shape), alpha_values)
        
        # PReLU operation with per-element alpha
        prelu_layer = tf.keras.layers.PReLU(
            alpha_initializer=tf.keras.initializers.Constant(alpha_array),
            shared_axes=None,  # Per-element alpha
            name='prelu'
        )
        output = prelu_layer(inputs)
            
        model = tf.keras.Model(inputs=[inputs], outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        import tensorflow as tf
        import numpy as np
        
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

