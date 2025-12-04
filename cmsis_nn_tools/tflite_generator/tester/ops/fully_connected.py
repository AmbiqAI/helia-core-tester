"""
FullyConnected operation implementation with dtype-aware quantization.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase
import keras

class OpFullyConnected(OperationBase):
    """
    FullyConnected operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for FullyConnected operation."""
        input_shape = self.desc['input_shape']
        filter_shape = self.desc['filter_shape']
        
        # Handle both 2D [batch, features] and 4D [batch, h, w, c] input shapes
        if len(input_shape) == 2:
            # 2D input: [batch, features]
            input_features = input_shape[1]
            batch_size = input_shape[0]
        else:
            # 4D input: [batch, h, w, c]
            input_features = input_shape[3]
            batch_size = input_shape[0]
        
        # Extract output units from filter_shape
        # filter_shape can be [output_units, input_features] or [output_units, 1, 1, input_features]
        if len(filter_shape) == 2:
            output_units = filter_shape[0]
        else:
            output_units = filter_shape[0]
        
        # Get activation and use_bias from descriptor
        activation_str = self.desc.get('activation', 'NONE')
        use_bias = self.desc.get('use_bias', True)
        
        # Build model with input layer
        inputs = keras.layers.Input(shape=(input_features,), batch_size=batch_size, name='input')
        
        # Dense layer without activation (we'll apply activation separately if needed)
        x = keras.layers.Dense(output_units, activation=None, use_bias=use_bias, name='dense')(inputs)
        
        # Apply activation if specified
        if activation_str == 'RELU':
            x = keras.layers.ReLU()(x)
        elif activation_str == 'RELU6':
            x = keras.layers.ReLU(max_value=6)(x)
        elif activation_str == 'TANH':
            x = keras.layers.Activation('tanh')(x)
        elif activation_str == 'SIGMOID':
            x = keras.layers.Activation('sigmoid')(x)
        elif activation_str != 'NONE':
            raise ValueError(f"Unsupported activation: {activation_str}")
        
        model = keras.models.Model(inputs=inputs, outputs=x)
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
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        
        # Generate representative dataset
        def representative_data_gen():
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
