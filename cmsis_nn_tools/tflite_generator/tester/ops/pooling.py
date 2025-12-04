"""
Pooling operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase


class OpPooling(OperationBase):
    """
    Pooling operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Pooling operation."""
        input_shape = self.desc['input_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Determine pooling type
        pooling_type = self.desc.get('pooling_type', 'AVERAGE')
        
        # Normalize padding to lowercase
        padding = self.desc.get('padding', 'valid')
        if isinstance(padding, str):
            padding = padding.lower()
        
        if pooling_type == 'AVERAGE':
            x = tf.keras.layers.AveragePooling2D(
                pool_size=self.desc.get('pool_size', [2, 2]),
                strides=self.desc.get('strides', [2, 2]),
                padding=padding
            )(inputs)
        elif pooling_type == 'MAX':
            x = tf.keras.layers.MaxPooling2D(
                pool_size=self.desc.get('pool_size', [2, 2]),
                strides=self.desc.get('strides', [2, 2]),
                padding=padding
            )(inputs)
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")
            
        model = tf.keras.Model(inputs=inputs, outputs=x)
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
            converter.inference_input_type  = tf.int16
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
