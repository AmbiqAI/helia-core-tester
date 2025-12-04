"""
TransposeConv operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase


class OpTransposeConv(OperationBase):
    """
    TransposeConv operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for TransposeConv operation."""
        input_shape = self.desc['input_shape']
        filter_shape = self.desc['filter_shape']
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # TransposeConv layer
        # filter_shape is [KH, KW, OutCh, InCh] in TensorFlow format
        transpose_conv_kwargs = {
            'filters': filter_shape[2],  # Number of output channels (third dimension)
            'kernel_size': filter_shape[0:2],  # Kernel height and width (first two dimensions)
            'strides': self.desc.get('strides', [1, 1]),
            'padding': self.desc.get('padding', 'valid').lower(),
            'use_bias': self.desc.get('use_bias', True)
        }
        
        # Add bias initializer only if use_bias is True
        if transpose_conv_kwargs['use_bias']:
            transpose_conv_kwargs['bias_initializer'] = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
        
        x = tf.keras.layers.Conv2DTranspose(**transpose_conv_kwargs)(inputs)
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
            converter.target_spec.supported_types = [tf.int16]
            # For int16 quantization, keep input/output as float32
            # For int16 quantization, keep input/output as float32
        
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
