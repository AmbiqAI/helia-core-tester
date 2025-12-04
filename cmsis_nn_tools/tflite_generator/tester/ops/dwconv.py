"""
DepthwiseConv2D operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase


class OpDepthwiseConv2D(OperationBase):
    """
    DepthwiseConv2D operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for DepthwiseConv2D operation."""
        input_shape = self.desc['input_shape']
        filter_shape = self.desc['filter_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # DepthwiseConv2D layer with random bias initialization
        dwconv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=filter_shape[1:3],
            strides=self.desc.get('strides', [1, 1]),
            padding=self.desc.get('padding', 'valid'),
            use_bias=True,
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
            name='depthwise_conv2d'
        )
        x = dwconv(inputs)
        
        # Apply activation if specified
        activation = self.desc.get('activation', 'NONE')
        if activation == 'RELU':
            x = tf.keras.layers.ReLU()(x)
        elif activation == 'RELU6':
            x = tf.keras.layers.ReLU(max_value=6)(x)
        elif activation == 'TANH':
            x = tf.keras.layers.Activation('tanh')(x)
        elif activation == 'SIGMOID':
            x = tf.keras.layers.Activation('sigmoid')(x)
        elif activation != 'NONE':
            raise ValueError(f"Unsupported activation: {activation}")
            
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
