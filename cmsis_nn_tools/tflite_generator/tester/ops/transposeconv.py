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
        """Build Keras model for TransposeConv operation.
        """
        input_shape = self.desc['input_shape']
        filter_shape = self.desc['filter_shape']
        
        # Build input layer (exclude batch dimension in Input shape)
        inputs = tf.keras.Input(
            shape=input_shape[1:],
            batch_size=input_shape[0] if len(input_shape) > 0 else None,
            dtype=tf.float32,
            name='input'
        )
        
        # TransposeConv layer
        # filter_shape is [KH, KW, OutCh, InCh] in TensorFlow format
        transpose_conv_kwargs = {
            'filters': filter_shape[2],  # Number of output channels (third dimension)
            'kernel_size': filter_shape[0:2],  # Kernel height and width (first two dimensions)
            'strides': tuple(self.desc.get('strides', [1, 1])),
            'padding': str(self.desc.get('padding', 'valid')).lower(),
            'use_bias': self.desc.get('use_bias', True),
            'name': 'transpose_conv'
        }
    
        transpose_conv_kwargs['kernel_initializer'] = tf.keras.initializers.GlorotUniform(seed=123)
        
        if transpose_conv_kwargs['use_bias']:
            transpose_conv_kwargs['bias_initializer'] = tf.keras.initializers.RandomUniform(
                minval=-0.5, maxval=0.5, seed=321
            )
        
        layer = tf.keras.layers.Conv2DTranspose(**transpose_conv_kwargs)
        outputs = layer(inputs)
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name='transpose_conv_model')
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
            rep_rng = np.random.default_rng(42)
            for _ in range(128): 
                if 'input_shape' in self.desc:
                    inputs = rep_rng.uniform(-8.0, 8.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = rep_rng.uniform(-8.0, 8.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = rep_rng.uniform(-8.0, 8.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
