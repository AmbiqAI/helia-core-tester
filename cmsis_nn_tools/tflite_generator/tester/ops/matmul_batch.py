"""
MatMul operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase


class OpMatMul(OperationBase):
    """
    MatMul operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for MatMul operation."""
        input_1_shape = self.desc['input_1_shape']
        input_2_shape = self.desc['input_2_shape']
        
        # Get transpose options (adj_x and adj_y)
        adj_x = self.desc.get('adj_x', False)
        adj_y = self.desc.get('adj_y', False)
        
        # Build inputs (exclude batch dimension in Input shape)
        input1 = tf.keras.Input(shape=input_1_shape[1:], dtype=tf.float32, name='input1')
        input2 = tf.keras.Input(shape=input_2_shape[1:], dtype=tf.float32, name='input2')
        
        # Apply transpose if needed
        if adj_x:
            # Transpose last two dimensions: [..., M, K] -> [..., K, M]
            x1 = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(input1)
        else:
            x1 = input1
            
        if adj_y:
            # Transpose last two dimensions: [..., K, N] -> [..., N, K]
            x2 = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(input2)
        else:
            x2 = input2
        
        # Batch matrix multiplication: [batch, M, K] @ [batch, K, N] -> [batch, M, N]
        output = tf.keras.layers.Lambda(lambda inputs: tf.matmul(inputs[0], inputs[1]))([x1, x2])
        
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        
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
