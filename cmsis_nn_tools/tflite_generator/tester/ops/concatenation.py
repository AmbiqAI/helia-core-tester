"""
Concatenation operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase


class OpConcatenation(OperationBase):
    """
    Concatenation operation - concatenates tensors along an axis.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Concatenation operation."""
        # Concatenation requires multiple inputs
        input_shapes = []
        if 'input_1_shape' in self.desc:
            # Multiple named inputs
            i = 1
            while f'input_{i}_shape' in self.desc:
                input_shapes.append(self.desc[f'input_{i}_shape'])
                i += 1
        else:
            # Single input shape
            input_shapes = [self.desc.get('input_shape')]
        
        axis = self.desc.get('axis', -1)
        
        # Create input layers
        inputs = []
        for i, shape in enumerate(input_shapes):
            inp = tf.keras.Input(shape=shape[1:], dtype=tf.float32, name=f'input{i+1}')
            inputs.append(inp)
        
        # Adjust axis to account for batch dimension
        if axis >= 0:
            axis_adjusted = axis - 1 if axis > 0 else axis
        else:
            axis_adjusted = axis
        
        # Concatenation operation
        x = tf.keras.layers.Concatenate(axis=axis_adjusted)(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        import tensorflow as tf
        import numpy as np
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
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
        
        def representative_data_gen():
            for _ in range(100):
                inputs_list = []
                i = 1
                while f'input_{i}_shape' in self.desc:
                    shape = self.desc[f'input_{i}_shape']
                    inputs_list.append(self.rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))
                    i += 1
                if not inputs_list and 'input_shape' in self.desc:
                    inputs_list.append(self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32))
                yield inputs_list
        
        converter.representative_dataset = representative_data_gen
        
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)

