"""
Shape operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpShape(OperationBase):
    """
    Shape operation - returns the shape of a tensor.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Shape operation."""
        input_shape = self.desc['input_shape']
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Shape operation - returns int32 tensor
        x = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.shape(x), tf.int32),
            name='shape'
        )(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Shape outputs int32; keep float input to avoid invalid quantized I/O without quantization
        converter.optimizations = []
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.int32
        
        def representative_data_gen():
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
        
        converter.representative_dataset = representative_data_gen
        
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for Shape.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        tflite_path = Path(output_dir) / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = tuple(input_details[0]['shape'])
        output_shape = tuple(output_details[0]['shape'])

        builder = TemplateContextBuilder()

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        self.rng.__setstate__(rng_state)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).astype(np.int32)

        context = {
            'name': name,
            'prefix': name,
            'input_data_array': builder.format_array_as_c_literal(input_data),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'output_size': int(np.prod(output_shape)),
            'shape_array': ", ".join(str(int(x)) for x in input_shape),
        }

        includes_api_dir = Path(output_dir) / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        h_content = self.render_template("shape/shape.h.j2", context)
        (includes_api_dir / f"{name}_shape.h").write_text(h_content)
        c_content = self.render_template("shape/shape.c.j2", context)
        (Path(output_dir) / f"{name}_shape.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Shape'),
            'operator_name': 'shape',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (Path(output_dir) / "CMakeLists.txt").write_text(cmake_content)
