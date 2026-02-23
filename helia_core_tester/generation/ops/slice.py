"""
Slice operation implementation.
"""

from typing import Dict, Any
import tensorflow as tf
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpSlice(OperationBase):
    """
    Slice operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Slice operation."""
        input_shape = self.desc['input_shape']
        begin = self.desc.get('begin')
        size = self.desc.get('size')
        
        if begin is None or size is None:
            raise ValueError("Slice operation requires 'begin' and 'size' in descriptor")
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Slice operation
        def slice_op(x):
            return tf.slice(x, begin[1:], size[1:])  # Remove batch dimension
        
        x = tf.keras.layers.Lambda(slice_op, name='slice')(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for Slice.
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

        qp = input_details[0].get('quantization_parameters', {})
        scale = qp.get('scales', [1.0])
        zp = qp.get('zero_points', [0])
        scale = float(scale[0] if isinstance(scale, (list, np.ndarray)) else scale)
        zp = int(zp[0] if isinstance(zp, (list, np.ndarray)) else zp)
        dtype = input_details[0]['dtype']
        if dtype == np.int16:
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
            c_type = 'int16_t'
        else:
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
            c_type = 'int8_t'

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_f = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        self.rng.__setstate__(rng_state)

        input_q = np.round(input_f / float(scale) + float(zp)).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)

        interpreter.set_tensor(input_details[0]['index'], input_q)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)

        begin = self.desc.get('begin')
        size = self.desc.get('size')
        rank = len(input_shape)
        if begin is None or size is None:
            raise ValueError("Slice descriptor missing begin/size")
        resolved_size = []
        for i, s in enumerate(size):
            if int(s) == -1:
                resolved_size.append(int(input_shape[i]) - int(begin[i]))
            else:
                resolved_size.append(int(s))

        context = {
            'name': name,
            'prefix': name,
            'input_shape_array': ", ".join(str(int(x)) for x in input_shape),
            'begin_array': ", ".join(str(int(x)) for x in begin),
            'size_array': ", ".join(str(int(x)) for x in resolved_size),
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': c_type,
            'output_dtype': c_type,
            'output_size': int(np.prod(output_shape)),
            'rank': rank,
            'max_rank': rank,
        }

        includes_api_dir = Path(output_dir) / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        h_content = self.render_template("slice/slice.h.j2", context)
        (includes_api_dir / f"{name}_slice.h").write_text(h_content)
        c_content = self.render_template("slice/slice.c.j2", context)
        (Path(output_dir) / f"{name}_slice.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Slice'),
            'operator_name': 'slice',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (Path(output_dir) / "CMakeLists.txt").write_text(cmake_content)
