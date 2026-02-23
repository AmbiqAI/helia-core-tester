"""
Unpack operation implementation.
"""

from typing import Dict, Any
import tensorflow as tf
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpUnpack(OperationBase):
    """
    Unpack operation - splits a tensor along an axis.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Unpack operation."""
        input_shape = self.desc['input_shape']
        axis = self.desc.get('axis', 0)
        num_tensors = self.desc.get('num_tensors', 1)
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Adjust axis to account for batch dimension removal
        if axis >= 0:
            axis_adjusted = axis - 1 if axis > 0 else axis
        else:
            axis_adjusted = axis
        
        # Unpack operation - split along axis
        x = tf.unstack(inputs, axis=axis_adjusted, num=num_tensors)
        
        # Return all outputs to match TFLite Unpack behavior
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for Unpack.
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
        output_shapes = [tuple(d['shape']) for d in output_details]

        axis = self.desc.get('axis', 0)
        num_tensors = self.desc.get('num_tensors', len(output_details))
        rank = len(input_shape)
        if axis < 0:
            axis = axis + rank

        outer_size = int(np.prod(input_shape[:axis])) if axis > 0 else 1
        inner_size = int(np.prod(input_shape[axis + 1:])) if axis < rank - 1 else 1

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
        outputs = []
        for idx, d in enumerate(output_details[:num_tensors]):
            out_data = interpreter.get_tensor(d['index'])
            out_data = np.array(out_data)
            outputs.append({
                'name': f"{name}_out_{idx}",
                'expected_output_array': builder.format_array_as_c_literal(out_data),
                'size': int(np.prod(out_data.shape)),
            })

        context = {
            'name': name,
            'prefix': name,
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'outputs': outputs,
            'input_dtype': c_type,
            'output_dtype': c_type,
            'outer_size': outer_size,
            'inner_size': inner_size,
            'num_tensors': num_tensors,
        }

        includes_api_dir = Path(output_dir) / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        h_content = self.render_template("unpack/unpack.h.j2", context)
        (includes_api_dir / f"{name}_unpack.h").write_text(h_content)
        c_content = self.render_template("unpack/unpack.c.j2", context)
        (Path(output_dir) / f"{name}_unpack.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Unpack'),
            'operator_name': 'unpack',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (Path(output_dir) / "CMakeLists.txt").write_text(cmake_content)
