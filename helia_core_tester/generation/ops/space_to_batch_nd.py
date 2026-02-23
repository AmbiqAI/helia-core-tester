"""
SpaceToBatchND operation implementation.
"""

from typing import Dict, Any
import tensorflow as tf
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpSpaceToBatchND(OperationBase):
    """
    SpaceToBatchND operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for SpaceToBatchND operation."""
        input_shape = self.desc['input_shape']
        block_shape = self.desc.get('block_shape', [2, 2])
        paddings = self.desc.get('paddings', [[0, 0], [0, 0]])
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # SpaceToBatchND operation (use raw op for compatibility)
        def _s2b(x):
            block_shape_const = tf.constant(block_shape, dtype=tf.int32)
            paddings_const = tf.constant(paddings, dtype=tf.int32)
            if hasattr(tf.raw_ops, "SpaceToBatchND"):
                return tf.raw_ops.SpaceToBatchND(
                    input=x,
                    block_shape=block_shape_const,
                    paddings=paddings_const,
                )
            if hasattr(tf.nn, "space_to_batch_nd"):
                return tf.nn.space_to_batch_nd(
                    x,
                    block_shape=block_shape_const,
                    paddings=paddings_const,
                )
            if hasattr(tf.compat.v1, "space_to_batch_nd"):
                return tf.compat.v1.space_to_batch_nd(
                    x,
                    block_shape=block_shape_const,
                    paddings=paddings_const,
                )
            raise AttributeError("SpaceToBatchND op not available in this TensorFlow build")

        x = tf.keras.layers.Lambda(_s2b)(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for SpaceToBatchND.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        tflite_path = Path(output_dir) / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S16':
            kernel_fn = 'arm_space_to_batch_nd_s16'
            c_type = 'int16_t'
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            kernel_fn = 'arm_space_to_batch_nd_s8'
            c_type = 'int8_t'
            np_in_dtype = np.int8
            qmin, qmax = -128, 127

        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = tuple(input_details[0]['shape'])
        output_shape = tuple(output_details[0]['shape'])

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        input_qp = input_details[0].get('quantization_parameters', {})
        input_scale = input_qp.get('scales', [1.0])
        input_zp = input_qp.get('zero_points', [0])
        if isinstance(input_scale, (list, np.ndarray)):
            input_scale = input_scale[0] if len(input_scale) > 0 else 1.0
        if isinstance(input_zp, (list, np.ndarray)):
            input_zp = input_zp[0] if len(input_zp) > 0 else 0
        input_scale = float(input_scale)
        input_zp = int(input_zp)

        output_qp = output_details[0].get('quantization_parameters', {})
        output_zp = output_qp.get('zero_points', [0])
        if isinstance(output_zp, (list, np.ndarray)):
            output_zp = output_zp[0] if len(output_zp) > 0 else 0
        output_zp = int(output_zp)

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        self.rng.__setstate__(rng_state)

        input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)

        interpreter.set_tensor(input_details[0]['index'], input_q)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)

        paddings = self.desc.get('paddings', [[0, 0], [0, 0]])
        paddings_flat = [int(paddings[0][0]), int(paddings[1][0]), int(paddings[0][1]), int(paddings[1][1])]
        block_shape = self.desc.get('block_shape', [1, 1])

        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'block_shape': block_shape,
            'paddings': paddings_flat,
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': c_type,
            'output_dtype': c_type,
            'kernel_fn': kernel_fn,
            'output_size': int(np.prod(output_shape)),
            'output_zero_point': int(output_zp),
        }

        includes_api_dir = Path(output_dir) / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("space_to_batch_nd/space_to_batch_nd.h.j2", context)
        (includes_api_dir / f"{name}_space_to_batch_nd.h").write_text(h_content)
        c_content = self.render_template("space_to_batch_nd/space_to_batch_nd.c.j2", context)
        (Path(output_dir) / f"{name}_space_to_batch_nd.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'SpaceToBatchND'),
            'operator_name': 'space_to_batch_nd',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (Path(output_dir) / "CMakeLists.txt").write_text(cmake_content)
