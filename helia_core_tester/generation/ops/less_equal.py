"""
LessEqual comparison operation implementation.
"""

from typing import Dict, Any
import tensorflow as tf
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpLessEqual(OperationBase):
    """
    LessEqual comparison operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for LessEqual operation."""
        input_1_shape = self.desc['input_1_shape']
        input_2_shape = self.desc['input_2_shape']
        
        input1 = tf.keras.Input(shape=input_1_shape[1:], dtype=tf.float32, name='input1')
        input2 = tf.keras.Input(shape=input_2_shape[1:], dtype=tf.float32, name='input2')
        
        # LessEqual operation outputs boolean
        output = tf.keras.layers.Lambda(lambda x: tf.less_equal(x[0], x[1]))([input1, input2])
        
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        self._convert_with_activation_quantization(model, out_path, output_type=tf.bool, rep_seed=rep_seed)

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for LessEqual.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        tflite_path = Path(output_dir) / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S16':
            kernel_fn = 'arm_less_equal_s16'
            c_type = 'int16_t'
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            kernel_fn = 'arm_less_equal_s8'
            c_type = 'int8_t'
            np_in_dtype = np.int8
            qmin, qmax = -128, 127

        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape_1 = tuple(input_details[0]['shape'])
        input_shape_2 = tuple(input_details[1]['shape'])
        output_shape = tuple(output_details[0]['shape'])

        builder = TemplateContextBuilder()
        input_1_dims = builder.nhwc_to_cmsis_dims(input_shape_1)
        input_2_dims = builder.nhwc_to_cmsis_dims(input_shape_2)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        input_qp_1 = input_details[0].get('quantization_parameters', {})
        input_qp_2 = input_details[1].get('quantization_parameters', {})
        input_scale_1 = input_qp_1.get('scales', [1.0])
        input_zp_1 = input_qp_1.get('zero_points', [0])
        input_scale_2 = input_qp_2.get('scales', [1.0])
        input_zp_2 = input_qp_2.get('zero_points', [0])
        input_scale_1 = float(input_scale_1[0] if isinstance(input_scale_1, (list, np.ndarray)) else input_scale_1)
        input_zp_1 = int(input_zp_1[0] if isinstance(input_zp_1, (list, np.ndarray)) else input_zp_1)
        input_scale_2 = float(input_scale_2[0] if isinstance(input_scale_2, (list, np.ndarray)) else input_scale_2)
        input_zp_2 = int(input_zp_2[0] if isinstance(input_zp_2, (list, np.ndarray)) else input_zp_2)

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_1_f = self.rng.uniform(-1.0, 1.0, size=input_shape_1).astype(np.float32)
        input_2_f = self.rng.uniform(-1.0, 1.0, size=input_shape_2).astype(np.float32)
        self.rng.__setstate__(rng_state)

        input_1_q = np.round(input_1_f / float(input_scale_1) + float(input_zp_1)).astype(np.int32)
        input_1_q = np.clip(input_1_q, qmin, qmax).astype(np_in_dtype)
        input_2_q = np.round(input_2_f / float(input_scale_2) + float(input_zp_2)).astype(np.int32)
        input_2_q = np.clip(input_2_q, qmin, qmax).astype(np_in_dtype)

        interpreter.set_tensor(input_details[0]['index'], input_1_q)
        interpreter.set_tensor(input_details[1]['index'], input_2_q)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).astype(bool)

        context = {
            'name': name,
            'prefix': name,
            'input_1_dims': input_1_dims,
            'input_2_dims': input_2_dims,
            'output_dims': output_dims,
            'input_1_data_array': builder.format_array_as_c_literal(input_1_q),
            'input_2_data_array': builder.format_array_as_c_literal(input_2_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data.astype(np.uint8)),
            'input_dtype': c_type,
            'kernel_fn': kernel_fn,
            'output_size': int(np.prod(output_shape)),
            'input_1_offset': int(-input_zp_1),
            'input_1_mult': 1,
            'input_1_shift': 0,
            'input_2_offset': int(-input_zp_2),
            'input_2_mult': 1,
            'input_2_shift': 0,
            'left_shift': 0,
        }

        includes_api_dir = Path(output_dir) / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        h_content = self.render_template("comparison/comparison.h.j2", context)
        (includes_api_dir / f"{name}_comparison.h").write_text(h_content)
        c_content = self.render_template("comparison/comparison.c.j2", context)
        (Path(output_dir) / f"{name}_comparison.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'LessEqual'),
            'operator_name': 'comparison',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (Path(output_dir) / "CMakeLists.txt").write_text(cmake_content)
