"""
Pack operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpPack(OperationBase):
    """
    Pack operation - stacks tensors along a new axis.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Pack operation."""
        # Pack requires multiple inputs
        input_shapes = []
        if 'input_1_shape' in self.desc:
            # Multiple named inputs
            i = 1
            while f'input_{i}_shape' in self.desc:
                input_shapes.append(self.desc[f'input_{i}_shape'])
                i += 1
        else:
            # Single input shape (shouldn't happen for Pack, but handle it)
            input_shapes = [self.desc.get('input_shape')]
        
        axis = self.desc.get('axis', 0)
        num_tensors = self.desc.get('num_tensors', len(input_shapes))
        
        # Create input layers
        inputs = []
        for i, shape in enumerate(input_shapes[:num_tensors]):
            inp = tf.keras.Input(shape=shape[1:], dtype=tf.float32, name=f'input{i+1}')
            inputs.append(inp)
        
        # Pack operation - stack along new axis
        # Adjust axis: if axis >= 0, add 1 to account for new batch dimension
        if axis >= 0:
            axis_adjusted = axis + 1
        else:
            axis_adjusted = axis
        
        x = tf.stack(inputs, axis=axis_adjusted)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
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

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for Pack.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        tflite_path = Path(output_dir) / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        num_tensors = self.desc.get('num_tensors', len(input_details))
        axis = self.desc.get('axis', 0)

        input_shapes = [tuple(d['shape']) for d in input_details[:num_tensors]]
        output_shape = tuple(output_details[0]['shape'])

        rank = len(input_shapes[0])
        if axis < 0:
            axis = axis + rank + 1

        outer_size = int(np.prod(input_shapes[0][:axis])) if axis > 0 else 1
        inner_size = int(np.prod(input_shapes[0][axis:])) if axis < rank else 1

        builder = TemplateContextBuilder()

        input_arrays = []
        for d in input_details[:num_tensors]:
            qp = d.get('quantization_parameters', {})
            scale = qp.get('scales', [1.0])
            zp = qp.get('zero_points', [0])
            scale = float(scale[0] if isinstance(scale, (list, np.ndarray)) else scale)
            zp = int(zp[0] if isinstance(zp, (list, np.ndarray)) else zp)
            dtype = d['dtype']
            if dtype == np.int16:
                np_in_dtype = np.int16
                qmin, qmax = -32768, 32767
                c_type = 'int16_t'
            else:
                np_in_dtype = np.int8
                qmin, qmax = -128, 127
                c_type = 'int8_t'

            rng_state = self.rng.__getstate__()
            self.rng = np.random.default_rng(self.seed + len(input_arrays))
            input_f = self.rng.uniform(-1.0, 1.0, size=d['shape']).astype(np.float32)
            self.rng.__setstate__(rng_state)

            input_q = np.round(input_f / float(scale) + float(zp)).astype(np.int32)
            input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
            input_arrays.append(input_q)

        for i, d in enumerate(input_details[:num_tensors]):
            interpreter.set_tensor(d['index'], input_arrays[i])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)

        context = {
            'name': name,
            'prefix': name,
            'input_1_data_array': builder.format_array_as_c_literal(input_arrays[0]),
            'input_data_arrays': [builder.format_array_as_c_literal(arr) for arr in input_arrays],
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': c_type,
            'output_dtype': c_type,
            'num_tensors': num_tensors,
            'outer_size': outer_size,
            'inner_size': inner_size,
            'output_size': int(np.prod(output_shape)),
        }

        includes_api_dir = Path(output_dir) / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        h_content = self.render_template("pack/pack.h.j2", context)
        (includes_api_dir / f"{name}_pack.h").write_text(h_content)
        c_content = self.render_template("pack/pack.c.j2", context)
        (Path(output_dir) / f"{name}_pack.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Pack'),
            'operator_name': 'pack',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (Path(output_dir) / "CMakeLists.txt").write_text(cmake_content)
