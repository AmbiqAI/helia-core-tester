"""
Transpose operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpTranspose(OperationBase):
    """
    Transpose operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Transpose operation."""
        input_shape = self.desc['input_shape']
        perm = self.desc.get('hint', {}).get('force_permutation')
        if perm is None:
            rank = len(input_shape)
            if rank == 4:
                perm = [0, 2, 1, 3]
            elif rank == 3:
                perm = [0, 2, 1]
            elif rank == 2:
                perm = [1, 0]
            elif rank == 1:
                perm = [0]
            else:
                raise ValueError(f"Unsupported input rank for Transpose: {rank}")
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        x = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=perm))(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def _select_cmsis_transpose_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Transpose operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_transpose_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_transpose_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Transpose dtype: {activation_dtype}")

    def needs_keras_model(self) -> bool:
        if self.desc.get("hint", {}).get("force_cmsis", False):
            return False
        if self.desc.get("expected_status") == "ARM_CMSIS_NN_ARG_ERROR":
            return False
        return True

    def allow_no_tflite(self) -> bool:
        if self.desc.get("hint", {}).get("force_cmsis", False):
            return True
        if self.desc.get("expected_status") == "ARM_CMSIS_NN_ARG_ERROR":
            return True
        return False

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        if self.desc.get("hint", {}).get("force_cmsis", False):
            raise RuntimeError("Transpose CMSIS-only test; skip TFLite generation.")
        if self.desc.get("expected_status") == "ARM_CMSIS_NN_ARG_ERROR":
            raise RuntimeError("Transpose expected error; skip TFLite generation.")
        super().convert_to_tflite(model, out_path, rep_seed)
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Transpose operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        expected_status = self.desc.get('expected_status', 'ARM_CMSIS_NN_SUCCESS')
        force_cmsis = self.desc.get('hint', {}).get('force_cmsis', False)
        force_perm = self.desc.get('hint', {}).get('force_permutation')

        if not force_cmsis and expected_status == "ARM_CMSIS_NN_SUCCESS":
            tflite_path = output_dir / f"{name}.tflite"
            if not tflite_path.exists():
                raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_transpose_kernel()
        
        if force_cmsis or expected_status != "ARM_CMSIS_NN_SUCCESS":
            input_shape = tuple(self.desc['input_shape'])
            output_shape = None
            op_tensors = None
            subgraph_input_indices = set()
            subgraph_output_indices = set()
        else:
            # Load LiteRT model for shape extraction
            from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
            model, subgraph = self.load_litert_model(str(tflite_path))
            op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
            
            # Extract shapes from LiteRT.
            # Prefer subgraph I/O tensors to avoid picking the permutation tensor.
            input_shape = None
            output_shape = None
            subgraph_input_indices = set(subgraph.inputs or [])
            subgraph_output_indices = set(subgraph.outputs or [])

            for input_tensor_info in op_tensors['inputs']:
                tensor_idx = input_tensor_info.get('index', -1)
                tensor_shape = input_tensor_info.get('shape')
                if tensor_idx in subgraph_input_indices:
                    input_shape = tensor_shape
                    break

            if input_shape is None and op_tensors['inputs']:
                input_shape = op_tensors['inputs'][0]['shape']

            for output_tensor_info in op_tensors['outputs']:
                tensor_idx = output_tensor_info.get('index', -1)
                tensor_shape = output_tensor_info.get('shape')
                if tensor_idx in subgraph_output_indices:
                    output_shape = tensor_shape
                    break

            if output_shape is None and op_tensors['outputs']:
                output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        builder = TemplateContextBuilder()
        # Resolve permutation
        if force_perm is not None:
            permutation = list(force_perm)
        else:
            if not force_cmsis and expected_status == "ARM_CMSIS_NN_SUCCESS":
                interpreter = self.load_litert_interpreter(str(tflite_path))
                permutation = [0, 2, 1, 3]
                tensor_details = interpreter.get_tensor_details()
                for tensor in tensor_details:
                    if tensor['name'] and 'perm' in tensor['name'].lower():
                        perm_data = interpreter.get_tensor(tensor['index'])
                        if perm_data is not None and len(perm_data) >= 1:
                            permutation = [int(x) for x in perm_data]
                        break
            else:
                rank = len(input_shape)
                if rank == 4:
                    permutation = [0, 2, 1, 3]
                elif rank == 3:
                    permutation = [0, 2, 1]
                elif rank == 2:
                    permutation = [1, 0]
                elif rank == 1:
                    permutation = [0]
                else:
                    raise ValueError(f"Unsupported input rank for Transpose: {rank}")

        num_dims = len(input_shape)

        # Convert shapes to CMSIS dims (match kernel expectations for num_dims < 4)
        if num_dims == 1:
            input_dims = {'n': int(input_shape[0]), 'h': 1, 'w': 1, 'c': 1}
            output_dims = {'n': int(input_shape[0]), 'h': 1, 'w': 1, 'c': 1}
        elif num_dims == 2:
            input_dims = {'n': int(input_shape[0]), 'h': int(input_shape[1]), 'w': 1, 'c': 1}
            output_dims = {'n': int(input_shape[1]), 'h': int(input_shape[0]), 'w': 1, 'c': 1}
        elif num_dims == 3:
            input_dims = {'n': int(input_shape[0]), 'h': int(input_shape[1]), 'w': int(input_shape[2]), 'c': 1}
            if len(permutation) != 3 or any(p >= 3 or p < 0 for p in permutation):
                output_dims = {'n': int(input_shape[0]), 'h': int(input_shape[1]), 'w': int(input_shape[2]), 'c': 1}
            else:
                out_shape = tuple(int(input_shape[i]) for i in permutation)
                output_dims = {'n': int(out_shape[0]), 'h': int(out_shape[1]), 'w': int(out_shape[2]), 'c': 1}
        else:
            input_dims = builder.nhwc_to_cmsis_dims(input_shape)
            if output_shape is None:
                output_shape = tuple(int(input_shape[i]) for i in permutation)
            output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        self.rng.__setstate__(rng_state)

        # Quantize inputs
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")

        if force_cmsis or expected_status != "ARM_CMSIS_NN_SUCCESS":
            input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
            if expected_status == "ARM_CMSIS_NN_SUCCESS":
                output_data = np.transpose(input_q, permutation)
            else:
                output_data = np.zeros((1,), dtype=np_in_dtype)
        else:
            # Extract quantization from LiteRT (match the selected input tensor)
            input_quant = op_tensors['inputs'][0]['quantization']
            for input_tensor_info in op_tensors['inputs']:
                tensor_idx = input_tensor_info.get('index', -1)
                if tensor_idx in subgraph_input_indices:
                    input_quant = input_tensor_info['quantization']
                    break
            input_scale = input_quant.get('scale', 1.0)
            input_zp = input_quant.get('zero_point', 0)
            if isinstance(input_scale, (list, np.ndarray)):
                input_scale = float(input_scale[0]) if len(input_scale) > 0 else 1.0
            if isinstance(input_zp, (list, np.ndarray)):
                input_zp = int(input_zp[0]) if len(input_zp) > 0 else 0
            input_scale = float(input_scale)
            input_zp = int(input_zp)

            input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
            input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)

            interpreter = self.load_litert_interpreter(str(tflite_path))
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], input_q)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = np.array(output_data)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        permutation_array_str = builder.format_array_as_c_literal(np.array(permutation, dtype=np.uint32))
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'num_dims': num_dims,
            'permutation_array': permutation_array_str,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'expected_status': expected_status,
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("TransposeFunctions/transpose/transpose.h.j2", context)
        h_path = includes_api_dir / f"{name}_transpose.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("TransposeFunctions/transpose/transpose.c.j2", context)
        c_path = output_dir / f"{name}_transpose.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Transpose'),
            'operator_name': 'transpose'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
