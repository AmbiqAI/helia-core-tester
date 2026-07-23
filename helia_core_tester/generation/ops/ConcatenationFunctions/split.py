"""
Split operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpSplit(OperationBase):
    """
    Split operation - splits a tensor into multiple tensors.
    """

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Split uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_split_op

        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        elif activation_dtype == "FP16":
            dtype = "float16"
        else:
            raise NotImplementedError(f"Unsupported Split dtype: {activation_dtype}")

        input_shape = tuple(self.desc["input_shape"])
        axis = int(self.desc.get("axis", -1))
        num_splits = self.desc.get("num_splits", None)
        size_splits = self.desc.get("size_splits", None)

        model_bytes = build_split_op(
            input_shape=input_shape,
            axis=axis,
            num_splits=num_splits,
            size_splits=size_splits,
            dtype=dtype,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)
    
    def _select_cmsis_split_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Split operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_split_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_split_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        elif activation_dtype == 'FP16':
            return {
                'kernel_fn': 'arm_split_f16',
                'input_c_type': 'float16_t',
                'output_c_type': 'float16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Split dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Split operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_split_kernel()
        
        # Build shapes from descriptor
        input_shape = tuple(self.desc["input_shape"])
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        
        # Extract axis and num_splits/size_splits from descriptor
        axis = int(self.desc.get('axis', -1))
        num_splits = self.desc.get('num_splits', None)
        size_splits = self.desc.get('size_splits', None)
        
        # Convert axis to 0-based
        if axis < 0:
            axis = len(input_shape) + axis
        
        # Calculate split_dims
        if size_splits is not None:
            split_dims = [int(v) for v in size_splits]
            num_splits = len(split_dims)
        elif num_splits is not None:
            axis_size = int(input_shape[axis])
            split_size = axis_size // int(num_splits)
            split_dims = [split_size] * int(num_splits)
        else:
            raise ValueError("Split operation requires either 'num_splits' or 'size_splits'")
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
            input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
            input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        elif kernel_info["input_c_type"] == "float16_t":
            np_in_dtype = np.float16
            input_q = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np_in_dtype)
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")

        self.rng.__setstate__(rng_state)

        if size_splits is not None:
            indices = np.cumsum(split_dims)[:-1]
            output_arrays = np.split(input_q, indices, axis=axis)
        else:
            output_arrays = np.split(input_q, int(num_splits), axis=axis)

        outputs = []
        for idx, out_data in enumerate(output_arrays):
            out_data = np.array(out_data, dtype=np_in_dtype)
            outputs.append({
                'name': f"{name}_out_{idx}",
                'expected_output_array': builder.format_array_as_c_literal(out_data),
                'size': int(np.prod(out_data.shape)),
            })
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        split_dims_array_str = builder.format_array_as_c_literal(np.array(split_dims, dtype=np.int32))
        input_shape_array_str = builder.format_array_as_c_literal(np.array(input_shape, dtype=np.int32))
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'input_dims_count': len(input_shape),
            'axis': axis,
            'num_splits': num_splits,
            'split_dims_array': split_dims_array_str,
            'input_shape_array': input_shape_array_str,
            'input_data_array': input_array_str,
            'outputs': outputs,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        if kernel_info["input_c_type"] == "float16_t":
            context["validation_mode"] = "float"
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("ConcatenationFunctions/split/split.h.j2", context)
        h_path = includes_api_dir / f"{name}_split.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("ConcatenationFunctions/split/split.c.j2", context)
        c_path = output_dir / f"{name}_split.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Split'),
            'operator_name': 'split'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
