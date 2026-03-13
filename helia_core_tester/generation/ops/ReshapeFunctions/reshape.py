"""
Reshape operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpReshape(OperationBase):
    """
    Reshape operation.
    """

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Reshape uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_reshape_op

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype != 'S8':
            raise NotImplementedError(f"Unsupported Reshape dtype: {activation_dtype} (only S8 supported)")

        input_shape = tuple(self.desc['input_shape'])
        target_shape = tuple(self.desc.get('target_shape'))
        if target_shape is None:
            raise ValueError("Reshape operation requires 'target_shape' in descriptor")

        model_bytes = build_reshape_op(
            input_shape=input_shape,
            target_shape=target_shape,
            dtype="int8",
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)
    
    def _select_cmsis_reshape_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Reshape operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_reshape_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Reshape dtype: {activation_dtype} (only S8 supported)")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Reshape operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_reshape_kernel()
        
        input_shape = tuple(self.desc['input_shape'])
        output_shape = tuple(self.desc.get('target_shape'))
        if output_shape is None:
            raise ValueError("Reshape operation requires 'target_shape' in descriptor")
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Calculate total size (should be same for input and output)
        total_size = int(np.prod(input_shape))
        
        # Generate input data
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_q = self.rng.integers(-128, 128, size=input_shape, dtype=np.int8)
        self.rng.__setstate__(rng_state)

        output_data = input_q.reshape(output_shape)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'total_size': total_size,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("ReshapeFunctions/reshape/reshape.h.j2", context)
        h_path = includes_api_dir / f"{name}_reshape.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("ReshapeFunctions/reshape/reshape.c.j2", context)
        c_path = output_dir / f"{name}_reshape.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Reshape'),
            'operator_name': 'reshape'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
