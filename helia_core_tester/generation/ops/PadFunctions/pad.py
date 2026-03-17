"""
Pad operation implementation for Helia-Core Tester.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpPad(OperationBase):
    """
    Pad operation.
    """

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Pad uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert model to LiteRT (single-op)."""
        from helia_core_tester.generation.utils.litert_builder import build_pad_op

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        dtype = 'int16' if activation_dtype == 'S16' else 'int8'

        input_shape = tuple(self.desc['input_shape'])
        paddings = self.desc.get('paddings')
        if paddings is None:
            raise ValueError("Pad requires paddings")

        model_bytes = build_pad_op(
            input_shape=input_shape,
            paddings=paddings,
            dtype=dtype,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)
    
    def _select_cmsis_pad_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Pad operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_pad_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_pad_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Pad dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Pad operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_pad_kernel()
        
        input_shape = tuple(self.desc['input_shape'])
        paddings = self.desc.get('paddings')
        if paddings is None:
            raise ValueError("Pad requires paddings")
        pre_pad = [int(p[0]) for p in paddings]
        post_pad = [int(p[1]) for p in paddings]
        output_shape = tuple(
            int(input_shape[i] + pre_pad[i] + post_pad[i]) for i in range(4)
        )
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        pre_pad_dims = builder.nhwc_to_cmsis_dims(pre_pad)
        post_pad_dims = builder.nhwc_to_cmsis_dims(post_pad)

        pad_value = int(self.desc.get('pad_value', 0))

        # Generate input data
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        # Generate input values
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
            input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
            input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")

        output_data = np.pad(
            input_q,
            ((pre_pad[0], post_pad[0]),
             (pre_pad[1], post_pad[1]),
             (pre_pad[2], post_pad[2]),
             (pre_pad[3], post_pad[3])),
            mode="constant",
            constant_values=pad_value,
        )

        self.rng.__setstate__(rng_state)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'pre_pad_dims': pre_pad_dims,
            'post_pad_dims': post_pad_dims,
            'pad_value': pad_value,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("PadFunctions/pad/pad.h.j2", context)
        h_path = includes_api_dir / f"{name}_pad.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("PadFunctions/pad/pad.c.j2", context)
        c_path = output_dir / f"{name}_pad.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Pad'),
            'operator_name': 'pad'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
