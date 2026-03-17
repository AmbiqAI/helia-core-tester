"""
SpaceToDepth operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpSpaceToDepth(OperationBase):
    """
    SpaceToDepth operation.
    """

    def needs_keras_model(self) -> bool:
        return False
    
    def build_keras_model(self):
        raise NotImplementedError("SpaceToDepth uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert model to LiteRT (single-op) to avoid TF/TFLite dependency."""
        from helia_core_tester.generation.utils.litert_builder import build_space_to_depth_op

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        dtype = 'int16' if activation_dtype == 'S16' else 'int8'
        model_bytes = build_space_to_depth_op(
            input_shape=self.desc['input_shape'],
            block_size=int(self.desc.get('block_size', 2)),
            dtype=dtype,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)


    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for SpaceToDepth.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        tflite_path = Path(output_dir) / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S16':
            kernel_fn = 'arm_space_to_depth_s16'
            c_type = 'int16_t'
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            kernel_fn = 'arm_space_to_depth_s8'
            c_type = 'int8_t'
            np_in_dtype = np.int8
            qmin, qmax = -128, 127

        input_shape = tuple(self.desc['input_shape'])
        block_size = int(self.desc.get('block_size', 2))
        n, h, w, c = input_shape
        output_shape = (n, h // block_size, w // block_size, c * block_size * block_size)

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        self.rng.__setstate__(rng_state)

        x = input_q.reshape(n, h // block_size, block_size, w // block_size, block_size, c)
        x = np.transpose(x, (0, 1, 3, 2, 4, 5))
        output_data = x.reshape(output_shape)

        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'block_size': int(self.desc.get('block_size', 1)),
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': c_type,
            'output_dtype': c_type,
            'kernel_fn': kernel_fn,
            'output_size': int(np.prod(output_shape)),
        }

        includes_api_dir = Path(output_dir) / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("ReshapeFunctions/space_to_depth/space_to_depth.h.j2", context)
        (includes_api_dir / f"{name}_space_to_depth.h").write_text(h_content)
        c_content = self.render_template("ReshapeFunctions/space_to_depth/space_to_depth.c.j2", context)
        (Path(output_dir) / f"{name}_space_to_depth.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'SpaceToDepth'),
            'operator_name': 'space_to_depth',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (Path(output_dir) / "CMakeLists.txt").write_text(cmake_content)
