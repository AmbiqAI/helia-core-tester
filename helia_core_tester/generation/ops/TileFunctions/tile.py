"""Tile operation implementation."""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpTile(OperationBase):
    """Tile operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Tile uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import (
            build_shape_transform_op, TensorSpec,
        )
        import ai_edge_litert.schema_py_generated as litert

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        dtype = 'int16' if activation_dtype == 'S16' else 'int8'

        input_shape = tuple(self.desc['input_shape'])
        multiples = tuple(self.desc['multiples'])
        output_shape = tuple(s * m for s, m in zip(input_shape, multiples))

        multiples_tensor = TensorSpec(
            name="multiples",
            shape=(len(multiples),),
            tensor_type=litert.TensorType.INT32,
            is_input=False,
            data=np.array(multiples, dtype=np.int32),
        )

        model_bytes = build_shape_transform_op(
            op_name="TILE",
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=dtype,
            extra_input_tensors=[multiples_tensor],
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S16':
            return {'kernel_fn': 'arm_tile_s16', 'c_type': 'int16_t', 'np_dtype': 'int16', 'qmin': -32768, 'qmax': 32767}
        return {'kernel_fn': 'arm_tile_s8', 'c_type': 'int8_t', 'np_dtype': 'int8', 'qmin': -128, 'qmax': 127}

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        ki = self._select_kernel()
        input_shape = list(self.desc['input_shape'])
        multiples = list(self.desc['multiples'])
        output_shape = [s * m for s, m in zip(input_shape, multiples)]
        rank = len(input_shape)

        rng = self._seeded_rng()
        np_dtype = np.int16 if ki['np_dtype'] == 'int16' else np.int8
        input_data = rng.integers(ki['qmin'], ki['qmax'] + 1, size=input_shape, dtype=np_dtype)
        output_data = np.tile(input_data, multiples)

        builder = TemplateContextBuilder()
        context = {
            'name': name,
            'prefix': name,
            'rank': rank,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'multiples': multiples,
            'input_size': int(np.prod(input_shape)),
            'output_size': int(np.prod(output_shape)),
            'input_data_array': builder.format_array_as_c_literal(input_data),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'c_type': ki['c_type'],
            'kernel_fn': ki['kernel_fn'],
        }

        includes_dir = output_dir / "includes"
        includes_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("TileFunctions/tile/tile.h.j2", context)
        (includes_dir / f"{name}_tile.h").write_text(h_content)

        c_content = self.render_template("TileFunctions/tile/tile.c.j2", context)
        (output_dir / f"{name}_tile.c").write_text(c_content)

        cmake_content = self.render_template("common/CMakeLists.txt.j2", {
            'name': name, 'operator': 'Tile', 'operator_name': 'tile'
        })
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
