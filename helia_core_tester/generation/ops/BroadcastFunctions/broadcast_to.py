"""BroadcastTo operation implementation."""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpBroadcastTo(OperationBase):
    """BroadcastTo operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("BroadcastTo uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import (
            build_shape_transform_op, TensorSpec,
        )
        import ai_edge_litert.schema_py_generated as litert

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        dtype = 'int16' if activation_dtype == 'S16' else 'int8'

        input_shape = tuple(self.desc['input_shape'])
        output_shape = tuple(self.desc['output_shape'])

        shape_tensor = TensorSpec(
            name="shape",
            shape=(len(output_shape),),
            tensor_type=litert.TensorType.INT32,
            is_input=False,
            data=np.array(output_shape, dtype=np.int32),
        )

        model_bytes = build_shape_transform_op(
            op_name="BROADCAST_TO",
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=dtype,
            extra_input_tensors=[shape_tensor],
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S16':
            return {'kernel_fn': 'arm_broadcast_to_s16', 'c_type': 'int16_t', 'np_dtype': 'int16', 'qmin': -32768, 'qmax': 32767}
        return {'kernel_fn': 'arm_broadcast_to_s8', 'c_type': 'int8_t', 'np_dtype': 'int8', 'qmin': -128, 'qmax': 127}

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        ki = self._select_kernel()
        input_shape = list(self.desc['input_shape'])
        output_shape = list(self.desc['output_shape'])
        rank = len(output_shape)

        rng = self._seeded_rng()
        np_dtype = np.int16 if ki['np_dtype'] == 'int16' else np.int8
        input_data = rng.integers(ki['qmin'], ki['qmax'] + 1, size=input_shape, dtype=np_dtype)
        output_data = np.broadcast_to(input_data, output_shape).copy()

        builder = TemplateContextBuilder()
        context = {
            'name': name,
            'prefix': name,
            'rank': rank,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'input_size': int(np.prod(input_shape)),
            'output_size': int(np.prod(output_shape)),
            'input_data_array': builder.format_array_as_c_literal(input_data),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'c_type': ki['c_type'],
            'kernel_fn': ki['kernel_fn'],
        }

        includes_dir = output_dir / "includes"
        includes_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("BroadcastFunctions/broadcast_to/broadcast_to.h.j2", context)
        (includes_dir / f"{name}_broadcast_to.h").write_text(h_content)

        c_content = self.render_template("BroadcastFunctions/broadcast_to/broadcast_to.c.j2", context)
        (output_dir / f"{name}_broadcast_to.c").write_text(c_content)

        cmake_content = self.render_template("common/CMakeLists.txt.j2", {
            'name': name, 'operator': 'BroadcastTo', 'operator_name': 'broadcast_to'
        })
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
