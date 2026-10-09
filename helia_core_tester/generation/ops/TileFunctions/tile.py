"""Tile operation implementation."""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpTile(OperationBase):
    """Tile operation."""

    def needs_tflite(self) -> bool:
        # The golden is computed in numpy; nothing reads a .tflite.
        return False

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

        if len(multiples) != rank or any(m < 1 for m in multiples):
            raise ValueError(f"{name}: multiples {multiples} do not fit input rank {rank}")
        output_data = np.tile(input_data, multiples)

        builder = TemplateContextBuilder()
        context = {
            'name': name,
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

        self.render_harness_case(
            Path(output_dir), stem="tile", context=context, pool=tile_argument_pool(context),
            validation_key="TileFunctions/tile/tile.c.j2", label="Tile", operator="Tile",
        )


from helia_core_tester.generation.harness.simple import shaped_case_pool  # noqa: E402


def tile_argument_pool(context):
    n = context["name"]
    return shaped_case_pool(
        context, shapes=(("input_shape", "input_shape"), ("multiples", "multiples")), params_type="cmsis_nn_tile_params",
        params={"rank": context["rank"], "input_shape": f"{n}_input_shape", "multiples": f"{n}_multiples"},
        inputs=(("input", "input", "input_data_array"),), output_count=str(context["output_size"]))
