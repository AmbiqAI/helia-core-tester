"""
Squeeze operation implementation.
"""

from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.copy_pool import copy_argument_pool


def squeezed_shape(input_shape: Sequence[int], axes: Optional[Sequence[int]]) -> Tuple[int, ...]:
    """TF squeeze of the full (batch-included) shape: every size-1 dim, or only `axes`."""
    shape = tuple(int(d) for d in input_shape)
    rank = len(shape)
    if axes is None:
        return tuple(d for d in shape if d != 1)
    picked = set()
    for axis in axes:
        a = int(axis)
        if not -rank <= a < rank:
            raise ValueError(f"squeeze axis {a} is out of range for rank {rank}")
        a %= rank
        if shape[a] != 1:
            raise ValueError(f"squeeze axis {a} has size {shape[a]}, not 1")
        picked.add(a)
    return tuple(d for i, d in enumerate(shape) if i not in picked)


class OpSqueeze(OperationBase):
    """
    Squeeze operation - removes dimensions of size 1.

    The data layout does not change, so the kernel is arm_reshape_s8 and the golden is
    the reference reshape of the input to the squeezed shape.
    """

    def uses_reference(self) -> bool:
        return True

    def _select_cmsis_squeeze_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_reshape_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        raise NotImplementedError(f"Unsupported Squeeze dtype: {activation_dtype} (only S8 supported)")

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_info = self._select_cmsis_squeeze_kernel()

        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        # A fully squeezed tensor is a scalar; the harness and the reference carry it as (1,).
        output_shape = squeezed_shape(input_shape, self.desc.get('axes')) or (1,)

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        total_size = int(np.prod(input_shape))

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        self.rng.__setstate__(rng_state)

        # Quantized over the [-1, 1] draw range, the range the converter used to calibrate over.
        in_quant = self.activation_quant("input", (-1.0, 1.0), "s8")
        input_q = policy.quantize(input_data, in_quant)

        output_data = self.reference_golden(ReferenceCall(
            "reshape", {"unused": 0}, {"input": np.ascontiguousarray(input_q)}, {"output": output_shape},
            quant={"input": in_quant.to_json()}))

        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'total_size': total_size,
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }

        self.render_harness_case(
            output_dir, stem="squeeze", context=context, pool=copy_argument_pool(context),
            validation_key="TesterExtensions/squeeze/squeeze.c.j2", label="Squeeze", operator="Squeeze",
        )
