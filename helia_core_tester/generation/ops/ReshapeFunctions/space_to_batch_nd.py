"""
SpaceToBatchND operation implementation.
"""

import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpSpaceToBatchND(OperationBase):
    """
    SpaceToBatchND operation.
    """

    @staticmethod
    def _space_to_batch_nd_numpy(input_np: np.ndarray, block_shape: list, paddings: list, pad_value: int) -> np.ndarray:
        batch, h, w, c = input_np.shape
        bh, bw = int(block_shape[0]), int(block_shape[1])
        (pt, pb), (pl, pr) = paddings[0], paddings[1]
        padded = np.pad(
            input_np,
            ((0, 0), (pt, pb), (pl, pr), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
        ph, pw = padded.shape[1], padded.shape[2]
        out_h = ph // bh
        out_w = pw // bw
        x = padded.reshape(batch, out_h, bh, out_w, bw, c)
        x = np.transpose(x, (2, 4, 0, 1, 3, 5))
        x = x.reshape(batch * bh * bw, out_h, out_w, c)
        return x

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for SpaceToBatchND.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S16':
            kernel_fn = 'arm_space_to_batch_nd_s16'
            c_type = 'int16_t'
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            kernel_fn = 'arm_space_to_batch_nd_s8'
            c_type = 'int8_t'
            np_in_dtype = np.int8
            qmin, qmax = -128, 127

        input_shape = tuple(self.desc['input_shape'])
        block_shape = self.desc.get('block_shape', [1, 1])
        paddings = self.desc.get('paddings', [[0, 0], [0, 0]])
        bh, bw = int(block_shape[0]), int(block_shape[1])
        pad_h = int(paddings[0][0]) + int(paddings[0][1])
        pad_w = int(paddings[1][0]) + int(paddings[1][1])
        out_batch = input_shape[0] * bh * bw
        out_h = (input_shape[1] + pad_h) // bh
        out_w = (input_shape[2] + pad_w) // bw
        output_shape = (out_batch, out_h, out_w, input_shape[3])

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        output_zp = 0

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        self.rng.__setstate__(rng_state)
        output_data = self._space_to_batch_nd_numpy(input_q, block_shape, paddings, output_zp)

        paddings_flat = [int(paddings[0][0]), int(paddings[1][0]), int(paddings[0][1]), int(paddings[1][1])]

        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'block_shape': block_shape,
            'paddings': paddings_flat,
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': c_type,
            'output_dtype': c_type,
            'kernel_fn': kernel_fn,
            'output_size': int(np.prod(output_shape)),
            'output_zero_point': int(output_zp),
        }

        self.render_harness_case(
            Path(output_dir), stem="space_to_batch_nd", context=context, pool=space_to_batch_nd_argument_pool(context),
            validation_key="ReshapeFunctions/space_to_batch_nd/space_to_batch_nd.c.j2", label="SpaceToBatchND", operator="SpaceToBatchND",
        )


from helia_core_tester.generation.harness import Declaration  # noqa: E402
from helia_core_tester.generation.harness.simple import tensor_case_pool  # noqa: E402


def space_to_batch_nd_argument_pool(context):
    n, block, pads = context["name"], context["block_shape"], context["paddings"]
    extra = (Declaration(f"{n}_block_shape", "cmsis_nn_tile", {"h": block[0], "w": block[1]}),
             Declaration(f"{n}_pad_dims", "cmsis_nn_dims", {"n": pads[0], "h": pads[1], "w": pads[2], "c": pads[3]}))
    return tensor_case_pool(context, {"block_shape": f"&{n}_block_shape", "pad": f"&{n}_pad_dims",
                                      "output_offset": context["output_zero_point"]},
                            extra_header=extra, output_count=f"({context['output_size']})")
