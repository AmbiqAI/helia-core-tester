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

    def uses_reference(self) -> bool:
        return True

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for SpaceToBatchND.
        """
        from helia_core_tester.generation.reference.call import ReferenceCall
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
        output_data = self.reference_golden(ReferenceCall(
            "space_to_batch_nd", {"unused": 0},
            {"input": np.ascontiguousarray(input_q),
             "block_shape": np.array([bh, bw], dtype=np.int32),
             "paddings": np.array([[int(v) for v in paddings[0]], [int(v) for v in paddings[1]]], dtype=np.int32),
             "pad_value": np.array([output_zp], dtype=np_in_dtype)},
            {"output": output_shape}))

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
