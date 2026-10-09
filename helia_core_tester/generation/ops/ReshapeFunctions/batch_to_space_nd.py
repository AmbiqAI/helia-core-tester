"""
BatchToSpaceND operation implementation.
"""

import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpBatchToSpaceND(OperationBase):
    """
    BatchToSpaceND operation.
    """

    def needs_tflite(self) -> bool:
        # The golden is computed in numpy; nothing reads a .tflite.
        return False

    @staticmethod
    def _batch_to_space_nd_numpy(input_np: np.ndarray, block_shape: list, crops: list) -> np.ndarray:
        """Reference BatchToSpaceND for NHWC [N, H, W, C]. block_shape and crops are 2-element."""
        batch, h, w, c = input_np.shape
        b0, b1 = int(block_shape[0]), int(block_shape[1])
        out_batch = batch // (b0 * b1)
        x = input_np.reshape(out_batch, b0, b1, h, w, c)
        x = np.transpose(x, (0, 3, 1, 4, 2, 5))
        x = x.reshape(out_batch, h * b0, w * b1, c)
        (c0_lo, c0_hi), (c1_lo, c1_hi) = crops[0], crops[1]
        x = x[:, c0_lo : (h * b0 - c0_hi), c1_lo : (w * b1 - c1_hi), :]
        return x


    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for BatchToSpaceND.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S16':
            kernel_fn = 'arm_batch_to_space_nd_s16'
            c_type = 'int16_t'
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            kernel_fn = 'arm_batch_to_space_nd_s8'
            c_type = 'int8_t'
            np_in_dtype = np.int8
            qmin, qmax = -128, 127

        crops = self.desc.get('crops', [[0, 0], [0, 0]])
        block_shape = self.desc.get('block_shape', [1, 1])
        input_shape = tuple(self.desc['input_shape'])

        builder = TemplateContextBuilder()
        b0, b1 = int(block_shape[0]), int(block_shape[1])
        out_batch = input_shape[0] // (b0 * b1)
        out_h = input_shape[1] * b0 - int(crops[0][0]) - int(crops[0][1])
        out_w = input_shape[2] * b1 - int(crops[1][0]) - int(crops[1][1])
        output_shape = (out_batch, out_h, out_w, input_shape[3])
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        self.rng.__setstate__(rng_state)
        output_data = self._batch_to_space_nd_numpy(input_q, block_shape, crops)

        crops_flat = [int(crops[0][0]), int(crops[1][0]), int(crops[0][1]), int(crops[1][1])]

        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'block_shape': block_shape,
            'crops': crops_flat,
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': c_type,
            'output_dtype': c_type,
            'kernel_fn': kernel_fn,
            'output_size': int(np.prod(output_shape)),
        }

        self.render_harness_case(
            Path(output_dir), stem="batch_to_space_nd", context=context, pool=batch_to_space_nd_argument_pool(context),
            validation_key="ReshapeFunctions/batch_to_space_nd/batch_to_space_nd.c.j2", label="BatchToSpaceND", operator="BatchToSpaceND",
        )


from helia_core_tester.generation.harness import Declaration  # noqa: E402
from helia_core_tester.generation.harness.simple import tensor_case_pool  # noqa: E402


def batch_to_space_nd_argument_pool(context):
    n, block, crops = context["name"], context["block_shape"], context["crops"]
    extra = (Declaration(f"{n}_block_shape", "cmsis_nn_tile", {"h": block[0], "w": block[1]}),
             Declaration(f"{n}_crop_dims", "cmsis_nn_dims", {"n": crops[0], "h": crops[1], "w": crops[2], "c": crops[3]}))
    return tensor_case_pool(context, {"block_shape": f"&{n}_block_shape", "crop": f"&{n}_crop_dims"},
                            extra_header=extra, output_count=f"({context['output_size']})")
