"""
SpaceToBatchND operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpSpaceToBatchND(OperationBase):
    """
    SpaceToBatchND operation.
    """

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("SpaceToBatchND uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_space_to_batch_nd_op

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        dtype = 'int16' if activation_dtype == 'S16' else 'int8'

        model_bytes = build_space_to_batch_nd_op(
            input_shape=self.desc['input_shape'],
            block_shape=self.desc.get('block_shape', [1, 1]),
            paddings=self.desc.get('paddings', [[0, 0], [0, 0]]),
            dtype=dtype,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)

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
        tflite_path = Path(output_dir) / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

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
            'prefix': name,
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

        includes_api_dir = Path(output_dir) / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("space_to_batch_nd/space_to_batch_nd.h.j2", context)
        (includes_api_dir / f"{name}_space_to_batch_nd.h").write_text(h_content)
        c_content = self.render_template("space_to_batch_nd/space_to_batch_nd.c.j2", context)
        (Path(output_dir) / f"{name}_space_to_batch_nd.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'SpaceToBatchND'),
            'operator_name': 'space_to_batch_nd',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (Path(output_dir) / "CMakeLists.txt").write_text(cmake_content)
