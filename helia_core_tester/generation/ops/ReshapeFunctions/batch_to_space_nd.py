"""
BatchToSpaceND operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpBatchToSpaceND(OperationBase):
    """
    BatchToSpaceND operation.
    """

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("BatchToSpaceND uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_batch_to_space_nd_op

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        dtype = 'int16' if activation_dtype == 'S16' else 'int8'

        model_bytes = build_batch_to_space_nd_op(
            input_shape=self.desc['input_shape'],
            block_shape=self.desc.get('block_shape', [1, 1]),
            crops=self.desc.get('crops', [[0, 0], [0, 0]]),
            dtype=dtype,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)

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

    @staticmethod
    def _extract_quantization(details):
        """Return (scale, zero_point) from interpreter tensor details."""
        qp = details.get('quantization_parameters') or {}
        scales = qp.get('scales') if isinstance(qp, dict) else None
        zero_points = qp.get('zero_points') if isinstance(qp, dict) else None
        if scales is not None and len(scales) > 0:
            scale = float(scales[0])
        else:
            scale = details.get('quantization', (1.0, 0))[0]
            scale = float(scale) if scale is not None else 1.0
        if zero_points is not None and len(zero_points) > 0:
            zero_point = int(zero_points[0])
        else:
            zero_point = details.get('quantization', (1.0, 0))[1]
            zero_point = int(zero_point) if zero_point is not None else 0
        return scale, zero_point

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for BatchToSpaceND.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        tflite_path = Path(output_dir) / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

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

        includes_api_dir = Path(output_dir) / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("ReshapeFunctions/batch_to_space_nd/batch_to_space_nd.h.j2", context)
        (includes_api_dir / f"{name}_batch_to_space_nd.h").write_text(h_content)
        c_content = self.render_template("ReshapeFunctions/batch_to_space_nd/batch_to_space_nd.c.j2", context)
        (Path(output_dir) / f"{name}_batch_to_space_nd.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'BatchToSpaceND'),
            'operator_name': 'batch_to_space_nd',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (Path(output_dir) / "CMakeLists.txt").write_text(cmake_content)
