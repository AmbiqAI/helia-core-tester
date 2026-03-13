"""
ResizeNearestNeighbor operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpResizeNearestNeighbor(OperationBase):
    """
    ResizeNearestNeighbor operation.
    """

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("ResizeNearestNeighbor uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_resize_nearest_neighbor_op

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        dtype = 'int16' if activation_dtype == 'S16' else 'int8'

        input_shape = tuple(self.desc['input_shape'])
        new_size = self.desc.get('size')
        if new_size is None:
            raise ValueError("ResizeNearestNeighbor requires 'size' in descriptor")

        model_bytes = build_resize_nearest_neighbor_op(
            input_shape=input_shape,
            new_size=new_size,
            align_corners=bool(self.desc.get('align_corners', False)),
            half_pixel_centers=bool(self.desc.get('half_pixel_centers', False)),
            dtype=dtype,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)

    @staticmethod
    def _nearest_index(out_idx: int, in_size: int, out_size: int, align_corners: bool, half_pixel_centers: bool) -> int:
        if out_size == 1:
            return 0
        if align_corners and out_size > 1:
            scale = (in_size - 1) / (out_size - 1)
        else:
            scale = in_size / out_size
        offset = 0.5 if half_pixel_centers else 0.0
        scaled = (out_idx + offset) * scale
        if align_corners:
            idx = int(np.round(scaled))
        else:
            idx = int(np.floor(scaled))
        if idx > in_size - 1:
            idx = in_size - 1
        if half_pixel_centers and idx < 0:
            idx = 0
        return idx

    @classmethod
    def _resize_nearest_neighbor_np(
        cls,
        input_q: np.ndarray,
        out_h: int,
        out_w: int,
        align_corners: bool,
        half_pixel_centers: bool,
    ) -> np.ndarray:
        n, in_h, in_w, c = input_q.shape
        output = np.zeros((n, out_h, out_w, c), dtype=input_q.dtype)
        for y in range(out_h):
            in_y = cls._nearest_index(y, in_h, out_h, align_corners, half_pixel_centers)
            for x in range(out_w):
                in_x = cls._nearest_index(x, in_w, out_w, align_corners, half_pixel_centers)
                output[:, y, x, :] = input_q[:, in_y, in_x, :]
        return output

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S16':
            kernel_fn = 'arm_resize_nearest_neighbor_s16'
            c_type = 'int16_t'
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            kernel_fn = 'arm_resize_nearest_neighbor_s8'
            c_type = 'int8_t'
            np_in_dtype = np.int8
            qmin, qmax = -128, 127

        input_shape = tuple(self.desc['input_shape'])
        size = self.desc.get('size')
        if size is None:
            raise ValueError("ResizeNearestNeighbor requires 'size' in descriptor")
        out_h, out_w = int(size[0]), int(size[1])
        output_shape = (input_shape[0], out_h, out_w, input_shape[3])

        align_corners = bool(self.desc.get('align_corners', False))
        half_pixel_centers = bool(self.desc.get('half_pixel_centers', False))

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        self.rng.__setstate__(rng_state)

        output_data = self._resize_nearest_neighbor_np(input_q, out_h, out_w, align_corners, half_pixel_centers)

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        size_shape = (2,)
        size_dims = builder.nhwc_to_cmsis_dims(size_shape)

        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'output_size_dims': size_dims,
            'output_size_array': builder.format_array_as_c_literal(np.array([out_h, out_w], dtype=np.int32)),
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': c_type,
            'output_dtype': c_type,
            'kernel_fn': kernel_fn,
            'output_size': int(np.prod(output_shape)),
            'align_corners': 1 if align_corners else 0,
            'half_pixel_centers': 1 if half_pixel_centers else 0,
            'buffer_size': int(out_h + out_w),
        }

        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("resize_nearest_neighbor/resize_nearest_neighbor.h.j2", context)
        (includes_api_dir / f"{name}_resize_nearest_neighbor.h").write_text(h_content)
        c_content = self.render_template("resize_nearest_neighbor/resize_nearest_neighbor.c.j2", context)
        (output_dir / f"{name}_resize_nearest_neighbor.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'ResizeNearestNeighbor'),
            'operator_name': 'resize_nearest_neighbor',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (output_dir / "CMakeLists.txt").write_text(cmake_content)

