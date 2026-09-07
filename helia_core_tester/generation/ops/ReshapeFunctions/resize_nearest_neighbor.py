"""
ResizeNearestNeighbor operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.io.dtypes import descriptor_dtype_to_litert_dtype
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpResizeNearestNeighbor(OperationBase):
    """
    ResizeNearestNeighbor operation.
    """

    # Dtypes the LiteRT model side can build. The harness side is narrower; see generate_c_files.
    SUPPORTED_MODEL_DTYPES = ('S8', 'S16', 'FP32', 'FP16')

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("ResizeNearestNeighbor uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_resize_nearest_neighbor_op

        activation_dtype = str(self.desc.get('activation_dtype', 'S8')).upper()
        if activation_dtype not in self.SUPPORTED_MODEL_DTYPES:
            raise NotImplementedError(f"Unsupported ResizeNearestNeighbor dtype: {activation_dtype}")
        dtype = descriptor_dtype_to_litert_dtype(activation_dtype)

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
        # Mirrors GetNearestNeighbor in ns-cmsis-nn Include/arm_nnsupportfunctions.h (the TFLite
        # reference rule): roundf on the align_corners path, floorf otherwise, all in float32.
        # The same arithmetic in double picks a different source pixel wherever the float32
        # product lands on the other side of a .5 or of an integer, so every operand is wrapped:
        # a numpy float32 scalar mixed with a Python int or float promotes to float64, while
        # float32-only arithmetic stays float32.
        f32 = np.float32
        if align_corners and out_size > 1:
            scale = f32(in_size - 1) / f32(out_size - 1)
        else:
            scale = f32(in_size) / f32(out_size)
        offset = f32(0.5) if half_pixel_centers else f32(0.0)
        scaled = (f32(out_idx) + offset) * scale
        whole = np.floor(scaled)
        if align_corners:
            # roundf takes ties away from zero and scaled is never negative here, so a tie goes
            # up. np.round is ties-to-even and disagrees; floor(scaled + 0.5) disagrees too,
            # because that sum itself rounds up in float32 just below a tie.
            idx = int(whole) + (1 if scaled - whole >= f32(0.5) else 0)
        else:
            idx = int(whole)
        idx = min(idx, in_size - 1)
        if half_pixel_centers:
            idx = max(0, idx)
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

        activation_dtype = str(self.desc.get('activation_dtype', 'S8')).upper()
        if activation_dtype == 'S16':
            kernel_fn = 'arm_resize_nearest_neighbor_s16'
            c_type = 'int16_t'
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        elif activation_dtype == 'S8':
            kernel_fn = 'arm_resize_nearest_neighbor_s8'
            c_type = 'int8_t'
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        else:
            # The model side already builds float resize, but the float kernels are not wired
            # here yet (no kernel_registry entry, no float branch above), so refuse rather than
            # pair a float model with the s8 kernel and integer sampling. ValueError, not
            # NotImplementedError: the generation pipeline treats the latter as "this operator
            # has no C generation yet" and drops the descriptor with only an INFO line.
            raise ValueError(f"ResizeNearestNeighbor harness generation supports S8 and S16, got {activation_dtype}")

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

        h_content = self.render_template("ReshapeFunctions/resize_nearest_neighbor/resize_nearest_neighbor.h.j2", context)
        (includes_api_dir / f"{name}_resize_nearest_neighbor.h").write_text(h_content)
        c_content = self.render_template("ReshapeFunctions/resize_nearest_neighbor/resize_nearest_neighbor.c.j2", context)
        (output_dir / f"{name}_resize_nearest_neighbor.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'ResizeNearestNeighbor'),
            'operator_name': 'resize_nearest_neighbor',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (output_dir / "CMakeLists.txt").write_text(cmake_content)

