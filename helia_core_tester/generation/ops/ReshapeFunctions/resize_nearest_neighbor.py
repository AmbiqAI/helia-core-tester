"""
ResizeNearestNeighbor operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpResizeNearestNeighbor(OperationBase):
    """
    ResizeNearestNeighbor operation.
    """

    def needs_tflite(self) -> bool:
        # The golden is computed in numpy; nothing reads a .tflite.
        return False

    # Dtypes the LiteRT model side can build. The harness side is narrower; see generate_c_files.
    SUPPORTED_MODEL_DTYPES = ('S8', 'S16', 'FP32', 'FP16')

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

        self.render_harness_case(
            Path(output_dir), stem="resize_nearest_neighbor", context=context, pool=resize_nearest_neighbor_argument_pool(context),
            validation_key="ReshapeFunctions/resize_nearest_neighbor/resize_nearest_neighbor.c.j2", label="ResizeNearestNeighbor", operator="ResizeNearestNeighbor",
        )



from helia_core_tester.generation.harness import ArrayLiteral, Declaration, GuardedBuffer  # noqa: E402
from helia_core_tester.generation.harness.simple import dims_declaration, tensor_case_pool  # noqa: E402


def resize_nearest_neighbor_argument_pool(context):
    """The resize kernels take a context whose guarded int32 buffer the case owns."""
    n = context["name"]
    extra = (dims_declaration(f"{n}_output_size_dims", context["output_size_dims"]),
             Declaration(f"{n}_output_size_data", "int32_t", ArrayLiteral(context["output_size_array"]), array=True))
    source = (Declaration(f"{n}_ctx", "cmsis_nn_context", {"buf": f"{n}_buffer", "size": f"sizeof({n}_buffer)"},
                          storage="static"),
              Declaration(f"{n}_params", "cmsis_nn_resize_params",
                          {"align_corners": str(context["align_corners"]),
                           "half_pixel_centers": str(context["half_pixel_centers"])}, storage="static"))
    label = '{{ validation_label | default("ResizeNearestNeighbor") }}'
    pool = tensor_case_pool(
        context,
        {"ctx": f"&{n}_ctx", "resize_params": f"&{n}_params", "input_shape": f"&{n}_input_dims",
         "output_size_shape": f"&{n}_output_size_dims", "output_size_data": f"{n}_output_size_data",
         "output_shape": f"&{n}_output_dims"},
        extra_header=extra, output_count=f"({context['output_size']})", owns_ctx=True,
        guarded=(GuardedBuffer(f"{n}_buffer", "int32_t", str(context["buffer_size"])),),
        test_prologue=f"    HELIA_GUARD_ARM({n}_buffer, true /* pure scratch: poison to catch read-before-write */);",
        extra_checks=f'    HELIA_GUARD_CHECK({n}_buffer, "{label} scratch", failures);',
    )
    from dataclasses import replace

    return replace(pool, source=source)
