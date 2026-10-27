"""
ResizeNearestNeighbor operation implementation.
"""

import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpResizeNearestNeighbor(OperationBase):
    """
    ResizeNearestNeighbor operation.
    """

    def uses_reference(self) -> bool:
        return True

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.reference.call import ReferenceCall
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
            # The float kernels are not wired here yet (no kernel_registry entry, no float
            # branch above), so refuse rather than pair a float case with the s8 kernel and
            # integer sampling. ValueError, not NotImplementedError: the generation pipeline
            # treats the latter as "this operator has no C generation yet" and drops the
            # descriptor with only an INFO line.
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

        # TFLite GetNearestNeighbor in float32 (the rule the kernel transcribes).
        output_data = self.reference_golden(ReferenceCall(
            "resize_nearest_neighbor",
            {"align_corners": int(align_corners), "half_pixel_centers": int(half_pixel_centers)},
            {"input": np.ascontiguousarray(input_q)}, {"output": output_shape}))

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
