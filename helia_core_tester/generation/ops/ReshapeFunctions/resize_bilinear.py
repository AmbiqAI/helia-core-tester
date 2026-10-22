"""Direct INT8 bilinear resize cases with C reference goldens (TFLite's ResizeBilinearInteger)."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

_KERNELS = ("arm_resize_bilinear_s8", "arm_resize_bilinear_s8_pow2")


class OpResizeBilinear(OperationBase):
    def uses_reference(self) -> bool:
        return True

    def validate_descriptor(self) -> None:
        name = self.desc.get("name", "<unnamed>")
        if self.desc.get("activation_dtype", "S8") != "S8":
            raise ValueError(f"{name}: ResizeBilinear direct kernels require S8")
        kernel = (self.desc.get("hint") or {}).get("kernel", "arm_resize_bilinear_s8")
        if kernel not in _KERNELS:
            raise ValueError(f"{name}: unsupported bilinear entry {kernel!r}")
        shape, size = self.desc.get("input_shape"), self.desc.get("size")
        if not isinstance(shape, (list, tuple)) or len(shape) != 4 or any(int(d) < 1 for d in shape):
            raise ValueError(f"{name}: ResizeBilinear input_shape must be four positive dims, got {shape!r}")
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(int(d) < 1 for d in size):
            raise ValueError(f"{name}: ResizeBilinear size must be two positive dims, got {size!r}")
        if self.desc.get("align_corners") and self.desc.get("half_pixel_centers"):
            raise ValueError(f"{name}: align_corners and half_pixel_centers cannot both be set")

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.reference.call import ReferenceCall

        self.validate_descriptor()
        name = self.desc["name"]
        kernel = (self.desc.get("hint") or {}).get("kernel", "arm_resize_bilinear_s8")
        align_corners = int(bool(self.desc.get("align_corners", False)))
        half_pixel_centers = int(bool(self.desc.get("half_pixel_centers", False)))
        input_shape = tuple(int(d) for d in self.desc["input_shape"])
        output_shape = (input_shape[0], int(self.desc["size"][0]), int(self.desc["size"][1]), input_shape[3])
        inputs = np.random.default_rng(self.seed).integers(-128, 128, input_shape, dtype=np.int8)
        expected = self.reference_golden(ReferenceCall(
            "resize_bilinear_s8", {"align_corners": align_corners, "half_pixel_centers": half_pixel_centers},
            {"input": inputs}, {"output": output_shape}))
        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "kernel_fn": kernel,
            "input_dims": builder.nhwc_to_cmsis_dims(input_shape),
            "output_dims": builder.nhwc_to_cmsis_dims(output_shape),
            "input_dtype": "int8_t",
            "output_dtype": "int8_t",
            "input_data_array": builder.format_array_as_c_literal(inputs),
            "expected_output_array": builder.format_array_as_c_literal(expected),
            "output_size": int(expected.size),
            "align_corners": align_corners,
            "half_pixel_centers": half_pixel_centers,
            "validation_mode": "exact_int",
            "validation_tolerance": 0,
        }
        template = "ReshapeFunctions/resize_bilinear/resize_bilinear"
        self._write_op_outputs(
            output_dir,
            "resize_bilinear",
            template + ".h.j2",
            template + ".c.j2",
            context,
            {"name": name, "operator": "ResizeBilinear", "operator_name": "resize_bilinear"},
        )


from helia_core_tester.generation.harness.model import Declaration  # noqa: E402
from helia_core_tester.generation.harness.registry import harness_pool  # noqa: E402
from helia_core_tester.generation.harness.simple import tensor_case_pool  # noqa: E402


@harness_pool("ReshapeFunctions/resize_bilinear/resize_bilinear.c.j2", label="ResizeBilinear")
def resize_bilinear_argument_pool(context):
    """arm_resize_bilinear_s8[_pow2](params, input_dims, input, output_dims, output): no scratch."""
    n = context["name"]
    source = (Declaration(f"{n}_params", "cmsis_nn_resize_params",
                          {"align_corners": str(context["align_corners"]),
                           "half_pixel_centers": str(context["half_pixel_centers"])}, storage="static"),)
    pool = tensor_case_pool(context, {"resize_params": f"&{n}_params"}, output_count=f"({context['output_size']})")
    return replace(pool, source=source)
