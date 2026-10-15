"""Direct INT8 bilinear resize cases with LiteRT reference outputs."""

from pathlib import Path

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.litert_builder import build_resize_bilinear_op
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


class OpResizeBilinear(OperationBase):
    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("ResizeBilinear uses a single-op LiteRT model.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        if self.desc.get("activation_dtype", "S8") != "S8":
            raise ValueError("ResizeBilinear direct kernels require S8")
        Path(out_path).write_bytes(
            build_resize_bilinear_op(
                input_shape=self.desc["input_shape"],
                new_size=self.desc["size"],
                align_corners=bool(self.desc.get("align_corners", False)),
                half_pixel_centers=bool(self.desc.get("half_pixel_centers", False)),
            )
        )

    def generate_c_files(self, output_dir: Path) -> None:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType

        name = self.desc["name"]
        kernel = self.desc.get("hint", {}).get("kernel", "arm_resize_bilinear_s8")
        if kernel not in {"arm_resize_bilinear_s8", "arm_resize_bilinear_s8_pow2"}:
            raise ValueError(f"Unsupported bilinear entry: {kernel}")
        interpreter = Interpreter(
            model_path=str(output_dir / f"{name}.tflite"),
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        )
        interpreter.allocate_tensors()
        inputs = np.random.default_rng(self.seed).integers(
            -128,
            128,
            self.desc["input_shape"],
            dtype=np.int8,
        )
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], inputs)
        interpreter.invoke()
        expected = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "kernel_fn": kernel,
            "input_dims": builder.nhwc_to_cmsis_dims(inputs.shape),
            "output_dims": builder.nhwc_to_cmsis_dims(expected.shape),
            "input_dtype": "int8_t",
            "output_dtype": "int8_t",
            "input_data_array": builder.format_array_as_c_literal(inputs),
            "expected_output_array": builder.format_array_as_c_literal(expected),
            "output_size": int(expected.size),
            "align_corners": int(bool(self.desc.get("align_corners", False))),
            "half_pixel_centers": int(bool(self.desc.get("half_pixel_centers", False))),
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
            {
                "name": name,
                "operator": "ResizeBilinear",
                "operator_name": "resize_bilinear",
            },
        )
