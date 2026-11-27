"""Direct integer and floating-point bilinear resize cases with LiteRT reference outputs."""

from pathlib import Path

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.litert_builder import build_resize_bilinear_op
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


class OpResizeBilinear(OperationBase):
    def _storage_dtype(self) -> str:
        dtype = self.tensor_dtype("input", default=self.desc.get("activation_dtype", "S8"))
        if dtype not in {"S8", "FP16", "FP32"} or self.tensor_dtype("output", default=dtype) != dtype:
            raise ValueError("ResizeBilinear direct kernels require S8, FP16 or FP32 with matching output")
        return dtype

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("ResizeBilinear uses a single-op LiteRT model.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        dtype = self._storage_dtype()
        Path(out_path).write_bytes(
            build_resize_bilinear_op(
                input_shape=self.desc["input_shape"],
                new_size=self.desc["size"],
                dtype="int8" if dtype == "S8" else "float32",
                align_corners=bool(self.desc.get("align_corners", False)),
                half_pixel_centers=bool(self.desc.get("half_pixel_centers", False)),
            )
        )

    def generate_c_files(self, output_dir: Path) -> None:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType

        name = self.desc["name"]
        dtype = self._storage_dtype()
        suffix = {"S8": "s8", "FP16": "f16", "FP32": "f32"}[dtype]
        entry = f"arm_resize_bilinear_{suffix}"
        kernel = self.desc.get("hint", {}).get("kernel", entry)
        if kernel != entry and not (dtype == "S8" and kernel == "arm_resize_bilinear_s8_pow2"):
            raise ValueError(f"Unsupported bilinear entry for {dtype}: {kernel}")
        interpreter = Interpreter(
            model_path=str(output_dir / f"{name}.tflite"),
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        )
        interpreter.allocate_tensors()
        rng = np.random.default_rng(self.seed)
        shape = self.desc["input_shape"]
        if dtype == "S8":
            inputs = rng.integers(-128, 128, shape, dtype=np.int8)
        else:
            storage = np.float16 if dtype == "FP16" else np.float32
            inputs = rng.uniform(-2, 2, shape).astype(storage)
        # For half storage, widen the actual stored samples before reference interpolation.
        detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(detail["index"], inputs.astype(detail["dtype"], copy=False))
        interpreter.invoke()
        expected = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        expected = expected.astype(inputs.dtype, copy=False)
        c_type = self.tensor_c_type("input", default=dtype)
        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "kernel_fn": kernel,
            "input_dims": builder.nhwc_to_cmsis_dims(inputs.shape),
            "output_dims": builder.nhwc_to_cmsis_dims(expected.shape),
            "input_dtype": c_type,
            "output_dtype": c_type,
            "input_data_array": builder.format_array_as_c_literal(inputs),
            "expected_output_array": builder.format_array_as_c_literal(expected),
            "output_size": int(expected.size),
            "align_corners": int(bool(self.desc.get("align_corners", False))),
            "half_pixel_centers": int(bool(self.desc.get("half_pixel_centers", False))),
            "validation_mode": "exact_int" if dtype == "S8" else "float",
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
