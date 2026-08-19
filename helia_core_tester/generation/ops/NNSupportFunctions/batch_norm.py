"""Float batch normalization operation implementation."""

from pathlib import Path

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase


class OpBatchNorm(OperationBase):
    """Generate float batch normalization parity tests."""

    def allow_no_tflite(self) -> bool:
        return True

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("BatchNorm uses direct CMSIS-NN generated tests.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        raise NotImplementedError("BatchNorm does not produce a TFLite model.")

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        input_shape = tuple(self.desc["input_shape"])
        channels = int(input_shape[-1])
        activation_dtype = self.tensor_dtype("input", default="FP32")
        if activation_dtype == "FP16":
            float_dtype = np.float16
            data_dtype = "float16_t"
            kernel_fn = "arm_batch_norm_f16"
        elif activation_dtype == "FP32":
            float_dtype = np.float32
            data_dtype = "float"
            kernel_fn = "arm_batch_norm_f32"
        else:
            raise NotImplementedError(f"Unsupported BatchNorm dtype: {activation_dtype}")

        input_data = self._sample_uniform(input_shape, dtype=float_dtype)
        scale = np.linspace(0.5, 1.5, num=channels, dtype=float_dtype)
        bias = np.linspace(-0.25, 0.25, num=channels, dtype=float_dtype)
        output_data = (
            input_data.astype(np.float32) * scale.astype(np.float32).reshape((1, 1, 1, channels))
            + bias.astype(np.float32).reshape((1, 1, 1, channels))
        ).astype(float_dtype)

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "input_dims": builder.nhwc_to_cmsis_dims(input_shape),
            "input_data_array": builder.format_array_as_c_literal(input_data),
            "scale_array": builder.format_array_as_c_literal(scale),
            "bias_array": builder.format_array_as_c_literal(bias),
            "expected_output_array": builder.format_array_as_c_literal(output_data),
            "channels": channels,
            "layout": str(self.desc.get("layout", "ARM_NN_LAYOUT_NHWC")),
            "input_dtype": data_dtype,
            "output_dtype": data_dtype,
            "kernel_fn": kernel_fn,
        }

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "BatchNorm"),
            "operator_name": "batch_norm",
        }
        self._write_op_outputs(
            output_dir,
            "batch_norm",
            "NNSupportFunctions/batch_norm/batch_norm.h.j2",
            "NNSupportFunctions/batch_norm/batch_norm.c.j2",
            context,
            cmake_context,
        )
