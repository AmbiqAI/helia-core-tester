"""Float batch normalization operation implementation."""

from pathlib import Path

import numpy as np

from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration
from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.reference.call import ReferenceCall


def batch_norm_argument_pool(context: dict) -> ArgumentPool:
    n, ctype = context["name"], context["input_dtype"]
    arrays = tuple(Declaration(f"{n}_{key}", ctype, ArrayLiteral(context[f"{key}_array"]), array=True)
                   for key in ("scale", "bias"))
    return tensor_case_pool(context, {"scale": f"{n}_scale", "bias": f"{n}_bias", "layout": context["layout"]},
                            dims=("input_dims",), extra_header=arrays, output_count=dims_count(context["input_dims"]))


class OpBatchNorm(OperationBase):
    """Generate float batch normalization parity tests."""

    def allow_no_tflite(self) -> bool:
        return True

    def uses_reference(self) -> bool:
        return True

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
        output_data = self.reference_golden(ReferenceCall(
            "batch_norm_f16" if activation_dtype == "FP16" else "batch_norm_f32", {"unused": 0},
            {"input": input_data, "scale": scale, "bias": bias}, {"output": input_shape}))
        output_data, nonfinite_context = self.apply_nonfinite_policy(
            output_data, reference=lambda ops: self.reference_probe([ops[0], scale, bias]), inputs=[input_data]
        )

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
        context.update(nonfinite_context)

        self.render_harness_case(
            output_dir, stem="batch_norm", context=context, pool=batch_norm_argument_pool(context),
            validation_key="NNSupportFunctions/batch_norm/batch_norm.c.j2", label="BatchNorm", operator="BatchNorm",
            sidecar=True,
        )
