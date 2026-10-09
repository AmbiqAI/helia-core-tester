"""Float activation operation (arm_nn_activation_f32/f16): goldens from the C reference.

Each activation's exact result rounded once to the output; float16 tanh is CMSIS-NN's table with
interpolation, as the named variants tanh_lut_f16 (per-step rounding) and tanh_lut_mve_f16 (fused,
the Helium path), chosen by the generation profile.
"""

from pathlib import Path

import numpy as np

from helia_core_tester.core.cpu_targets import get_cpu_profile
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness.simple import tensor_case_pool


class OpNNActivationFloat(OperationBase):
    """Generate float activation parity tests."""

    def uses_reference(self) -> bool:
        return True

    def _activation_symbol(self) -> str:
        return str(self.desc["activation_type"]).upper()

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.reference.abi import float_activation_code
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        input_shape = tuple(int(d) for d in self.desc["input_shape"])
        activation_dtype = self.tensor_dtype("input", default="FP32")
        if activation_dtype == "FP16":
            float_dtype = np.float16
            input_dtype = output_dtype = "float16_t"
            kernel_fn = "arm_nn_activation_f16"
        elif activation_dtype == "FP32":
            float_dtype = np.float32
            input_dtype = output_dtype = "float"
            kernel_fn = "arm_nn_activation_f32"
        else:
            raise NotImplementedError(f"Unsupported NNActivationFloat dtype: {activation_dtype}")

        input_data = self._sample_uniform(
            input_shape,
            low=float(self.desc.get("input_min", -1.0)),
            high=float(self.desc.get("input_max", 1.0)),
            dtype=float_dtype,
        )
        activation_type = self._activation_symbol()
        act_param = float(self.desc.get("act_param", 0.0))
        # Select the reference by generation profile, not compiled kernel route.
        # M55-generated cases also run on the scalar fallback; off-grid rounding
        # can differ there, so existing comparison budgets remain necessary.
        if activation_type == "ARM_NN_FLT_ACT_TANH" and activation_dtype == "FP16":
            entry = "tanh_lut_mve_f16" if get_cpu_profile(self.target_cpu).has_mve else "tanh_lut_f16"
            params = {"unused": 0}
        else:
            entry = "nn_activation_f16" if activation_dtype == "FP16" else "nn_activation_f32"
            params = {"activation_type": float_activation_code(activation_type), "act_param": act_param}
        output_data = self.reference_golden(ReferenceCall(
            entry, params, {"input": np.ascontiguousarray(input_data)}, {"output": input_shape},
            quant={"activation_type": activation_type},
        ))

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "size": int(np.prod(input_shape)),
            "input_data_array": builder.format_array_as_c_literal(input_data),
            "expected_output_array": builder.format_array_as_c_literal(output_data),
            "activation_symbol": activation_type,
            "act_param_literal": builder.format_float_literal(self.desc.get("act_param", 0.0)),
            "input_dtype": input_dtype,
            "output_dtype": output_dtype,
            "kernel_fn": kernel_fn,
        }

        self.render_harness_case(
            output_dir, stem="nn_activation_float", context=context,
            pool=tensor_case_pool(context, {"size": context["size"], "type": context["activation_symbol"],
                                            "act_param": context["act_param_literal"]}, dims=(), output_count=str(context["size"])),
            validation_key="ActivationFunctions/nn_activation_float/nn_activation_float.c.j2", label="NNActivationFloat",
            operator="NNActivationFloat", sidecar=True,
        )
