"""Float activation operation (arm_nn_activation_f32/f16): goldens from the C reference.

Relu, relu6, leaky relu and none are exact in the kernel, so their golden is the exact result rounded once.
On the shipped build sigmoid, tanh and hardswish follow the kernel's own arithmetic on the case's target
(kernel_faithful: nn_activation_f32_cmsis per float32 target, nn_activation_f16_mve and tanh_lut_mve_f16 on
Helium), compared at PyTorch's tolerances; the exact result stays the oracle (kf.check_close_to_exact). Every
other build, and float16 off Helium, keeps the exact golden widened by the activation's guard budget. A
descriptor's tightened comparison wins on every build (kf.comparison_context).
"""

from pathlib import Path

import numpy as np

from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness.simple import tensor_case_pool


class OpNNActivationFloat(OperationBase):
    """Generate float activation parity tests."""

    def uses_reference(self) -> bool:
        return True

    def _activation_symbol(self) -> str:
        return str(self.desc["activation_type"]).upper()

    def _kernel_call(self, activation_type: str, activation_dtype: str, input_data: np.ndarray, act_param: float,
                     exact):
        """The golden's reference call, and whether it is the kernel's own arithmetic on this build.

        Relu, relu6, leaky relu and none round at most one product, so the exact call is the kernel's result on
        every build. On the shipped build sigmoid, tanh and hardswish follow the kernel: float32 on each float
        target (Helium, scalar FPU, soft float), float16 on Helium. Every other case keeps the exact golden,
        widened by the activation's guard budget (kf.activation_atol)."""
        from helia_core_tester.generation.reference.abi import float_activation_code
        from helia_core_tester.generation.reference.call import ReferenceCall

        code = float_activation_code(activation_type)
        if activation_type not in kf.APPROXIMATED_ACTIVATIONS:
            return exact, True
        target = kf.float_route(self) if activation_dtype == "FP32" else None
        if target is not None:
            return kf.nn_activation_f32_cmsis_call(input_data, code, act_param, target), True
        if activation_dtype == "FP16" and kf.mve_float16_route(self):
            if activation_type == "ARM_NN_FLT_ACT_TANH":
                return ReferenceCall("tanh_lut_mve_f16", {"unused": 0}, dict(exact.inputs), dict(exact.output_shapes),
                                     quant=exact.quant), True
            return kf.nn_activation_f16_mve_call(input_data, code, act_param), True
        return exact, False

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
        exact = ReferenceCall(
            "nn_activation_f16" if activation_dtype == "FP16" else "nn_activation_f32",
            {"activation_type": float_activation_code(activation_type), "act_param": act_param},
            {"input": np.ascontiguousarray(input_data)}, {"output": input_shape},
            quant={"activation_type": activation_type},
        )
        call, faithful = self._kernel_call(activation_type, activation_dtype, input_data, act_param, exact)
        output_data = self.reference_golden(call)
        if call is not exact:
            kf.check_close_to_exact(name, call, exact, kf.activation_atol(input_data, activation_type))

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
        bound = 0.0 if faithful else kf.activation_atol(input_data, activation_type)
        context.update(kf.comparison_context(self, float_dtype, bound))

        self.render_harness_case(
            output_dir, stem="nn_activation_float", context=context,
            pool=tensor_case_pool(context, {"size": context["size"], "type": context["activation_symbol"],
                                            "act_param": context["act_param_literal"]}, dims=(), output_count=str(context["size"])),
            validation_key="ActivationFunctions/nn_activation_float/nn_activation_float.c.j2", label="NNActivationFloat",
            operator="NNActivationFloat", sidecar=True,
        )
