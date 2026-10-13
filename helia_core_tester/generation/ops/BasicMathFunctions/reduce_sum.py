"""
ReduceSum operation implementation.

Float-only: ns-cmsis-nn provides arm_reduce_sum_f32/f16 (no quantized sum
kernels exist; quantized SUM is lowered onto arm_mean_* by the compiler).
On the shipped build a Helium float16 case's golden is the kernel's own float32
accumulation (kernel_faithful); every other case's is the exact sum rounded once,
compared within kernel_faithful.float32_accumulation_bound.
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpReduceSum(OperationBase):
    """
    ReduceSum operation (FP32/FP16).
    """

    def uses_reference(self) -> bool:
        return True

    def _select_cmsis_reduce_sum_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for ReduceSum operation.
        """
        activation_dtype = self.tensor_dtype("input", default="FP32")

        if activation_dtype == 'FP32':
            return {
                'kernel_fn': 'arm_reduce_sum_f32',
                'input_c_type': 'float',
                'output_c_type': 'float',
                'float_kernel': True,
            }
        elif activation_dtype == 'FP16':
            return {
                'kernel_fn': 'arm_reduce_sum_f16',
                'input_c_type': 'float16_t',
                'output_c_type': 'float16_t',
                'float_kernel': True,
            }
        else:
            raise NotImplementedError(
                f"Unsupported ReduceSum dtype: {activation_dtype} (float-only kernels)")

    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for ReduceSum operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        from helia_core_tester.generation.reference.call import ReferenceCall

        name = self.desc['name']

        kernel_info = self._select_cmsis_reduce_sum_kernel()
        float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32

        input_shape = tuple(self.desc['input_shape'])

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)

        axes = self.desc.get('axes', [1, 2])
        if not isinstance(axes, list):
            axes = [axes]

        # Keep CMSIS output dims 4D even when TFLite output rank is reduced
        axis_dims_cmsis = builder.build_reduce_axis_dims(len(input_shape), axes)
        output_dims = builder.build_reduce_output_dims(
            input_shape=input_shape,
            axes=axes,
            keepdims=bool(self.desc.get('keepdims', True))
        )

        input_q = self._sample_uniform(input_shape, dtype=float_dtype)

        # Helium float16: the kernel's float32 accumulation; elsewhere the exact sum rounded once.
        normalized_axes = builder.normalize_reduction_axes(len(input_shape), axes)
        faithful = float_dtype == np.float16 and kf.mve_float16_route(self)
        if faithful:
            # The kernel's own float32 accumulation order on Helium, rounded as the kernel rounds it.
            route = kf.reduce_f16_route("sum", input_q.shape, normalized_axes)
            output_data = self.reference_golden(
                kf.reduce_f16_f32acc_call(input_q, normalized_axes, mean=False, route=route))
            rows, weights = kf.reduce_operands(input_q, normalized_axes, mean=False)
            kf.check_within_bound(name, output_data, rows, weights, None, float_dtype)
            comparison = kf.comparison_context(self, float_dtype)
        else:
            # The exact sum rounded once; the kernel sums the reduced elements in float32, in an order its
            # build picks, which float32_accumulation_bound covers.
            exact = ReferenceCall(
                "reduce_sum_f16" if float_dtype == np.float16 else "reduce_sum_f32",
                {"axis_mask": sum(1 << a for a in normalized_axes)},
                {"input": np.ascontiguousarray(input_q)},
                {"output": tuple(1 if i in normalized_axes else int(d) for i, d in enumerate(input_shape))})
            output_data = self.reference_golden(exact)
            count = int(np.prod([input_q.shape[a] for a in normalized_axes]))
            comparison = kf.comparison_context(
                self, float_dtype, *kf.float32_accumulation_bound(kf.abs_magnitude(exact), count, float_dtype))
        output_data, nonfinite_context = self.apply_nonfinite_policy(
            output_data, reference=self.reference_probe, inputs=[input_q]
        )
        nonfinite_context = {**comparison, **nonfinite_context}

        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'axis_dims': axis_dims_cmsis,
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'float_kernel': True,
            'validation_mode': 'float',
        }
        context.update(nonfinite_context)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'ReduceSum'),
            'operator_name': 'reduce_sum',
        }
        self._write_op_outputs(
            output_dir,
            "reduce_sum",
            "BasicMathFunctions/reduce_sum/reduce_sum.h.j2",
            "BasicMathFunctions/reduce_sum/reduce_sum.c.j2",
            context,
            cmake_context,
        )


from helia_core_tester.generation.harness.registry import harness_pool  # noqa: E402
from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool  # noqa: E402


@harness_pool("BasicMathFunctions/reduce_sum/reduce_sum.c.j2", label="ReduceSum")
def reduce_sum_argument_pool(context):
    return tensor_case_pool(context, {}, dims=("input_dims", "output_dims", "axis_dims"),
                            output_count=dims_count(context["output_dims"]))
