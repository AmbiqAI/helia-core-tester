"""
Mean operation implementation for Helia-Core Tester.
"""

import numpy as np
from pathlib import Path
from typing import Dict
from helia_core_tester.generation.ops._shared.base import OperationBase  


class OpMean(OperationBase):
    """
    Mean operation.
    """
    
    def uses_reference(self) -> bool:
        return True

    def _is_float_kernel(self) -> bool:
        return self.tensor_dtype("input", default="S8") in ("FP32", "FP16")

    def _axes(self, rank: int):
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        axes = self.desc.get('axes', [1, 2])
        if not isinstance(axes, list):
            axes = [axes]
        return TemplateContextBuilder.normalize_reduction_axes(rank, axes)

    def _select_cmsis_mean_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Mean operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.tensor_dtype("input", default="S8")
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_mean_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_mean_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        elif activation_dtype == 'FP32':
            return {
                'kernel_fn': 'arm_nn_mean_f32',
                'input_c_type': 'float',
                'output_c_type': 'float',
                'float_kernel': True,
            }
        elif activation_dtype == 'FP16':
            return {
                'kernel_fn': 'arm_nn_mean_f16',
                'input_c_type': 'float16_t',
                'output_c_type': 'float16_t',
                'float_kernel': True,
            }
        else:
            raise NotImplementedError(f"Unsupported Mean dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Mean operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference.bindings import get_bindings
        from helia_core_tester.generation.reference.call import ReferenceCall

        name = self.desc['name']
        kernel_info = self._select_cmsis_mean_kernel()
        if kernel_info.get('float_kernel'):
            self._generate_float_c_files(output_dir, kernel_info)
            return

        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        normalized_axes = self._axes(len(input_shape))
        axis_dims_cmsis = builder.build_reduce_axis_dims(len(input_shape), normalized_axes)
        keepdims = bool(self.desc.get('keepdims', True))
        output_dims = builder.build_reduce_output_dims(input_shape=input_shape, axes=normalized_axes, keepdims=keepdims)
        out_shape = tuple(1 if i in normalized_axes else d for i, d in enumerate(input_shape))
        count = int(np.prod([input_shape[a] for a in normalized_axes]))

        kind = "s16" if kernel_info["input_c_type"] == "int16_t" else "s8"
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        self.rng.__setstate__(rng_state)
        # The [-1, 1] the converter calibrated on; a mean stays inside its input's range.
        in_quant = self.activation_quant("input", (-1.0, 1.0), kind)
        input_q = policy.quantize(input_data, in_quant)
        real = policy.dequantize(input_q, in_quant).mean(axis=tuple(normalized_axes), keepdims=True)
        out_quant = self.activation_quant("output", policy.data_range(real), kind)
        folded = get_bindings().prepare("mean_prepare", {
            "input_scale": in_quant.scale, "output_scale": out_quant.scale, "count": count})
        output_data = self.reference_golden(ReferenceCall(
            f"mean_{kind}",
            {"axis_mask": sum(1 << a for a in normalized_axes), "input_zero_point": in_quant.zero_point,
             "output_zero_point": out_quant.zero_point, **folded},
            {"input": np.ascontiguousarray(input_q)}, {"output": out_shape},
            quant={"input": in_quant.to_json(), "output": out_quant.to_json()}))
        input_zp, output_zp = in_quant.zero_point, out_quant.zero_point
        out_mult, out_shift = folded["multiplier"], folded["shift"]
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)

        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'axis_dims': axis_dims_cmsis,
            # arm_mean_* add input_offset * count to the raw sum: the negated zero point.
            'input_offset': int(-input_zp),
            'out_offset': int(output_zp),
            'out_mult': int(out_mult),
            'out_shift': int(out_shift),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Mean'),
            'operator_name': 'mean'
        }
        self._write_op_outputs(
            output_dir,
            "mean",
            "BasicMathFunctions/mean/mean.h.j2",
            "BasicMathFunctions/mean/mean.c.j2",
            context,
            cmake_context,
        )
        

    def _generate_float_c_files(self, output_dir: Path, kernel_info: Dict[str, str]) -> None:
        """
        Generate C/H files for arm_nn_mean_f32/f16 (ns-cmsis-nn #414/#412).

        Mirrors OpReduceSum's float path: same call shape (4D NHWC input
        dims + 4D binary axis mask + 4D output dims), with the kernel
        dividing by the reduction count before the single store. The golden
        is the exact mean rounded once (the C reference).
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32

        input_shape = tuple(self.desc['input_shape'])

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)

        axes = self.desc.get('axes', [1, 2])
        if not isinstance(axes, list):
            axes = [axes]

        # Keep CMSIS output dims 4D even when TFLite output rank is reduced
        normalized_axes = builder.normalize_reduction_axes(len(input_shape), axes)
        axis_dims_cmsis = builder.build_reduce_axis_dims(len(input_shape), normalized_axes)
        output_dims = builder.build_reduce_output_dims(
            input_shape=input_shape,
            axes=normalized_axes,
            keepdims=bool(self.desc.get('keepdims', True))
        )

        input_q = self._sample_uniform(input_shape, dtype=float_dtype)

        from helia_core_tester.generation.reference.call import ReferenceCall

        output_data = self.reference_golden(ReferenceCall(
            "mean_f16" if float_dtype == np.float16 else "mean_f32",
            {"axis_mask": sum(1 << a for a in normalized_axes)},
            {"input": np.ascontiguousarray(input_q)},
            {"output": tuple(1 if i in normalized_axes else d for i, d in enumerate(input_shape))}))
        output_data, nonfinite_context = self.apply_nonfinite_policy(
            output_data, reference=self.reference_probe, inputs=[input_q]
        )

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
            'operator': self.desc.get('operator', 'Mean'),
            'operator_name': 'mean',
        }
        self._write_op_outputs(
            output_dir,
            "mean",
            "BasicMathFunctions/mean/mean_float.h.j2",
            "BasicMathFunctions/mean/mean_float.c.j2",
            context,
            cmake_context,
        )


from helia_core_tester.generation.harness.registry import harness_pool  # noqa: E402
from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool  # noqa: E402

_REDUCE_DIMS = ("input_dims", "output_dims", "axis_dims")


@harness_pool("BasicMathFunctions/mean/mean.c.j2", label="Mean")
def mean_argument_pool(context):
    return tensor_case_pool(context, {"input_offset": context["input_offset"], "out_offset": context["out_offset"],
                                      "out_mult": context["out_mult"], "out_shift": context["out_shift"]},
                            dims=_REDUCE_DIMS, output_count=dims_count(context["output_dims"]))


@harness_pool("BasicMathFunctions/mean/mean_float.c.j2", label="Mean")
def mean_float_argument_pool(context):
    return tensor_case_pool(context, {}, dims=_REDUCE_DIMS, output_count=dims_count(context["output_dims"]))
