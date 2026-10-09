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
    
    def _is_float_kernel(self) -> bool:
        return self.tensor_dtype("input", default="S8") in ("FP32", "FP16")

    def uses_reference(self) -> bool:
        return not self._is_float_kernel()

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
        from helia_core_tester.generation.reference import bindings as b
        from helia_core_tester.generation.reference import params as ref_params
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference.case import ReferenceCall

        name = self.desc['name']
        kernel_info = self._select_cmsis_mean_kernel()
        if kernel_info.get('float_kernel'):
            self._generate_float_c_files(output_dir, kernel_info)
            return
        kind = "s16" if kernel_info["input_c_type"] == "int16_t" else "s8"

        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        axes = self.desc.get('axes', [1, 2])
        if not isinstance(axes, list):
            axes = [axes]
        keepdims = bool(self.desc.get('keepdims', True))
        # Keep CMSIS output dims 4D even when the TFLite output rank is reduced (keepdims=false).
        normalized_axes = builder.normalize_reduction_axes(len(input_shape), axes)
        axis_dims_cmsis = builder.build_reduce_axis_dims(len(input_shape), normalized_axes)
        output_dims = builder.build_reduce_output_dims(input_shape=input_shape, axes=normalized_axes, keepdims=keepdims)
        reduction_size = int(np.prod([input_shape[a] for a in normalized_axes]))
        output_shape = tuple(
            (1 if keepdims else None) if i in normalized_axes else d for i, d in enumerate(input_shape)
        )
        output_shape = tuple(d for d in output_shape if d is not None) or (1,)

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        self.rng.__setstate__(rng_state)

        in_quant = self.activation_quant("input", input_data, kind)
        out_quant = self.activation_quant("output", input_data.mean(axis=tuple(normalized_axes)), kind)
        input_q = policy.quantize(input_data, in_quant)
        input_zp, output_zp = in_quant.zero_point, out_quant.zero_point

        # TFLM prepares QuantizeMultiplier(si / so) and folds 1/count inside the kernel;
        # the CMSIS kernel takes the folded pair, so fold it the same way.
        base_mult, base_shift = ref_params.mean_params(in_quant.scale, out_quant.scale)
        out_mult, out_shift = b.get_bindings().mean_fold(base_mult, base_shift, reduction_size)
        call = ReferenceCall(
            f"mean_{kind}",
            {"input_zero_point": input_zp, "output_zero_point": output_zp, "multiplier": base_mult,
             "shift": base_shift, "keep_dims": int(keepdims), "axes": [int(a) for a in normalized_axes]},
            {"input": input_q}, output_shape, input_q.dtype.name,
            quant={"input": in_quant.to_json(), "output": out_quant.to_json()},
        )
        output_data = self.reference_golden(call)

        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'axis_dims': axis_dims_cmsis,
            # arm_mean_s8/s16 accumulate sum + input_offset * count, so the offset is -zp.
            'input_offset': -int(input_zp),
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
        dividing by the reduction count before the single store. Goldens
        accumulate in float32 for both dtypes and divide in float32,
        matching the kernels' documented semantics (the f16 kernel widens
        to float32 and rounds once after the divide).
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

        reduction_count = 1
        for axis in normalized_axes:
            reduction_count *= int(input_shape[axis])

        # Golden with float32 accumulation and a float32 divide for both
        # dtypes (single final rounding for f16).
        def reference(operands):
            return (
                np.sum(
                    operands[0].astype(np.float32),
                    axis=tuple(normalized_axes),
                    keepdims=True,
                )
                / np.float32(reduction_count)
            ).astype(float_dtype)

        output_data = reference([input_q])
        output_data, nonfinite_context = self.apply_nonfinite_policy(
            output_data, reference=reference, inputs=[input_q]
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
