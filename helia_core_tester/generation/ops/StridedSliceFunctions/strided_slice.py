"""
StridedSlice operation implementation.
"""

from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.reference import policy

_INT_DTYPES = {"int8_t": (np.int8, -128, 127), "int16_t": (np.int16, -32768, 32767)}


def strided_slice_output_shape(input_shape, begin, end, strides, shrink_axis_mask) -> Tuple[List[int], List[int]]:
    """TFLite StridedSlice extents (no begin/end masks): the per-axis counts with shrunk axes as 1,
    and the output shape without them."""
    counts, shape = [], []
    for d, size in enumerate(input_shape):
        b, e, s = int(begin[d]), int(end[d]), int(strides[d])
        lo, hi = (0, size) if s > 0 else (-1, size - 1)
        b = min(max(b + size if b < 0 else b, lo), hi)
        if shrink_axis_mask & (1 << d):
            counts.append(1)
            continue
        e = min(max(e + size if e < 0 else e, lo), hi)
        n = len(range(b, e, s))
        counts.append(n)
        shape.append(n)
    return counts, shape


class OpStridedSlice(OperationBase):
    """
    StridedSlice operation.
    """

    def uses_reference(self) -> bool:
        return True

    def _select_cmsis_strided_slice_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for StridedSlice operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = str(self.desc.get('activation_dtype', 'S8')).upper()
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_strided_slice_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_strided_slice_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        elif activation_dtype == 'S32':
            return {
                'kernel_fn': 'arm_strided_slice_s32',
                'input_c_type': 'int32_t',
                'output_c_type': 'int32_t'
            }
        elif activation_dtype == 'FP32':
            return {
                'kernel_fn': 'arm_strided_slice_f32',
                'input_c_type': 'float',
                'output_c_type': 'float'
            }
        elif activation_dtype == 'FP16':
            return {
                'kernel_fn': 'arm_strided_slice_f16',
                'input_c_type': 'float16_t',
                'output_c_type': 'float16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported StridedSlice dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for StridedSlice operation.
        """
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_info = self._select_cmsis_strided_slice_kernel()
        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        rank = len(input_shape)
        shrink_axis_mask = int(self.desc.get('shrink_axis_mask', 0))
        begin = list(self.desc.get('begin', [0, 0, 0, 0]))
        end = self.desc.get('end', None)
        strides = list(self.desc.get('strides', [1, 1, 1, 1]))
        end = list(end) if end is not None else list(input_shape)
        if not (len(begin) == len(end) == len(strides) == rank):
            raise ValueError(f"{name}: begin/end/strides must each have {rank} entries")

        counts, output_shape = strided_slice_output_shape(input_shape, begin, end, strides, shrink_axis_mask)

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        # The kernel keeps every axis, shrunk ones at extent 1.
        output_dims = builder.nhwc_to_cmsis_dims(tuple(counts))

        # The kernel takes begin with negative indices resolved, and NHWC-padded begin and strides.
        begin_normalized = [input_shape[i] + b if b < 0 else b for i, b in enumerate(begin)]
        begin_dims = builder.nhwc_to_cmsis_dims((begin_normalized + [0] * 4)[:4])
        stride_dims = builder.nhwc_to_cmsis_dims((strides + [1] * 4)[:4])

        c_type = kernel_info["input_c_type"]
        if c_type in ("float16_t", "float"):
            float_dtype = np.float16 if c_type == "float16_t" else np.float32
            input_q = self._sample_uniform(input_shape, dtype=float_dtype)
        elif c_type == "int32_t":
            input_q = self._seeded_rng().integers(-1000, 1001, size=input_shape, dtype=np.int32)
        elif c_type in _INT_DTYPES:
            kind = "s16" if c_type == "int16_t" else "s8"
            input_data = self._seeded_rng().uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
            input_q = policy.quantize(input_data, self.activation_quant("input", (-1.0, 1.0), kind))
        else:
            raise ValueError(f"Unsupported input_c_type: {c_type}")

        output_data = self.reference_golden(ReferenceCall(
            "strided_slice",
            {"begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0,
             "shrink_axis_mask": shrink_axis_mask},
            {"input": input_q, "begin": np.array(begin, dtype=np.int32), "end": np.array(end, dtype=np.int32),
             "strides": np.array(strides, dtype=np.int32)},
            {"output": tuple(output_shape)},
        ))

        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'begin_dims': begin_dims,
            'stride_dims': stride_dims,
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        if c_type in ("float16_t", "float"):
            context["validation_mode"] = "float"

        self.render_harness_case(
            Path(output_dir), stem="strided_slice", context=context, pool=strided_slice_argument_pool(context),
            validation_key="StridedSliceFunctions/strided_slice/strided_slice.c.j2", label="StridedSlice", operator="StridedSlice",
        )

        print(f"Generated C/H files and CMakeLists.txt for {name}")


from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool  # noqa: E402


def strided_slice_argument_pool(context):
    return tensor_case_pool(context, {}, dims=("input_dims", "output_dims", "begin_dims", "stride_dims"),
                            output_count=dims_count(context["output_dims"]))
