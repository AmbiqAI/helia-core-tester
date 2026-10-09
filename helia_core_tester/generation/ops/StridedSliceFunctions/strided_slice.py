"""
StridedSlice operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpStridedSlice(OperationBase):
    """
    StridedSlice operation.
    """

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
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        kernel_info = self._select_cmsis_strided_slice_kernel()
        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        rank = len(input_shape)
        if not 1 <= rank <= 4:
            raise ValueError(f"{name}: StridedSlice supports rank 1..4, got {input_shape}")

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        shrink_axis_mask = int(self.desc.get('shrink_axis_mask', 0))

        begin = list(self.desc.get('begin', [0, 0, 0, 0]))
        end = self.desc.get('end', None)
        strides = list(self.desc.get('strides', [1, 1, 1, 1]))
        if len(begin) < 4:
            begin = begin + [0] * (4 - len(begin))
        if len(strides) < 4:
            strides = strides + [1] * (4 - len(strides))
        if any(int(st) == 0 for st in strides[:rank]):
            raise ValueError(f"{name}: zero stride in {strides}")

        begin_normalized = [input_shape[i] + b if b < 0 else b for i, b in enumerate(begin[:rank])]
        begin_normalized += [0] * (4 - len(begin_normalized))
        begin_dims = builder.nhwc_to_cmsis_dims(begin_normalized[:4])
        stride_dims = builder.nhwc_to_cmsis_dims(strides[:4])

        end_resolved = list(end) if end is not None else list(input_shape)
        end_resolved += list(input_shape[len(end_resolved):])
        # A shrunk axis takes the single element at begin, whatever end says.
        slices = tuple(
            slice(begin_normalized[i], begin_normalized[i] + 1, 1) if shrink_axis_mask & (1 << i)
            else slice(begin_normalized[i], end_resolved[i], strides[i])
            for i in range(rank)
        )

        # Generate input data and quantize when required.
        if kernel_info["input_c_type"] in ("float16_t", "float"):
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            input_q = self._sample_uniform(input_shape, dtype=float_dtype)
        else:
            rng_state = self.rng.__getstate__()
            self.rng = np.random.default_rng(self.seed)
            if kernel_info["input_c_type"] == "int32_t":
                input_q = self.rng.integers(-1000, 1001, size=input_shape, dtype=np.int32)
            else:
                from helia_core_tester.generation.reference import policy

                kind = {"int8_t": "s8", "int16_t": "s16"}.get(kernel_info["input_c_type"])
                if kind is None:
                    raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
                input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
                quant = policy.descriptor_quant(self.desc.get("quantization", {}).get("input"), kind)
                input_q = policy.quantize(input_data, quant or policy.activation_quant(input_data, kind))
            self.rng.__setstate__(rng_state)

        sliced = input_q[slices]
        output_dims = builder.nhwc_to_cmsis_dims(tuple(sliced.shape))
        if sliced.size == 0:
            raise ValueError(f"{name}: slice {slices} of {input_shape} is empty")
        output_data = np.ascontiguousarray(sliced).astype(input_q.dtype)

        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'begin_dims': begin_dims,
            'stride_dims': stride_dims,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        if kernel_info["input_c_type"] in ("float16_t", "float"):
            context["validation_mode"] = "float"
        
        # Render templates
        self.render_harness_case(
            Path(output_dir), stem="strided_slice", context=context, pool=strided_slice_argument_pool(context),
            validation_key="StridedSliceFunctions/strided_slice/strided_slice.c.j2", label="StridedSlice", operator="StridedSlice",
        )
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")


from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool  # noqa: E402


def strided_slice_argument_pool(context):
    return tensor_case_pool(context, {}, dims=("input_dims", "output_dims", "begin_dims", "stride_dims"),
                            output_count=dims_count(context["output_dims"]))
