"""
Concatenation operation implementation.
"""

from typing import Dict, Any, List
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpConcatenation(OperationBase):
    """
    Concatenation operation - concatenates tensors along an axis.
    """
    
    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Concatenation uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_concat_op

        activation_dtype = self.tensor_dtype("input", default=str(self.desc.get("activation_dtype", "S8")))
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        elif activation_dtype == "S32":
            dtype = "int32"
        elif activation_dtype == "FP32":
            dtype = "float32"
        elif activation_dtype == "FP16":
            dtype = "float16"
        else:
            raise NotImplementedError(f"Unsupported Concatenation dtype: {activation_dtype}")

        input_shapes = []
        if "input_1_shape" in self.desc:
            i = 1
            while f"input_{i}_shape" in self.desc:
                input_shapes.append(tuple(self.desc[f"input_{i}_shape"]))
                i += 1
        elif "input_shape" in self.desc:
            input_shapes.append(tuple(self.desc["input_shape"]))

        axis = int(self.desc.get("axis", -1))

        model_bytes = build_concat_op(
            input_shapes=input_shapes,
            axis=axis,
            dtype=dtype,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)
    
    def _axis_call_style(self, axis: int, input_rank: int) -> str:
        if axis < 0:
            axis += input_rank
        axis_map = {
            2: "axis_x",
            1: "axis_y",
            3: "axis_z",
            0: "axis_w",
        }
        if input_rank != 4 or axis not in axis_map:
            raise NotImplementedError("Float Concatenation descriptors require rank-4 NHWC shapes")
        return axis_map[axis]

    def _select_cmsis_concatenation_kernel(self, input_rank: int) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Concatenation operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.tensor_dtype("input", default=str(self.desc.get('activation_dtype', 'S8')))
        
        if activation_dtype == 'S8':
            call_style = str(self.desc.get("hint", {}).get("call_style", "")).lower()
            if call_style == "axis_x":
                return {
                    'kernel_fn': 'arm_concatenation_s8_x',
                    'input_c_type': 'int8_t',
                    'output_c_type': 'int8_t'
                }
            if call_style == "axis_y":
                return {
                    'kernel_fn': 'arm_concatenation_s8_y',
                    'input_c_type': 'int8_t',
                    'output_c_type': 'int8_t'
                }
            if call_style == "axis_z":
                return {
                    'kernel_fn': 'arm_concatenation_s8_z',
                    'input_c_type': 'int8_t',
                    'output_c_type': 'int8_t'
                }
            if call_style == "axis_w":
                return {
                    'kernel_fn': 'arm_concatenation_s8_w',
                    'input_c_type': 'int8_t',
                    'output_c_type': 'int8_t'
                }
            return {
                'kernel_fn': 'arm_concatenation_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_concatenation_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        elif activation_dtype == 'S32':
            return {
                'kernel_fn': 'arm_concatenation_s32',
                'input_c_type': 'int32_t',
                'output_c_type': 'int32_t'
            }
        elif activation_dtype in {'FP32', 'FP16'}:
            call_style = str(self.desc.get("hint", {}).get("call_style", "")).lower()
            if call_style not in {"axis_x", "axis_y", "axis_z", "axis_w"}:
                call_style = self._axis_call_style(int(self.desc.get("axis", -1)), input_rank)
            suffix = "f16" if activation_dtype == "FP16" else "f32"
            c_type = "float16_t" if activation_dtype == "FP16" else "float"
            return {
                'kernel_fn': f'arm_concatenation_{suffix}_{call_style[-1]}',
                'input_c_type': c_type,
                'output_c_type': c_type,
                'call_style': call_style,
            }
        else:
            raise NotImplementedError(f"Unsupported Concatenation dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Concatenation operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Build input shapes from descriptor
        input_shapes = []
        if "input_1_shape" in self.desc:
            i = 1
            while f"input_{i}_shape" in self.desc:
                input_shapes.append(tuple(self.desc[f"input_{i}_shape"]))
                i += 1
        elif "input_shape" in self.desc:
            input_shapes.append(tuple(self.desc["input_shape"]))
        num_inputs = len(input_shapes)
        output_shape = None

        # Select CMSIS kernel + types
        input_rank = len(input_shapes[0]) if input_shapes else 0
        kernel_info = self._select_cmsis_concatenation_kernel(input_rank)
        
        is_const_variant = 'const' in name.lower()
        
        builder = TemplateContextBuilder()
        
        # Extract axis from descriptor (default to -1 for last dimension)
        axis = self.desc.get('axis', -1)
        # Convert negative axis based on actual rank (NHWC includes batch)
        input_rank = len(input_shapes[0]) if input_shapes and input_shapes[0] is not None else (len(output_shape) if output_shape else 0)
        if axis < 0:
            axis = input_rank + axis  # -1 -> last dimension
        
        # For const variants, we need to create a const input shape
        const_shape = None
        if is_const_variant and len(input_shapes) == 1:
            var_shape = input_shapes[0]
            # Calculate const input shape based on axis
            const_shape = list(var_shape)
            if 'scalar' in name.lower():
                # Const scalar: for axis=-1 (channel), shape is [1, H, W, 1]
                if axis == 3:  # Channel axis
                    const_shape = [1, var_shape[1], var_shape[2], 1]
                else:
                    const_shape[axis] = 1
            elif 'vector' in name.lower():
                # Const vector: for axis=2 (width), shape is [1, H, 1, C]
                if axis == 2:  # Width axis
                    const_shape = [1, var_shape[1], 1, var_shape[3]]
                else:
                    const_shape[axis] = 1
            else:
                # Default: same as scalar
                const_shape[axis] = 1
            
            input_shapes.append(tuple(const_shape))
            num_inputs = 2
            const_shape = tuple(const_shape)

        # Recompute output shape from inputs to avoid relying on potentially wrong op output metadata
        if input_shapes and all(shape is not None for shape in input_shapes):
            output_shape = list(input_shapes[0])
            output_shape[axis] = sum(int(shape[axis]) for shape in input_shapes)
            output_shape = tuple(output_shape)

        # Convert shapes to CMSIS dims
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        output_rank = len(output_shape)
        
        # Calculate input_concat_dims (dimension along axis for each input)
        input_concat_dims = []
        for input_shape in input_shapes:
            if axis < len(input_shape):
                input_concat_dims.append(int(input_shape[axis]))
            else:
                input_concat_dims.append(1)
        
        call_style = kernel_info.get("call_style") or str(self.desc.get("hint", {}).get("call_style", "")).lower()

        # Generate input data
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        input_q_list = []
        input_array_strs = []

        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        elif kernel_info["input_c_type"] == "int32_t":
            np_in_dtype = np.int32
            qmin, qmax = -2147483648, 2147483647
        elif kernel_info["input_c_type"] == "float":
            np_in_dtype = np.float32
            qmin, qmax = None, None
        elif kernel_info["input_c_type"] == "float16_t":
            np_in_dtype = np.float16
            qmin, qmax = None, None
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        # Generate variable input(s)
        for i, input_shape in enumerate(input_shapes):
            if call_style in ("axis_x", "axis_y", "axis_z", "axis_w"):
                # Use WZYX layout so X is the fastest-changing dimension in memory.
                wzyx_shape = (input_shape[0], input_shape[3], input_shape[1], input_shape[2])
                if np_in_dtype == np.int32:
                    input_q = self.rng.integers(-32, 32, size=wzyx_shape, dtype=np.int32)
                elif np.issubdtype(np.dtype(np_in_dtype), np.floating):
                    input_q = self.rng.uniform(-2.0, 2.0, size=wzyx_shape).astype(np_in_dtype)
                else:
                    input_q = self.rng.integers(qmin, qmax + 1, size=wzyx_shape, dtype=np_in_dtype)
            else:
                if np_in_dtype == np.int32:
                    input_q = self.rng.integers(-32, 32, size=input_shape, dtype=np.int32)
                elif np.issubdtype(np.dtype(np_in_dtype), np.floating):
                    input_q = self.rng.uniform(-2.0, 2.0, size=input_shape).astype(np_in_dtype)
                else:
                    input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
            input_q_list.append(input_q)
            input_array_strs.append(builder.format_array_as_c_literal(input_q))
        
        # Generate const input if needed
        const_input_q = None
        if is_const_variant:
            # Prefer using constant tensor data from the model if available
            const_shape = input_shapes[-1]
            const_value = 3
            if call_style in ("axis_x", "axis_y", "axis_z", "axis_w"):
                const_shape_wzyx = (const_shape[0], const_shape[3], const_shape[1], const_shape[2])
                const_input_q = np.full(const_shape_wzyx, const_value, dtype=np_in_dtype)
            else:
                const_input_q = np.full(const_shape, const_value, dtype=np_in_dtype)
            input_q_list[-1] = const_input_q
            input_array_strs[-1] = builder.format_array_as_c_literal(const_input_q)
        
        self.rng.__setstate__(rng_state)

        # Compute expected output (no arithmetic, just concatenate)
        if call_style == "axis_x":
            output_data = np.concatenate(input_q_list, axis=3)
        elif call_style == "axis_y":
            output_data = np.concatenate(input_q_list, axis=2)
        elif call_style == "axis_z":
            output_data = np.concatenate(input_q_list, axis=1)
        elif call_style == "axis_w":
            output_data = np.concatenate(input_q_list, axis=0)
        else:
            output_data = np.concatenate(input_q_list, axis=axis)
        
        # Format arrays
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        input_concat_dims_array_str = builder.format_array_as_c_literal(np.array(input_concat_dims, dtype=np.int32))
        output_shape_array_str = builder.format_array_as_c_literal(np.array(output_shape, dtype=np.int32))
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'output_dims': output_dims,
            'output_rank': output_rank,
            'num_inputs': num_inputs,
            'axis': axis,
            'input_concat_dims_array': input_concat_dims_array_str,
            'output_shape_array': output_shape_array_str,
            'input_data_arrays': input_array_strs,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'call_style': call_style,
            'input_x_array': builder.format_array_as_c_literal(np.array([s[2] for s in input_shapes], dtype=np.int32)),
            'input_y_array': builder.format_array_as_c_literal(np.array([s[1] for s in input_shapes], dtype=np.int32)),
            'input_z_array': builder.format_array_as_c_literal(np.array([s[3] for s in input_shapes], dtype=np.int32)),
            'input_w_array': builder.format_array_as_c_literal(np.array([s[0] for s in input_shapes], dtype=np.int32)),
            'output_x': int(output_shape[2]),
            'output_y': int(output_shape[1]),
            'output_z': int(output_shape[3]),
            'output_w': int(output_shape[0]),
            'offsets_array': builder.format_array_as_c_literal(
                np.array(np.cumsum([0] + [int(s[axis]) for s in input_shapes[:-1]]), dtype=np.int32)
            ),
        }
        if kernel_info["output_c_type"] in {"float", "float16_t"}:
            context["validation_mode"] = "float"
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("ConcatenationFunctions/concatenation/concatenation.h.j2", context)
        h_path = includes_api_dir / f"{name}_concatenation.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("ConcatenationFunctions/concatenation/concatenation.c.j2", context)
        c_path = output_dir / f"{name}_concatenation.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Concatenation'),
            'operator_name': 'concatenation'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
