"""Shared implementation for CMSIS-NN avg/max pool operators."""

from typing import Dict
from pathlib import Path
import numpy as np
import tensorflow as tf
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration
from helia_core_tester.generation.harness.faults import common_fault, struct_copy, with_fault

# Kernels ns-cmsis-nn defines but declares in no public header, and the public kernel whose
# prototype each shares.
UNDECLARED_ALIASES = {
    "arm_avg_pool_nhwc_f32": "arm_avg_pool_f32",
    "arm_avg_pool_nhwc_f16": "arm_avg_pool_f16",
    "arm_max_pool_nhwc_f32": "arm_max_pool_f32",
    "arm_max_pool_nhwc_f16": "arm_max_pool_f16",
}


def pool_argument_pool(context: Dict) -> ArgumentPool:
    """Every value a pooling case can pass to a public pooling kernel or its scratch query."""
    n = context["name"]
    float_kernel = bool(context.get("float_kernel"))
    pp = context["pool_params"]

    def dims(d: Dict) -> Dict:
        return {"n": d["n"], "h": d["h"], "w": d["w"], "c": d["c"]}

    act_min = context["pool_activation_min_literal"] if float_kernel else pp["activation_min"]
    act_max = context["pool_activation_max_literal"] if float_kernel else pp["activation_max"]
    header = [
        Declaration(f"{n}_input_dims", "cmsis_nn_dims", dims(context["input_dims"]), comment="Input dimensions"),
        Declaration(f"{n}_filter_dims", "cmsis_nn_dims", dims(context["filter_dims"]), comment="Filter dimensions"),
        Declaration(f"{n}_output_dims", "cmsis_nn_dims", dims(context["output_dims"]), comment="Output dimensions"),
        Declaration(f"{n}_pool_params", context["pool_params_type"],
                    {"stride.w": pp["stride_w"], "stride.h": pp["stride_h"], "padding.w": pp["pad_w"],
                     "padding.h": pp["pad_h"], "activation.min": act_min, "activation.max": act_max},
                    comment="Pooling parameters"),
        Declaration(f"{n}_input", context["input_dtype"], ArrayLiteral(context["input_data_array"]), array=True,
                    comment="Input data (for testing)"),
        Declaration(f"{n}_expected_output", context["output_dtype"], ArrayLiteral(context["expected_output_array"]),
                    array=True, comment="Expected output (golden)"),
    ]
    values = {
        "ctx": f"&{n}_ctx", "pool_params": f"&{n}_pool_params", "input_dims": f"&{n}_input_dims",
        "filter_dims": f"&{n}_filter_dims", "output_dims": f"&{n}_output_dims",
        "dim_dst_width": f"{n}_output_dims.w", "ch_src": f"{n}_input_dims.c",
    }
    output = context["output_dims"]
    return ArgumentPool(
        name=n, values=values, header=header, benchmark=False,
        output_count=f"({output['n']} * {output['h']} * {output['w']} * {output['c']})",
        scratch_buffer=int(context["buffer_size_max"]) > 0,
        no_scratch=not context.get("kernel_get_buffer_size_fn"),
        prototype_from=UNDECLARED_ALIASES.get(context["kernel_fn"]),
    )


def pool_fault(pool: ArgumentPool, kind: str, context: Dict) -> ArgumentPool:
    """The pool of a pooling fault case: the passing pool with the faulted argument edited."""
    n = context["name"]
    edit = common_fault(pool, kind)
    if edit is None and kind in ("zero_dim", "negative_dim"):
        edit = struct_copy(pool, kind, "input_dims", "cmsis_nn_dims", f"{n}_input_dims",
                           {"n": 0 if kind == "zero_dim" else -1})
    if edit is None:
        raise ValueError(f"{n}: no pooling fault edit for {kind!r}")
    return with_fault(pool, edit)


class PoolFamilyBase(OperationBase):
    """Shared implementation for `AvgPool` and `MaxPool` operators."""

    POOL_KIND = "AVERAGE"
    OPERATOR_NAME = "AvgPool"
    TEMPLATE_DIR = "PoolingFunctions/avg_pool"
    TEMPLATE_SUFFIX = "avg_pool"
    FAULT_KINDS = ("zero_dim", "negative_dim", "null_input", "null_output")

    def _check_fault_reachable(self, kind: str, float_kernel: bool, kernel_fn: str) -> None:
        """Reject fault kinds the selected pooling kernel does not diagnose."""
        if kind in ("null_input", "null_output") and not float_kernel:
            raise self.fault_unreachable(kind, f"{kernel_fn} does not check {kind}")

    def _int_kind(self):
        return {"S8": "s8", "S16": "s16"}.get(self.tensor_dtype("input").upper())

    def uses_reference(self) -> bool:
        # Integer cases take their golden from the TFLM reference pools; float stays on the converter path.
        return self._int_kind() is not None

    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Pooling operation."""
        input_shape = self.desc['input_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Normalize padding to lowercase
        padding = self.desc.get('padding', 'valid')
        if isinstance(padding, str):
            padding = padding.lower()
        
        if self.POOL_KIND == 'AVERAGE':
            x = tf.keras.layers.AveragePooling2D(
                pool_size=self.desc.get('pool_size', [2, 2]),
                strides=self.desc.get('strides', [2, 2]),
                padding=padding
            )(inputs)
        elif self.POOL_KIND == 'MAX':
            x = tf.keras.layers.MaxPooling2D(
                pool_size=self.desc.get('pool_size', [2, 2]),
                strides=self.desc.get('strides', [2, 2]),
                padding=padding
            )(inputs)
        else:
            raise ValueError(f"Unsupported pooling kind: {self.POOL_KIND}")
            
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)
    
    def _select_cmsis_pooling_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN pooling kernel function.
        
        Returns:
            Dictionary with kernel function name, C types, and buffer size function
        """
        activation_dtype = self.tensor_dtype("input").upper()
        hint = self.desc.get("hint", {})
        hint = hint if isinstance(hint, dict) else {}
        variant = str(hint.get("kernel_variant", "")).lower()
        use_nhwc_alias = variant == "nhwc_alias"
        if variant and not use_nhwc_alias:
            raise ValueError(f"Unsupported {self.OPERATOR_NAME} kernel_variant hint: {variant}")
        
        if self.POOL_KIND == 'MAX':
            if activation_dtype == 'S8':
                return {
                    'kernel_fn': 'arm_max_pool_s8',
                    'kernel_get_buffer_size_fn': None,  # Max pooling doesn't need buffer
                    'input_c_type': 'int8_t',
                    'output_c_type': 'int8_t',
                }
            elif activation_dtype == 'S16':
                return {
                    'kernel_fn': 'arm_max_pool_s16',
                    'kernel_get_buffer_size_fn': None,
                    'input_c_type': 'int16_t',
                    'output_c_type': 'int16_t',
                }
            elif activation_dtype == 'FP32':
                return {
                    'kernel_fn': 'arm_max_pool_nhwc_f32' if use_nhwc_alias else 'arm_max_pool_f32',
                    'kernel_get_buffer_size_fn': None,
                    'input_c_type': 'float',
                    'output_c_type': 'float',
                }
            elif activation_dtype == 'FP16':
                return {
                    'kernel_fn': 'arm_max_pool_nhwc_f16' if use_nhwc_alias else 'arm_max_pool_f16',
                    'kernel_get_buffer_size_fn': None,
                    'input_c_type': 'float16_t',
                    'output_c_type': 'float16_t',
                }
            else:
                raise NotImplementedError(f"Unsupported MaxPool dtype: {activation_dtype}")
        elif self.POOL_KIND == 'AVERAGE':
            if activation_dtype == 'S8':
                return {
                    'kernel_fn': 'arm_avgpool_s8',
                    'kernel_get_buffer_size_fn': 'arm_avgpool_s8_get_buffer_size',
                    'input_c_type': 'int8_t',
                    'output_c_type': 'int8_t',
                }
            elif activation_dtype == 'S16':
                return {
                    'kernel_fn': 'arm_avgpool_s16',
                    'kernel_get_buffer_size_fn': 'arm_avgpool_s16_get_buffer_size',
                    'input_c_type': 'int16_t',
                    'output_c_type': 'int16_t',
                }
            elif activation_dtype == 'FP32':
                return {
                    'kernel_fn': 'arm_avg_pool_nhwc_f32' if use_nhwc_alias else 'arm_avg_pool_f32',
                    'kernel_get_buffer_size_fn': None,
                    'input_c_type': 'float',
                    'output_c_type': 'float',
                }
            elif activation_dtype == 'FP16':
                return {
                    'kernel_fn': 'arm_avg_pool_nhwc_f16' if use_nhwc_alias else 'arm_avg_pool_f16',
                    'kernel_get_buffer_size_fn': None,
                    'input_c_type': 'float16_t',
                    'output_c_type': 'float16_t',
                }
            else:
                raise NotImplementedError(f"Unsupported AvgPool dtype: {activation_dtype}")
        else:
            raise ValueError(f"Unsupported pooling kind: {self.POOL_KIND}")
    
    def _pool_geometry(self, in_hw):
        from helia_core_tester.generation.reference import params as ref_params

        pool = self.desc.get('pool_size', [2, 2])
        pool_hw = (int(pool), int(pool)) if isinstance(pool, (int, float)) else (int(pool[0]), int(pool[1]))
        strides = self.desc.get('strides', [1, 1])
        stride_hw = (int(strides), int(strides)) if isinstance(strides, (int, float)) else (int(strides[0]), int(strides[1]))
        padding = str(self.desc.get('padding') or 'valid').upper()
        return pool_hw, stride_hw, ref_params.conv_geometry(padding, in_hw, pool_hw, stride_hw)

    def _int_shapes(self):
        """NHWC input and output shapes from the descriptor, as TFLite's pool prepare sizes them."""
        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        if len(input_shape) != 4 or any(d < 1 for d in input_shape):
            raise ValueError(f"{self.desc['name']}: pooling needs a positive NHWC input_shape, got {input_shape}")
        _, _, ((out_h, out_w), _, _) = self._pool_geometry(input_shape[1:3])
        return input_shape, (input_shape[0], out_h, out_w, input_shape[3])

    def _int_golden(self, kind, input_shape, output_shape, pool_hw, pool_params):
        """Quantized input and the TFLM reference pool output. Pools keep the input
        quantization on the output, so the clamp is the kernel's activation range."""
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference.case import ReferenceCall

        # Match the [-1, 1] calibration range.
        input_data = self._sample_uniform(input_shape)
        quant = self.activation_quant("input", input_data, kind)
        input_q = policy.quantize(input_data, quant)
        _, stride_hw, (_, pad_h, pad_w) = self._pool_geometry(input_shape[1:3])
        call = ReferenceCall(
            f"{'avgpool' if self.POOL_KIND == 'AVERAGE' else 'maxpool'}_{kind}",
            {
                "stride": list(stride_hw), "filter": list(pool_hw), "pad": [pad_h.pad, pad_w.pad],
                "pad_offset": [pad_h.offset, pad_w.offset],
                "act": {"min": int(pool_params["activation_min"]), "max": int(pool_params["activation_max"])},
            },
            {"input": input_q},
            output_shape,
            input_q.dtype.name,
            quant={"input": quant.to_json(), "output": quant.to_json()},
        )
        return input_q, self.reference_golden(call)

    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Pooling operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        kernel_info = self._select_cmsis_pooling_kernel()
        pooling_type = self.POOL_KIND
        kind = self._int_kind()

        if kind is None:
            tflite_path = output_dir / f"{name}.tflite"
            if not tflite_path.exists():
                raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
            op_tensors = self.load_primary_operator_tensors(str(tflite_path))
            input_shape = tuple(op_tensors['inputs'][0]['shape'])
            output_shape = tuple(op_tensors['outputs'][0]['shape'])
            output_quant = op_tensors['outputs'][0]['quantization']
        else:
            input_shape, output_shape = self._int_shapes()
            output_quant = None

        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Filter dims for pooling: pool_size from descriptor
        pool_size = self.desc.get('pool_size', [2, 2])
        if isinstance(pool_size, (int, float)):
            pool_h = pool_w = int(pool_size)
        else:
            pool_h = int(pool_size[0])
            pool_w = int(pool_size[1])
        
        filter_dims = {
            'n': 1,
            'h': pool_h,
            'w': pool_w,
            'c': 1
        }
        
        # Build pool parameters
        pool_params = builder.build_pool_params(
            self.desc,
            input_shape,
            (pool_h, pool_w),
            output_shape,
            output_quant
        )
        
        float_kernel = kind is None
        if float_kernel:
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            # The sweep lands after the narrowing so the tokens are written in the
            # kernel's own width; generate_input_data() draws integers, which have no
            # non-finite image to narrow.
            input_q = self._maybe_apply_input_mode(self.generate_input_data().astype(float_dtype))
            interpreter_input_dtype = self.load_litert_interpreter(str(tflite_path)).get_input_details()[0]['dtype']

            def float_reference(operands, _dtype=float_dtype, _in_dtype=interpreter_input_dtype):
                return self.run_inference(
                    str(tflite_path), operands[0].astype(_in_dtype)
                ).astype(_dtype)

            output_data = float_reference([input_q])
        else:
            input_q, output_data = self._int_golden(kind, input_shape, output_shape, (pool_h, pool_w), pool_params)

        # Format input and output arrays
        if float_kernel:
            output_data, nonfinite_context = self.apply_nonfinite_policy(
                output_data, reference=float_reference, inputs=[input_q]
            )
        else:
            nonfinite_context = {}
        input_data_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate buffer size max
        activation_dtype = self.tensor_dtype("input")
        buffer_size_max = builder.calculate_pooling_buffer_size_max(
            input_dims,
            output_dims,
            pooling_type=pooling_type,
            output_dtype=activation_dtype
        )
        
        # Build template context
        context = {
            'name': name,
            'input_dims': input_dims,
            'filter_dims': filter_dims,
            'output_dims': output_dims,
            'pool_params': pool_params,
            'input_data_array': input_data_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'requires_kernel_prototype': kernel_info["kernel_fn"].startswith(
                ("arm_avg_pool_nhwc_", "arm_max_pool_nhwc_")
            ),
            'buffer_size_max': buffer_size_max,
            'pooling_type': pooling_type,
            'pool_params_type': (
                'cmsis_nn_pool_params_f16'
                if kernel_info["input_c_type"] == "float16_t"
                else ('cmsis_nn_pool_params_f32' if kernel_info["input_c_type"] == "float" else 'cmsis_nn_pool_params')
            ),
            'float_kernel': float_kernel,
        }
        if float_kernel:
            context["pool_activation_min_literal"] = builder.format_float_literal(pool_params["activation_min"])
            context["pool_activation_max_literal"] = builder.format_float_literal(pool_params["activation_max"])
        context.update(nonfinite_context)
        pool = pool_argument_pool(context)
        fault = self.fault_kind()
        # The validation rules are keyed by the former template path; a fault case's sidecar keeps
        # the rules of the fault template it used to render from.
        validation_key = f"{self.TEMPLATE_DIR}/{self.TEMPLATE_SUFFIX}.c.j2"
        if fault:
            self._check_fault_reachable(fault, float_kernel, kernel_info["kernel_fn"])
            context.update(self.fault_context())
            pool = pool_fault(pool, fault, context)
            validation_key = f"{self.TEMPLATE_DIR}/{self.TEMPLATE_SUFFIX}_fault.c.j2"

        self.render_harness_files(output_dir, stem=self.TEMPLATE_SUFFIX, context=context, pool=pool,
                                  validation_key=validation_key, label="Pooling", sidecar=True)
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', self.OPERATOR_NAME),
            'operator_name': self.TEMPLATE_SUFFIX,
        }
        (output_dir / "CMakeLists.txt").write_text(self.render_template("common/CMakeLists.txt.j2", cmake_context))
        
