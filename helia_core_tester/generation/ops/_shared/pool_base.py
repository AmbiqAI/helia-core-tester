"""Shared implementation for CMSIS-NN avg/max pool operators."""

from typing import Dict
from pathlib import Path
import numpy as np
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

    def uses_reference(self) -> bool:
        return True

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
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Pooling operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        from helia_core_tester.generation.ops._shared.conv_reference import float_bounds, quantized_bounds
        from helia_core_tester.generation.reference import policy, weighted
        from helia_core_tester.generation.reference.call import ReferenceCall

        name = self.desc['name']
        kernel_info = self._select_cmsis_pooling_kernel()
        pooling_type = self.POOL_KIND
        entry_stem = "avg_pool" if pooling_type == "AVERAGE" else "max_pool"

        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        if len(input_shape) != 4:
            raise ValueError(f"{name}: pooling needs an NHWC input_shape")
        pool_h, pool_w = weighted.pair(self.desc, 'pool_size', 2)
        stride_h, stride_w = weighted.pair(self.desc, 'strides', 2)
        out_h, _ = weighted.same_or_valid(self.desc.get('padding', 'valid'), input_shape[1], pool_h, stride_h, 1)
        out_w, _ = weighted.same_or_valid(self.desc.get('padding', 'valid'), input_shape[2], pool_w, stride_w, 1)
        output_shape = (input_shape[0], out_h, out_w, input_shape[3])

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        filter_dims = {'n': 1, 'h': pool_h, 'w': pool_w, 'c': 1}
        geometry = builder.build_pool_params(self.desc, input_shape, (pool_h, pool_w), output_shape, {})
        window = {"stride_h": stride_h, "stride_w": stride_w, "filter_h": pool_h, "filter_w": pool_w,
                  "pad_h": int(geometry["pad_h"]), "pad_w": int(geometry["pad_w"])}

        float_kernel = kernel_info["input_c_type"] in {"float", "float16_t"}
        if float_kernel:
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            # The sweep lands after the narrowing so the tokens are written in the
            # kernel's own width; generate_input_data() draws integers, which have no
            # non-finite image to narrow.
            input_q = self._maybe_apply_input_mode(self.generate_input_data().astype(float_dtype))
            lo, hi = float_bounds(self)
            output_data = self.reference_golden(ReferenceCall(
                f"{entry_stem}_{'f16' if float_dtype == np.float16 else 'f32'}",
                {**window, "activation_min": lo, "activation_max": hi},
                {"input": np.ascontiguousarray(input_q)}, {"output": output_shape}))
            output_data, nonfinite_context = self.apply_nonfinite_policy(
                output_data, reference=self.reference_probe, inputs=[input_q]
            )
            pool_params = {**geometry, 'activation_min': lo, 'activation_max': hi}
        else:
            kind = "s16" if kernel_info["input_c_type"] == "int16_t" else "s8"
            # Pooling keeps the input quantization on its output; the [-1, 1] calibration range.
            quant = self.activation_quant("input", (-1.0, 1.0), kind)
            input_q = policy.quantize(self._sample_uniform(input_shape), quant)
            act_min, act_max = quantized_bounds(self, kind, quant.scale, quant.zero_point)
            output_data = self.reference_golden(ReferenceCall(
                f"{entry_stem}_{kind}", {**window, "activation_min": act_min, "activation_max": act_max},
                {"input": np.ascontiguousarray(input_q)}, {"output": output_shape},
                quant={"input": quant.to_json(), "output": quant.to_json()}))
            nonfinite_context = {}
            pool_params = {**geometry, 'activation_min': act_min, 'activation_max': act_max}

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
        
