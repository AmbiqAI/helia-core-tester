"""
TransposeConv operation implementation.
"""

from typing import Dict, Any
import numpy as np
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.entry import resolve_entry
from helia_core_tester.generation.harness import (
    ArgumentPool,
    ArrayLiteral,
    Declaration,
    GuardedBuffer,
    Provider,
    SizeQuery,
)
from helia_core_tester.generation.harness.faults import common_fault, null_context_buffer, struct_copy, with_fault
from pathlib import Path

TRANSPOSE_CONV_VALIDATION_KEY = "ConvolutionFunctions/transpose_conv/transpose_conv.c.j2"


def transpose_conv_argument_pool(context: Dict[str, Any]) -> ArgumentPool:
    """Every value a TransposeConv case can pass to a public transpose-conv kernel or its scratch queries."""
    n = context["name"]
    upper = n.upper()
    float_kernel = bool(context.get("float_kernel"))
    tc = context["transpose_conv_params"]

    def dims(d: Dict[str, Any]) -> Dict[str, Any]:
        return {"n": d["n"], "h": d["h"], "w": d["w"], "c": d["c"]}

    geometry = {
        "stride": {"w": tc["stride_w"], "h": tc["stride_h"]},
        "dilation": {"w": tc["dilation_w"], "h": tc["dilation_h"]},
        "padding": {"w": tc["pad_w"], "h": tc["pad_h"]},
        "padding_offsets": {"w": tc["pad_offset_w"], "h": tc["pad_offset_h"]},
    }
    if float_kernel:
        params_init = {**geometry, "activation": {"min": context["transpose_activation_min_literal"],
                                                  "max": context["transpose_activation_max_literal"]}}
    else:
        params_init = {"input_offset": tc["input_offset"], "output_offset": tc["output_offset"], **geometry,
                       "activation": {"min": tc["activation_min"], "max": tc["activation_max"]}}
    out_c = context["output_dims"]["c"]
    header = [
        Declaration(f"{n}_input_dims", "cmsis_nn_dims", dims(context["input_dims"]), comment="Input dimensions"),
        Declaration(f"{n}_filter_dims", "cmsis_nn_dims", dims(context["filter_dims"]),
                    comment="Filter dimensions (C_OUT, HK, WK, C_IN)"),
        Declaration(f"{n}_output_dims", "cmsis_nn_dims", dims(context["output_dims"]), comment="Output dimensions"),
        Declaration(f"{n}_bias_dims", "cmsis_nn_dims", {"n": 1, "h": 1, "w": 1, "c": out_c}, comment="Bias dimensions"),
        Declaration(f"{n}_transpose_conv_params",
                    context.get("transpose_conv_params_type") or "cmsis_nn_transpose_conv_params", params_init,
                    comment="Transpose convolution parameters"),
    ]
    has_biases = bool(context["has_biases"])
    params_expr = f"&{n}_transpose_conv_params"
    values = {
        "ctx": f"&{n}_ctx", "transpose_conv_params": params_expr, "transposed_conv_params": params_expr,
        "input_dims": f"&{n}_input_dims", "filter_dims": f"&{n}_filter_dims", "filter_data": f"{n}_weights",
        "bias_dims": f"&{n}_bias_dims", "bias_data": f"{n}_biases" if has_biases else "NULL",
        "output_dims": f"&{n}_output_dims", "out_dims": f"&{n}_output_dims",
        "layout": context.get("kernel_layout") or "ARM_NN_LAYOUT_NHWC",
    }
    if not float_kernel:
        quant = context["quant_params"]
        if quant.get("per_channel"):
            multiplier, shift = ArrayLiteral(quant["multiplier_array"]), ArrayLiteral(quant["shift_array"])
        else:
            multiplier, shift = f"{{ {quant['multiplier']} }}", f"{{ {quant['shift']} }}"
        header += [
            Declaration(f"{n}_multiplier", "int32_t", multiplier, storage="static", array=True,
                        comment="Quantization parameters (per-channel)"),
            Declaration(f"{n}_shift", "int32_t", shift, storage="static", array=True),
            Declaration(f"{n}_quant_params", "cmsis_nn_per_channel_quant_params",
                        {"multiplier": f"{n}_multiplier", "shift": f"{n}_shift"}),
        ]
        values["quant_params"] = f"&{n}_quant_params"
    bias_ctype = context["bias_dtype"]
    header += [
        Declaration(f"{n}_weights", context.get("weight_dtype") or "int8_t", ArrayLiteral(context["weights_array"]),
                    array=True, comment="Weights"),
        Declaration(f"{n}_biases", bias_ctype, ArrayLiteral(context["biases_array"]), array=True, comment="Biases")
        if has_biases else Declaration(f"{n}_biases", f"{bias_ctype}*", "NULL", comment="No biases"),
        Declaration(f"{n}_input", context["input_dtype"], ArrayLiteral(context["input_data_array"]), array=True,
                    comment="Input data (for testing)"),
        Declaration(f"{n}_expected_output", context["output_dtype"], ArrayLiteral(context["expected_output_array"]),
                    array=True, comment="Expected output (golden)"),
    ]
    reverse = Provider(
        param="reverse_conv_ctx", aliases=("output_ctx",), expr=f"&{n}_reverse_conv_ctx",
        declarations=(Declaration(f"{n}_reverse_conv_ctx", "cmsis_nn_context", storage="static",
                                  comment="Reverse convolution context (output_ctx of the kernels)"),),
        buffers=(GuardedBuffer(f"{n}_reverse_conv_ctx_buffer", "uint8_t", f"{upper}_REVERSE_CONV_CTX_SIZE",
                               count_value=str(context["reverse_conv_ctx_size"]), label="reverse_conv_ctx"),),
        setup=(f"    // Initialize reverse convolution context buffer (output_ctx parameter)\n"
               f"    {n}_reverse_conv_ctx.buf = {n}_reverse_conv_ctx_buffer;\n"
               f"    {n}_reverse_conv_ctx.size = {upper}_REVERSE_CONV_CTX_SIZE;"),
        size_query=SizeQuery(context["kernel_get_reverse_buffer_size_fn"], "reverse_required_buffer_size",
                             f"{upper}_REVERSE_CONV_CTX_SIZE"),
    )
    providers = [reverse]
    if context.get("has_weight_sum"):
        providers.append(Provider(
            param="weight_sum_ctx", expr=f"&{n}_weight_sum_ctx",
            declarations=(Declaration(f"{n}_weight_sum_ctx", "cmsis_nn_context", storage="static",
                                      comment="Weight sum context for s8 transpose convolutions"),),
            buffers=(GuardedBuffer(f"{n}_weight_sum_buffer", "uint8_t", f"{upper}_WEIGHT_SUM_BUFFER_SIZE",
                                   count_value=f"({out_c} * sizeof(int32_t))", label="weight_sum"),),
            setup=(f"    // Initialize weight sum context and compute weight_sum\n"
                   f"    {n}_weight_sum_ctx.buf = {n}_weight_sum_buffer;\n"
                   f"    {n}_weight_sum_ctx.size = {upper}_WEIGHT_SUM_BUFFER_SIZE;\n\n"
                   f"    int32_t lhs_offset = {n}_transpose_conv_params.input_offset;\n"
                   f"    arm_convolve_weight_sum(\n"
                   f"        (int32_t *){n}_weight_sum_ctx.buf,\n"
                   f"        {n}_weights,\n"
                   f"        &{n}_input_dims,\n"
                   f"        &{n}_filter_dims,\n"
                   f"        &{n}_output_dims,\n"
                   f"        lhs_offset,\n"
                   f"        {n + '_biases' if has_biases else 'NULL'}\n"
                   f"    );"),
        ))
    else:
        values["weight_sum_ctx"] = "NULL"
    output = context["output_dims"]
    return ArgumentPool(
        name=n, values=values, header=header, providers=tuple(providers),
        output_count=f"({output['n']} * {output['h']} * {output['w']} * {output['c']})", benchmark=False,
    )


def transpose_conv_fault(pool: ArgumentPool, kind: str, context: Dict[str, Any]) -> ArgumentPool:
    """The pool of a TransposeConv fault case: the passing pool with the faulted argument edited."""
    n = context["name"]
    edit = common_fault(pool, kind, layout=context.get("kernel_layout"))
    if edit is None and kind == "nonunit_dilation":
        edit = struct_copy(pool, kind, "transpose_conv_params",
                           context.get("transpose_conv_params_type") or "cmsis_nn_transpose_conv_params",
                           f"{n}_transpose_conv_params", {"dilation.w": 2})
    elif edit is None and kind == "null_weight_sum_ctx":
        edit = null_context_buffer(pool, kind, "weight_sum_ctx", f"{n}_weight_sum_ctx")
    elif edit is None and kind == "null_reverse_conv_ctx_buf":
        edit = null_context_buffer(pool, kind, "reverse_conv_ctx", f"{n}_reverse_conv_ctx")
    if edit is None:
        raise ValueError(f"{n}: no TransposeConv fault edit for {kind!r}")
    return with_fault(pool, edit)


class OpTransposeConv(OperationBase):
    """
    TransposeConv operation.
    """

    FAULT_KINDS = (
        "null_ctx_buf",
        "null_weight_sum_ctx",
        "null_reverse_conv_ctx_buf",
        "nonunit_dilation",
        "null_input",
        "null_output",
        "invalid_layout",
    )

    def _check_fault_reachable(self, kind: str, context: Dict[str, Any]) -> None:
        """Reject fault kinds the selected transpose-conv kernel route does not diagnose."""
        kernel_fn = context["kernel_fn"]
        if context.get("float_kernel"):
            if kind not in ("null_input", "null_output", "invalid_layout"):
                raise self.fault_unreachable(kind, f"{kernel_fn} has no such guard")
            return
        if kind in ("null_input", "null_output", "invalid_layout"):
            raise self.fault_unreachable(kind, f"{kernel_fn} does not check {kind}")
        if kind in ("null_reverse_conv_ctx_buf", "null_weight_sum_ctx"):
            params = context["transpose_conv_params"]
            reverse_conv = (
                params["stride_w"] <= 2
                and params["stride_h"] <= 2
                and context["input_dims"]["c"] > 16
            )
            if not reverse_conv:
                raise self.fault_unreachable(
                    kind,
                    f"{kernel_fn} only checks it on the reverse-conv route "
                    "(stride <= 2 and input channels > 16)",
                )

    def uses_reference(self) -> bool:
        return True

    def _generate_int_reference(self, output_dir: Path, kernel_info: Dict[str, str]) -> None:
        """Render an s8 case whose golden comes from the TFLM reference transpose
        convolution (activation none, as TFLM's kernel applies none)."""
        from helia_core_tester.generation.reference import weighted
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        builder = TemplateContextBuilder()
        spec = weighted.tconv_spec(self.desc)
        case = weighted.build_weighted_case(self.desc, spec, self.reference_rng("weights"), self.generate_input_data)
        self._reference_call = case.call
        out_ch, kh, kw, in_ch = spec.weight_shape
        input_dims = builder.nhwc_to_cmsis_dims(spec.input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(spec.output_shape)
        filter_dims = {"n": out_ch, "h": kh, "w": kw, "c": in_ch}
        transpose_conv_params = builder.build_transpose_conv_params(
            self.desc, spec.input_shape, (kh, kw), spec.output_shape,
            {"zero_point": case.input_quant.zero_point}, {"zero_point": case.output_quant.zero_point},
        )
        transpose_conv_params["activation_min"] = case.act_min
        transpose_conv_params["activation_max"] = case.act_max
        has_biases = case.bias_q is not None
        stride_h, stride_w = spec.params["stride"]
        buffer_size_max = builder.calculate_transpose_conv_buffer_size_max(
            input_dims, filter_dims, output_dims, output_dtype=self.desc.get("activation_dtype", "S8"),
            stride_h=int(stride_h), stride_w=int(stride_w),
        )
        # REVERSE_TCOL_EFFICIENT_THRESHOLD = 16: the reverse-conv route needs a filter-sized
        # context; the other route an output-sized one.
        if stride_w <= 2 and stride_h <= 2 and input_dims["c"] > 16:
            reverse_conv_ctx_size = input_dims["c"] * kw * kh * out_ch
        else:
            reverse_conv_ctx_size = output_dims["w"] * output_dims["h"] * output_dims["c"] * 4
        has_weight_sum = kernel_info["input_c_type"] == "int8_t"
        context = {
            "name": name,
            "input_dims": input_dims,
            "filter_dims": filter_dims,
            "output_dims": output_dims,
            "transpose_conv_params": transpose_conv_params,
            "quant_params": case.quant_context(builder),
            "weights_array": builder.format_array_as_c_literal(case.weights_c),
            "biases_array": builder.format_array_as_c_literal(case.bias_q) if has_biases else "",
            "has_biases": has_biases,
            "has_weight_sum": has_weight_sum,
            "weight_sum_size": output_dims["c"] * 4 if has_weight_sum else 0,
            "input_data_array": builder.format_array_as_c_literal(case.input_q),
            "expected_output_array": builder.format_array_as_c_literal(case.output_q),
            "input_dtype": kernel_info["input_c_type"],
            "output_dtype": kernel_info["output_c_type"],
            "bias_dtype": kernel_info["bias_c_type"],
            "kernel_fn": kernel_info["kernel_fn"],
            "kernel_get_buffer_size_fn": kernel_info["kernel_get_buffer_size_fn"],
            "kernel_get_reverse_buffer_size_fn": kernel_info["kernel_get_reverse_buffer_size_fn"],
            "buffer_size_max": buffer_size_max,
            "reverse_conv_ctx_size": reverse_conv_ctx_size,
        }
        self._render_transpose_conv(output_dir, context)
        cmake_content = self.render_template(
            "common/CMakeLists.txt.j2",
            {"name": name, "operator": self.desc.get("operator", "TransposeConv"), "operator_name": "transpose_conv"},
        )
        (output_dir / "CMakeLists.txt").write_text(cmake_content)

    def _select_cmsis_transpose_conv_kernel(self) -> Dict[str, str]:
        info = self._table_transpose_conv_kernel()
        entry = self.desc.get("entry")
        if entry:
            if self.desc.get("fault"):
                raise ValueError(f"{self.desc.get('name')}: entry {entry!r} is not supported with fault")
            info.update(resolve_entry(
                "TransposeConv", str(entry),
                activation_dtype=self.desc.get("activation_dtype", "S8"),
                weight_dtype=self.desc.get("weight_dtype", "S8"),
                cpu=self.target_cpu, desc=self.desc,
            ))
        return info

    def _render_transpose_conv(self, output_dir: Path, context: Dict[str, Any]) -> None:
        pool = transpose_conv_argument_pool(context)
        fault = self.fault_kind()
        if fault:
            self._check_fault_reachable(fault, context)
            context.update(self.fault_context())
            pool = transpose_conv_fault(pool, fault, context)
        self.render_harness_files(output_dir, stem="transpose_conv", context=context, pool=pool,
                                  validation_key=TRANSPOSE_CONV_VALIDATION_KEY, label="Transpose convolution")

    def _table_transpose_conv_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN transpose convolution kernel function.
        
        Returns:
            Dictionary with kernel function name, C types, and buffer size function
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8').upper()
        weight_dtype = self.desc.get('weight_dtype', 'S8').upper()
        hint = self.desc.get("hint", {})
        hint = hint if isinstance(hint, dict) else {}
        variant = str(hint.get("kernel_variant", "")).lower()
        if variant and variant != "wrapper":
            raise ValueError(f"Unsupported TransposeConv kernel_variant hint: {variant}")
        
        if activation_dtype == 'FP32' and weight_dtype == 'FP32':
            return {
                'kernel_fn': 'arm_transpose_conv_wrapper_f32' if variant == "wrapper" else 'arm_transpose_conv_f32',
                'kernel_get_buffer_size_fn': 'arm_transpose_conv_f32_get_buffer_size',
                'kernel_get_reverse_buffer_size_fn': 'arm_transpose_conv_f32_get_reverse_conv_buffer_size',
                'input_c_type': 'float',
                'output_c_type': 'float',
                'weight_c_type': 'float',
                'bias_c_type': 'float',
                'layout': 'ARM_NN_LAYOUT_NHWC',
            }
        if activation_dtype == 'FP16' and weight_dtype == 'FP16':
            return {
                'kernel_fn': 'arm_transpose_conv_wrapper_f16' if variant == "wrapper" else 'arm_transpose_conv_f16',
                'kernel_get_buffer_size_fn': 'arm_transpose_conv_f16_get_buffer_size',
                'kernel_get_reverse_buffer_size_fn': 'arm_transpose_conv_f16_get_reverse_conv_buffer_size',
                'input_c_type': 'float16_t',
                'output_c_type': 'float16_t',
                'weight_c_type': 'float16_t',
                'bias_c_type': 'float16_t',
                'layout': 'ARM_NN_LAYOUT_NHWC',
            }
        if activation_dtype == 'S8' and weight_dtype == 'S8':
            return {
                'kernel_fn': 'arm_transpose_conv_wrapper_s8',
                'kernel_get_buffer_size_fn': 'arm_transpose_conv_s8_get_buffer_size',
                'kernel_get_reverse_buffer_size_fn': 'arm_transpose_conv_s8_get_reverse_conv_buffer_size',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'weight_c_type': 'int8_t',
                'bias_c_type': 'int32_t'
            }
        else:
            raise NotImplementedError(f"Unsupported TransposeConv dtype combo: {activation_dtype} x {weight_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for TransposeConv operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_transpose_conv_kernel()
        float_kernel = kernel_info["input_c_type"] in {"float", "float16_t"}
        float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
        if not float_kernel:
            self._generate_int_reference(output_dir, kernel_info)
            return

        from helia_core_tester.generation.reference import weighted

        builder = TemplateContextBuilder()
        spec = weighted.tconv_spec(self.desc)
        input_shape, output_shape = spec.input_shape, spec.output_shape
        case = weighted.build_float_case(
            self.desc, spec, self.reference_rng("weights"),
            lambda: self._sample_uniform(input_shape, dtype=float_dtype), float_dtype,
        )
        self._reference_call = case.call
        out_ch, kh, kw, in_ch = spec.weight_shape
        filter_dims = {"n": out_ch, "h": kh, "w": kw, "c": in_ch}
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        unquantized = {"scale": 1.0, "zero_point": 0}
        transpose_conv_params = builder.build_transpose_conv_params(
            self.desc, input_shape, (kh, kw), output_shape, unquantized, unquantized
        )
        weights, biases, input_data = case.weights, case.bias, case.input
        has_biases = biases is not None
        output_data, nonfinite_context = self.apply_nonfinite_policy(
            case.output, reference=case.reference, inputs=[input_data]
        )

        weights_array_str = builder.format_array_as_c_literal(weights)
        biases_array_str = builder.format_array_as_c_literal(biases) if has_biases else ""
        input_data_array_str = builder.format_array_as_c_literal(np.asarray(input_data, dtype=float_dtype))
        expected_output_array_str = builder.format_array_as_c_literal(np.asarray(output_data, dtype=float_dtype))

        element_size = np.dtype(float_dtype).itemsize
        buffer_size_max = max(
            1024,
            int(
                (input_dims['n'] * input_dims['h'] * input_dims['w'] * input_dims['c']
                 + filter_dims['n'] * filter_dims['h'] * filter_dims['w'] * max(filter_dims['c'], 1)
                 + output_dims['n'] * output_dims['h'] * output_dims['w'] * output_dims['c']) * element_size
            ),
        )
        reverse_conv_ctx_size = max(
            1024,
            int(output_dims['w'] * output_dims['h'] * output_dims['c'] * element_size),
        )

        context = {
            'name': name,
            'input_dims': input_dims,
            'filter_dims': filter_dims,
            'output_dims': output_dims,
            'transpose_conv_params': transpose_conv_params,
            'weights_array': weights_array_str,
            'biases_array': biases_array_str,
            'has_biases': has_biases,
            'has_weight_sum': False,
            'weight_sum_size': 0,
            'input_data_array': input_data_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'weight_dtype': kernel_info.get("weight_c_type", kernel_info["input_c_type"]),
            'bias_dtype': kernel_info["bias_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'kernel_get_reverse_buffer_size_fn': kernel_info["kernel_get_reverse_buffer_size_fn"],
            'kernel_layout': kernel_info.get("layout", "ARM_NN_LAYOUT_NHWC"),
            'buffer_size_max': buffer_size_max,
            'reverse_conv_ctx_size': reverse_conv_ctx_size,
            'float_kernel': True,
            'transpose_conv_params_type': (
                'cmsis_nn_transpose_conv_params_f16'
                if kernel_info["input_c_type"] == "float16_t"
                else 'cmsis_nn_transpose_conv_params_f32'
            ),
            'transpose_activation_min_literal': builder.format_float_literal(transpose_conv_params['activation_min']),
            'transpose_activation_max_literal': builder.format_float_literal(transpose_conv_params['activation_max']),
        }
        context.update(nonfinite_context)
        self._render_transpose_conv(output_dir, context)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'TransposeConv'),
            'operator_name': 'transpose_conv'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        return
