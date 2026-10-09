"""DepthwiseConv operation implementation."""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.entry import check_entry_fault, resolve_entry
from helia_core_tester.generation.harness import (
    ArgumentPool,
    ArrayLiteral,
    Declaration,
    GuardedBuffer,
    Provider,
    RuleCheck,
)
from helia_core_tester.generation.harness.faults import common_fault, null_context_buffer, struct_copy, with_fault
from helia_core_tester.generation.kernel_dispatch import (
    DEPTHWISE_CONV_S8_PLANAR_RULE,
    autovectorize_declines_if,
    depthwise_3x3_scratch_bytes,
    resolve_depthwise_conv_kernel,
)


def depthwise_argument_pool(context: Dict[str, Any]) -> ArgumentPool:
    """Every value a DepthwiseConv case can pass to a public depthwise kernel, its sizer or
    the planar rule."""
    n = context["name"]
    float_kernel = bool(context.get("float_kernel"))
    has_biases = bool(context["has_biases"])
    dw = context["dw_conv_params"]

    def dims(d: Dict[str, Any]) -> Dict[str, Any]:
        return {"n": d["n"], "h": d["h"], "w": d["w"], "c": d["c"]}

    geometry = {
        "ch_mult": dw["ch_mult"],
        "stride": {"w": dw["stride_w"], "h": dw["stride_h"]},
        "dilation": {"w": dw["dilation_w"], "h": dw["dilation_h"]},
        "padding": {"w": dw["pad_w"], "h": dw["pad_h"]},
    }
    if float_kernel:
        params_init = {**geometry, "activation": {"min": context["dw_activation_min_literal"],
                                                  "max": context["dw_activation_max_literal"]}}
    else:
        params_init = {"input_offset": dw["input_offset"], "output_offset": dw["output_offset"], **geometry,
                       "activation": {"min": dw["activation_min"], "max": dw["activation_max"]}}
    header = [
        Declaration(f"{n}_input_dims", "cmsis_nn_dims", dims(context["input_dims"]), comment="Input dimensions"),
        Declaration(f"{n}_filter_dims", "cmsis_nn_dims", dims(context["filter_dims"]), comment="Filter dimensions"),
        Declaration(f"{n}_output_dims", "cmsis_nn_dims", dims(context["output_dims"]), comment="Output dimensions"),
        Declaration(f"{n}_dw_conv_params", context.get("dw_conv_params_type") or "cmsis_nn_dw_conv_params",
                    params_init, comment="Depthwise convolution parameters"),
    ]
    bias_ctype = context["bias_dtype"]
    bias_expr = "NULL"
    if has_biases:
        bias_expr = ("(const int64_t*)" if bias_ctype == "int64_t" else "") + f"{n}_biases"
    values = {
        "ctx": f"&{n}_ctx", "dw_conv_params": f"&{n}_dw_conv_params", "input_dims": f"&{n}_input_dims",
        "filter_dims": f"&{n}_filter_dims", "filter_data": f"{n}_weights", "bias_dims": f"&{n}_bias_dims",
        "bias_data": bias_expr, "output_dims": f"&{n}_output_dims",
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
    header += [
        Declaration(f"{n}_weights", context.get("weight_dtype") or "int8_t", ArrayLiteral(context["weights_array"]),
                    array=True, comment="Weights"),
        Declaration(f"{n}_biases", bias_ctype, ArrayLiteral(context["biases_array"]), array=True, comment="Biases")
        if has_biases else Declaration(f"{n}_biases", f"{bias_ctype}*", "NULL", comment="No biases"),
    ]
    output_c = context["output_dims"]["c"]
    providers = []
    if context.get("has_weight_sum"):
        header.append(Declaration(f"{n}_weight_sum", "int32_t", ArrayLiteral(context["weight_sum_array"]), array=True,
                                  comment="Weight sum (precomputed for S8 depthwise convolutions)"))
        providers.append(Provider(
            param="weight_sum_ctx",
            expr=f"&{n}_weight_sum_ctx",
            declarations=(Declaration(f"{n}_weight_sum_ctx", "cmsis_nn_context", storage="static",
                                      comment="Weight sum context for s8 depthwise wrappers"),),
            buffers=(GuardedBuffer(f"{n}_weight_sum_runtime", "int32_t", str(output_c), label="weight_sum"),),
            setup=(f"    // Compute weight sums at runtime when supported; fallback to precomputed values.\n"
                   f"    arm_cmsis_nn_status weight_sum_status = arm_depthwise_convolve_weight_sum(\n"
                   f"        {n}_weight_sum_runtime,\n"
                   f"        NULL,\n"
                   f"        {n}_weights,\n"
                   f"        &{n}_dw_conv_params,\n"
                   f"        &{n}_input_dims,\n"
                   f"        &{n}_filter_dims,\n"
                   f"        &{n}_output_dims,\n"
                   f"        {n}_dw_conv_params.input_offset,\n"
                   f"        {n + '_biases' if has_biases else 'NULL'}\n"
                   f"    );\n"
                   f"    if (weight_sum_status == ARM_CMSIS_NN_SUCCESS) {{\n"
                   f"        {n}_weight_sum_ctx.buf = (uint8_t *){n}_weight_sum_runtime;\n"
                   f"    }} else {{\n"
                   f"        {n}_weight_sum_ctx.buf = (uint8_t *){n}_weight_sum;\n"
                   f"    }}\n"
                   f"    {n}_weight_sum_ctx.size = {output_c} * sizeof(int32_t);"),
        ))
    header += [
        Declaration(f"{n}_input", context["input_dtype"], ArrayLiteral(context["input_data_array"]), array=True,
                    comment="Input data (for testing)"),
        Declaration(f"{n}_expected_output", context["output_dtype"], ArrayLiteral(context["expected_output_array"]),
                    array=True, comment="Expected output (golden)"),
    ]
    source = [Declaration(f"{n}_bias_dims", "cmsis_nn_dims", {"n": 1, "h": 1, "w": 1, "c": output_c},
                          comment="Bias dimensions: bias shape is [1, 1, 1, C_OUT]")]
    checks = ()
    if context.get("planar_supported") is not None:
        checks = (RuleCheck(context["planar_rule_fn"], 1 if context["planar_supported"] else 0, "planar_supported"),)
    output = context["output_dims"]
    return ArgumentPool(
        name=n, values=values, header=header, source=source, providers=tuple(providers), checks=checks,
        output_count=f"({output['n']} * {output['h']} * {output['w']} * {output['c']})", benchmark=False,
    )


def depthwise_fault(pool: ArgumentPool, kind: str, context: Dict[str, Any]) -> ArgumentPool:
    """The pool of a DepthwiseConv fault case: the passing pool with the faulted argument edited."""
    n = context["name"]
    edit = common_fault(pool, kind, layout=context.get("kernel_layout"))
    if edit is None and kind == "channel_mismatch":
        edit = struct_copy(pool, kind, "output_dims", "cmsis_nn_dims", f"{n}_output_dims",
                           {"c": int(context["input_dims"]["c"]) + 1})
    elif edit is None and kind == "null_weight_sum_ctx":
        edit = null_context_buffer(pool, kind, "weight_sum_ctx", f"{n}_weight_sum_ctx")
    if edit is None:
        raise ValueError(f"{n}: no DepthwiseConv fault edit for {kind!r}")
    return with_fault(pool, edit)


def _opt_dilation_supported(
    params: Dict[str, int],
    input_dims: Dict[str, int],
    filter_dims: Dict[str, int],
    output_dims: Dict[str, int],
) -> bool:
    """Mirror of ns-cmsis-nn arm_nn_dw_conv_opt_dilation_supported() (v7.36.0): the s8 opt and
    s16 fast depthwise kernels take unit dilation, or a dilated 1D layer with no vertical
    extent, unit stride and no vertical padding."""
    if params["dilation_w"] == 1 and params["dilation_h"] == 1:
        return True
    return (
        params["dilation_h"] == 1
        and params["dilation_w"] >= 1
        and filter_dims["h"] == 1
        and input_dims["h"] == 1
        and output_dims["h"] == 1
        and params["stride_w"] == 1
        and params["stride_h"] == 1
        and params["pad_h"] == 0
    )


def vector_sum_s8(
    vector_data: np.ndarray,
    vector_cols: int,
    vector_rows: int,
    lhs_offset: int,
    rhs_offset: int,
    bias_data: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Pure-Python port of CMSIS-NN's arm_vector_sum_s8 helper.

    Args:
        vector_data: Flattened (rows * cols) int8 buffer containing the kernel matrix.
        vector_cols: Number of columns per output channel (accumulation depth).
        vector_rows: Number of rows/output channels.
        lhs_offset: Input offset applied during accumulation.
        rhs_offset: Filter/weight offset applied during accumulation.
        bias_data: Optional per-output bias (length == vector_rows), int32.
    
    Returns:
        np.ndarray: int32 array with the kernel sums, matching CMSIS-NN behaviour.
    """
    kernel = np.asarray(vector_data, dtype=np.int8).reshape(vector_rows, vector_cols)
    
    if bias_data is not None:
        vector_sum_buf = np.asarray(bias_data, dtype=np.int32).copy()
    else:
        vector_sum_buf = np.zeros(vector_rows, dtype=np.int32)
    
    if lhs_offset != 0:
        sums = kernel.astype(np.int32).sum(axis=1)
        if rhs_offset != 0:
            sums = sums + vector_cols * rhs_offset
        vector_sum_buf += sums * lhs_offset
    
    return vector_sum_buf


class OpDepthwiseConv(OperationBase):
    """DepthwiseConv operation."""

    FAULT_KINDS = (
        "null_ctx_buf",
        "null_weight_sum_ctx",
        "channel_mismatch",
        "null_input",
        "null_output",
        "invalid_layout",
    )

    def _hint(self) -> Dict[str, Any]:
        hint = self.desc.get("hint", {})
        return hint if isinstance(hint, dict) else {}

    def _check_fault_reachable(self, kind: str, context: Dict[str, Any]) -> None:
        """Reject fault kinds the selected depthwise kernel route does not diagnose."""
        kernel_fn = context["kernel_fn"]
        if context.get("float_kernel"):
            if kind not in ("null_input", "null_output", "invalid_layout"):
                raise self.fault_unreachable(kind, f"{kernel_fn} has no such guard")
            if kind == "invalid_layout" and not context.get("kernel_needs_layout"):
                raise self.fault_unreachable(kind, f"{kernel_fn} takes no layout argument")
            return
        if kind in ("null_input", "null_output", "invalid_layout"):
            raise self.fault_unreachable(kind, f"{kernel_fn} does not check {kind}")
        if kind == "null_weight_sum_ctx" and kernel_fn != "arm_depthwise_conv_wrapper_s8":
            raise self.fault_unreachable(kind, f"{kernel_fn} takes no weight-sum context")

        params = context["dw_conv_params"]
        input_dims = context["input_dims"]
        filter_dims = context["filter_dims"]
        unit_dilation = params["dilation_w"] == 1 and params["dilation_h"] == 1
        if kernel_fn == "arm_depthwise_conv_wrapper_s8":
            is_3x3 = filter_dims["w"] == 3 and filter_dims["h"] == 3
            optimized = (
                params["ch_mult"] == 1
                and input_dims["n"] == 1
                and _opt_dilation_supported(params, input_dims, filter_dims, context["output_dims"])
                and not (is_3x3 and params["pad_h"] <= 1 and params["pad_w"] <= 1 and unit_dilation)
                and input_dims["c"] != 1
            )
        elif kernel_fn == "arm_depthwise_conv_wrapper_s16":
            optimized = (
                params["ch_mult"] == 1
                and _opt_dilation_supported(params, input_dims, filter_dims, context["output_dims"])
                and filter_dims["w"] * filter_dims["h"] < 512
            )
        else:
            optimized = params["ch_mult"] == 1 and input_dims["n"] == 1 and unit_dilation
        if not optimized:
            raise self.fault_unreachable(
                kind,
                f"{kernel_fn} only checks it on the optimized route "
                "(ch_mult 1; s8 and s16: unit dilation or dilated 1D; s8: batch 1, not a 3x3 filter with "
                "pad <= 1 and input_ch > 1; s16: filter w*h < 512; s4: batch 1, unit dilation)",
            )
        if kind == "null_ctx_buf" and kernel_fn != "arm_depthwise_conv_wrapper_s4":
            if "dsp" not in self.required_capabilities() and "mve" not in self.required_capabilities():
                raise self.fault_unreachable(
                    kind,
                    f"{kernel_fn} only checks ctx->buf when the scratch sizer is non-zero, "
                    "which needs required_capabilities: [dsp] or [mve]",
                )

    def _render_depthwise(self, output_dir: Path, context: Dict[str, Any]) -> None:
        pool = depthwise_argument_pool(context)
        fault = self.fault_kind()
        if fault:
            self._check_fault_reachable(fault, context)
            context.update(self.fault_context())
            pool = depthwise_fault(pool, fault, context)
        self.render_harness_files(
            output_dir,
            stem="depthwise_conv",
            context=context,
            pool=pool,
            validation_key="ConvolutionFunctions/depthwise_conv/depthwise_conv.c.j2",
            label="Depthwise convolution",
        )

    def uses_reference(self) -> bool:
        return True

    def _select_cmsis_depthwise_conv_kernel(self) -> Dict[str, str]:
        info = resolve_depthwise_conv_kernel(
            activation_dtype=self.desc.get("activation_dtype", "S8"),
            weight_dtype=self.desc.get("weight_dtype", "S8"),
            cpu=self.target_cpu,
        )
        info.setdefault("kernel_needs_layout", info["input_c_type"] in {"float", "float16_t"})
        info.setdefault("buffer_size_needs_layout", info["input_c_type"] in {"float", "float16_t"})
        entry = self.desc.get("entry")
        if entry:
            info.update(
                resolve_entry(
                    "DepthwiseConv",
                    str(entry),
                    activation_dtype=self.desc.get("activation_dtype", "S8"),
                    weight_dtype=self.desc.get("weight_dtype", "S8"),
                    cpu=self.target_cpu,
                    desc=self.desc,
                )
            )
            check_entry_fault(self.desc, info)

        variant = str(self._hint().get("kernel_variant", "")).lower()
        if not variant:
            return info

        if info["input_c_type"] not in {"float", "float16_t"}:
            raise ValueError(f"DepthwiseConv kernel_variant hints are only supported for FP descriptors, got {variant}")

        suffix = "f16" if info["input_c_type"] == "float16_t" else "f32"
        if variant == "wrapper":
            info["kernel_fn"] = f"arm_depthwise_conv_wrapper_{suffix}"
            info["kernel_get_buffer_size_fn"] = f"arm_depthwise_conv_wrapper_{suffix}_get_buffer_size"
            info["kernel_needs_layout"] = False
            info["buffer_size_needs_layout"] = False
        else:
            raise ValueError(f"Unsupported DepthwiseConv kernel_variant hint: {variant}")

        return info
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for DepthwiseConv, with the golden from the C
        reference (TFLite's DepthwiseConvPerChannel; float exact then rounded once).
        """
        from helia_core_tester.generation.ops._shared.conv_reference import depthwise_reference_case
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_info = self._select_cmsis_depthwise_conv_kernel()
        float_kernel = kernel_info["input_c_type"] in {"float", "float16_t"}
        float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
        weight_c_type = kernel_info.get("weight_c_type")
        if weight_c_type is None:
            raise ValueError(
                f"Kernel dispatch missing weight_c_type for DepthwiseConv descriptor '{name}' "
                f"({self.desc.get('activation_dtype', 'S8')} x {self.desc.get('weight_dtype', 'S8')})"
            )
        weight_dtype = str(self.desc.get("weight_dtype", "S8")).upper()

        case = depthwise_reference_case(self, kernel_info, float_kernel, float_dtype)
        input_shape, output_shape = case["input_shape"], case["output_shape"]
        weights, biases, output_data = case["weights"], case["biases"], case["output"]
        params = case["params"]
        output_channels = int(output_shape[3])

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        # filter_dims.n is 1; the depth multiplier travels in dw_conv_params.ch_mult.
        filter_dims = {'n': 1, 'h': int(weights.shape[1]) if weights.ndim == 4 else int(self.desc['filter_shape'][0]),
                       'w': int(weights.shape[2]) if weights.ndim == 4 else int(self.desc['filter_shape'][1]),
                       'c': output_channels}
        dw_conv_params = {**params, 'ch_mult': int(case["depth_multiplier"])}
        has_biases = biases is not None and biases.size > 0

        if float_kernel:
            input_data = case["input"]
            output_data, nonfinite_context = self.apply_nonfinite_policy(
                output_data, reference=self.reference_probe,
                inputs=[input_data, case["filter_operand"], case["bias_operand"]]
            )
            element_size = np.dtype(float_dtype).itemsize
            buffer_size_max = max(
                1024,
                int(
                    (input_dims['n'] * input_dims['h'] * input_dims['w'] * input_dims['c']
                     + filter_dims['n'] * filter_dims['h'] * filter_dims['w'] * max(filter_dims['c'], 1)
                     + output_dims['n'] * output_dims['h'] * output_dims['w'] * output_dims['c']) * element_size
                ),
            )
            if kernel_info.get("entry_scratch_bytes") is not None:
                buffer_size_max = max(buffer_size_max, int(kernel_info["entry_scratch_bytes"]))

            context = {
                'name': name,
                'input_dims': input_dims,
                'filter_dims': filter_dims,
                'output_dims': output_dims,
                'dw_conv_params': dw_conv_params,
                'weights_array': builder.format_array_as_c_literal(weights),
                'biases_array': builder.format_array_as_c_literal(biases) if has_biases else "",
                'has_biases': has_biases,
                'weight_sum_array': "",
                'has_weight_sum': False,
                'input_data_array': builder.format_array_as_c_literal(np.asarray(input_data, dtype=float_dtype)),
                'expected_output_array': builder.format_array_as_c_literal(np.asarray(output_data, dtype=float_dtype)),
                'input_dtype': kernel_info["input_c_type"],
                'output_dtype': kernel_info["output_c_type"],
                'weight_dtype': weight_c_type,
                'bias_dtype': kernel_info["bias_c_type"],
                'kernel_fn': kernel_info["kernel_fn"],
                'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
                'kernel_needs_layout': bool(kernel_info.get("kernel_needs_layout", False)),
                'buffer_size_needs_layout': bool(kernel_info.get("buffer_size_needs_layout", False)),
                'kernel_layout': kernel_info.get("layout", "ARM_NN_LAYOUT_NHWC"),
                'call_style': kernel_info.get("call_style", "baseline"),
                'buffer_size_max': buffer_size_max,
                'force_no_scratch': bool(self._hint().get("force_no_scratch", False)),
                'entry_scratch_bytes': kernel_info.get("entry_scratch_bytes"),
                'entry_extra_sizers': kernel_info.get("entry_extra_sizers"),
                'float_kernel': True,
                'expected_status': self.expected_status(),
                'autovectorize_declines': bool(self.desc.get("autovectorize_declines", False)),
                'autovectorize_declines_if': autovectorize_declines_if(kernel_info["input_c_type"]),
                'dw_conv_params_type': (
                    'cmsis_nn_dw_conv_params_f16'
                    if kernel_info["input_c_type"] == "float16_t"
                    else 'cmsis_nn_dw_conv_params_f32'
                ),
                'dw_activation_min_literal': builder.format_float_literal(dw_conv_params['activation_min']),
                'dw_activation_max_literal': builder.format_float_literal(dw_conv_params['activation_max']),
            }
            context.update(nonfinite_context)
            self._render_depthwise(output_dir, context)
            self._write_cmake(output_dir)
            return

        bias_dtype = kernel_info["bias_c_type"]
        # The s8 optimized kernels take the weight sum (weights times input offset, plus bias) precomputed.
        weight_sum_array_str = ""
        has_weight_sum = False
        if weight_dtype != "S4" and kernel_info["input_c_type"] == "int8_t":
            kernel_matrix = weights.transpose(3, 0, 1, 2).reshape(output_channels, -1)
            weight_sum = vector_sum_s8(
                vector_data=kernel_matrix,
                vector_cols=kernel_matrix.shape[1],
                vector_rows=output_channels,
                lhs_offset=-int(case["input_zero_point"]),
                rhs_offset=0,
                bias_data=biases.astype(np.int32) if has_biases else None,
            ).astype(np.int32)
            weight_sum_array_str = builder.format_array_as_c_literal(weight_sum)
            has_weight_sum = True

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        buffer_size_max = builder.calculate_depthwise_buffer_size_max(
            input_dims, filter_dims, output_dims,
            output_dtype=activation_dtype
        )
        if kernel_info.get("entry_scratch_bytes") is not None:
            buffer_size_max = max(buffer_size_max, int(kernel_info["entry_scratch_bytes"]))
        # The 3x3 entries size their own path from the input dims; scratch takes the larger answer.
        buffer_size_max = max(buffer_size_max,
                              depthwise_3x3_scratch_bytes(kernel_info.get("entry_extra_sizers"), input_dims))
        # An entry gets the weight-sum context exactly when its prototype takes one; the wrapper
        # keeps the rule it always had.
        takes_weight_sum_ctx = kernel_info["kernel_fn"] == "arm_depthwise_conv_wrapper_s8"
        if kernel_info.get("entry_family") == "contract":
            from helia_core_tester.contract import render as contract_render
            from helia_core_tester.contract.bind import takes

            takes_weight_sum_ctx = takes(
                contract_render.load_current_contracts().require(kernel_info["kernel_fn"]), "weight_sum_ctx")

        self.reject_autovectorize_declines()
        context = {
            'name': name,
            'input_dims': input_dims,
            'filter_dims': filter_dims,
            'output_dims': output_dims,
            'dw_conv_params': dw_conv_params,
            'quant_params': case["quant_params"],
            'weights_array': builder.format_array_as_c_literal(weights),
            'biases_array': builder.format_array_as_c_literal(biases) if has_biases else "",
            'has_biases': has_biases,
            'weight_sum_array': weight_sum_array_str,
            'has_weight_sum': has_weight_sum,
            'input_data_array': builder.format_array_as_c_literal(case["input"]),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'weight_dtype': weight_c_type,
            'bias_dtype': bias_dtype,
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'call_style': kernel_info.get("call_style", "baseline"),
            'buffer_size_max': buffer_size_max,
            'takes_weight_sum_ctx': takes_weight_sum_ctx,
            'entry_scratch_bytes': kernel_info.get("entry_scratch_bytes"),
            'entry_extra_sizers': kernel_info.get("entry_extra_sizers"),
            'expected_status': self.expected_status(),
            'planar_supported': self.desc.get("planar_supported"),
            'planar_rule_fn': DEPTHWISE_CONV_S8_PLANAR_RULE,
        }
        self._render_depthwise(output_dir, context)
        self._write_cmake(output_dir)

    def _write_cmake(self, output_dir: Path) -> None:
        cmake_context = {
            'name': self.desc['name'],
            'operator': self.desc.get('operator', 'DepthwiseConv'),
            'operator_name': 'depthwise_conv'
        }
        (output_dir / "CMakeLists.txt").write_text(self.render_template("common/CMakeLists.txt.j2", cmake_context))
