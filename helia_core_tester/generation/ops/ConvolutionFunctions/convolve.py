"""Convolve operation implementation."""

from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
from helia_core_tester.generation.ops._shared.base import OperationBase

from helia_core_tester.generation.ops._shared.conv_reference import conv_reference_case
from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, GuardedBuffer, Provider
from helia_core_tester.generation.harness.faults import common_fault, null_context_buffer, struct_copy, with_fault
from helia_core_tester.generation.entry import check_entry_fault, resolve_entry
from helia_core_tester.generation.kernel_dispatch import autovectorize_declines_if, resolve_convolve_kernel



def convolve_argument_pool(context: Dict[str, Any], *, has_biases: bool, bias_is_struct: bool) -> ArgumentPool:
    """Every value a Convolve case can pass to a public Convolve kernel or its sizer."""
    n = context["name"]
    upper = n.upper()
    float_kernel = bool(context["float_kernel"])
    conv = context["conv_params"]

    def dims(d: Dict[str, Any]) -> Dict[str, Any]:
        return {"n": d["n"], "h": d["h"], "w": d["w"], "c": d["c"]}

    geometry = {
        "stride": {"w": conv["stride_w"], "h": conv["stride_h"]},
        "dilation": {"w": conv["dilation_w"], "h": conv["dilation_h"]},
        "padding": {"w": conv["pad_w"], "h": conv["pad_h"]},
    }
    if float_kernel:
        params_init = {**geometry,
                       "activation": {"min": context["conv_activation_min_literal"],
                                      "max": context["conv_activation_max_literal"]},
                       "weight_format": context.get("weight_format_macro") or "ARM_NN_WEIGHT_FORMAT_STANDARD"}
    else:
        params_init = {"input_offset": conv["input_offset"], "output_offset": conv["output_offset"], **geometry,
                       "activation": {"min": conv["activation_min"], "max": conv["activation_max"]}}
    header = [
        Declaration(f"{n}_input_dims", "cmsis_nn_dims", dims(context["input_dims"]), comment="Input dimensions"),
        Declaration(f"{n}_filter_dims", "cmsis_nn_dims", dims(context["filter_dims"]), comment="Filter dimensions"),
        Declaration(f"{n}_output_dims", "cmsis_nn_dims", dims(context["output_dims"]), comment="Output dimensions"),
        Declaration(f"{n}_conv_params", context.get("conv_params_type") or "cmsis_nn_conv_params", params_init,
                    comment="Convolution parameters"),
    ]
    values = {
        "ctx": f"&{n}_ctx", "conv_params": f"&{n}_conv_params", "input_dims": f"&{n}_input_dims",
        "filter_dims": f"&{n}_filter_dims", "output_dims": f"&{n}_output_dims", "filter_data": f"{n}_weights",
        "bias_dims": f"&{n}_bias_dims" if has_biases else "NULL",
        "bias_data": (f"&{n}_bias_data" if bias_is_struct else f"{n}_biases") if has_biases else "NULL",
        "upscale_dims": "NULL", "layout": context["kernel_layout"],
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
    source = []
    if has_biases:
        source.append(Declaration(f"{n}_bias_dims", "cmsis_nn_dims",
                                  {"n": 0, "h": 0, "w": 0, "c": context["filter_dims"]["n"]}, comment="Bias dimensions"))
        if not float_kernel and bias_is_struct:
            source.append(Declaration(f"{n}_bias_data", "cmsis_nn_bias_data",
                                      {"data": f"{n}_biases", "is_int32_bias": bias_ctype == "int32_t"},
                                      comment="s16 requires cmsis_nn_bias_data struct wrapper"))
    weight_sum = Provider(
        param="weight_sum_ctx",
        expr=f"&{n}_weight_sum_ctx",
        declarations=(Declaration(f"{n}_weight_sum_ctx", "cmsis_nn_context", storage="static",
                                  comment="Weight sum context (precomputed input-offset/bias fold)"),),
        buffers=(GuardedBuffer(f"{n}_weight_sum_buffer", "uint8_t", f"{upper}_WEIGHT_SUM_BUFFER_SIZE",
                               count_value=f"({context['output_dims']['c']} * sizeof(int32_t))", label="weight_sum"),),
        setup=(f"    // Initialize weight sum context and buffer\n"
               f"    {n}_weight_sum_ctx.buf = {n}_weight_sum_buffer;\n"
               f"    {n}_weight_sum_ctx.size = {upper}_WEIGHT_SUM_BUFFER_SIZE;\n\n"
               f"    // Calculate weight sum: pre-computes weight * input_offset + bias\n"
               f"    int32_t lhs_offset = (int32_t){n}_conv_params.input_offset;\n"
               f"    arm_convolve_weight_sum((int32_t*){n}_weight_sum_ctx.buf,\n"
               f"                            {n}_weights,\n"
               f"                            &{n}_input_dims,\n"
               f"                            &{n}_filter_dims,\n"
               f"                            &{n}_output_dims,\n"
               f"                            lhs_offset,\n"
               f"                            {n + '_biases' if has_biases else 'NULL'});"),
    )
    output = context["output_dims"]
    return ArgumentPool(
        name=n, values=values, header=header, source=source, providers=(weight_sum,),
        output_count=f"({output['n']} * {output['h']} * {output['w']} * {output['c']})",
    )


def _depth(context: Dict[str, Any], dims: str) -> int:
    return int(context[dims]["c"])


# Shapes that break the whole-group rule (ns-cmsis-nn#725): the dims struct the kernel gets
# and the channel count it carries.
_DEPTH_FAULTS = {
    "zero_filter_depth": lambda c: ("filter_dims", 0),
    "filter_deeper_than_input": lambda c: ("filter_dims", 2 * _depth(c, "input_dims")),
    # One filter depth plus one channel: input_ch / filter_ch = 1 group that leaves a channel over.
    "partial_filter_group": lambda c: ("input_dims", _depth(c, "filter_dims") + 1),
    "negative_output_depth": lambda c: ("output_dims", -_depth(c, "output_dims")),
    # One output channel past a whole number of groups.
    "output_not_whole_groups": lambda c: ("output_dims", _depth(c, "output_dims") + 1),
    "negative_input_depth": lambda c: ("input_dims", -_depth(c, "input_dims")),
    "negative_filter_depth": lambda c: ("filter_dims", -_depth(c, "filter_dims")),
}


def convolve_fault(pool: ArgumentPool, kind: str, context: Dict[str, Any]) -> ArgumentPool:
    """The pool of a Convolve fault case: the passing pool with the faulted argument edited."""
    n = context["name"]
    edit = common_fault(pool, kind, layout=context.get("kernel_layout"))
    if edit is None and kind == "zero_stride":
        edit = struct_copy(pool, kind, "conv_params", context.get("conv_params_type") or "cmsis_nn_conv_params",
                           f"{n}_conv_params", {"stride.w": 0})
    elif edit is None and kind == "channel_group_mismatch":
        # groups = input_ch / filter_ch = 2 does not divide the odd input_ch.
        edit = struct_copy(pool, kind, "input_dims", "cmsis_nn_dims", f"{n}_input_dims",
                           {"c": 2 * int(context["filter_dims"]["c"]) + 1})
    elif edit is None and kind == "null_weight_sum_ctx":
        edit = null_context_buffer(pool, kind, "weight_sum_ctx", f"{n}_weight_sum_ctx")
    elif edit is None and kind in _DEPTH_FAULTS:
        param, value = _DEPTH_FAULTS[kind](context)
        edit = struct_copy(pool, kind, param, "cmsis_nn_dims", f"{n}_{param}", {"c": value})
    if edit is None:
        raise ValueError(f"{n}: no Convolve fault edit for {kind!r}")
    return with_fault(pool, edit)


class OpConvolve(OperationBase):
    """Convolve operation."""

    FAULT_KINDS = (
        "null_ctx_buf",
        "null_weight_sum_ctx",
        "zero_stride",
        "channel_group_mismatch",
        "null_input",
        "null_output",
        "invalid_layout",
        "zero_filter_depth",
        "filter_deeper_than_input",
        "partial_filter_group",
        "negative_output_depth",
        "output_not_whole_groups",
        "negative_input_depth",
        "negative_filter_depth",
    )
    # Shapes that break the whole-group rule arm_convolve_wrapper_s16 checks first (ns-cmsis-nn#725).
    S16_GROUP_FAULTS = (
        "zero_filter_depth",
        "filter_deeper_than_input",
        "partial_filter_group",
        "negative_output_depth",
        "output_not_whole_groups",
        "negative_input_depth",
        "negative_filter_depth",
    )

    def _hint(self) -> Dict[str, Any]:
        hint = self.desc.get("hint", {})
        return hint if isinstance(hint, dict) else {}

    def _check_fault_reachable(self, kind: str, context: Dict[str, Any]) -> None:
        kernel_fn = context["kernel_fn"]
        if kind in self.S16_GROUP_FAULTS and kernel_fn != "arm_convolve_wrapper_s16":
            raise self.fault_unreachable(kind, f"{kernel_fn} is not covered by the s16 whole-group rule")
        if context["float_kernel"]:
            if kind in ("null_ctx_buf", "null_weight_sum_ctx", "zero_stride", "channel_group_mismatch"):
                raise self.fault_unreachable(kind, f"{kernel_fn} has no such guard")
            if kind == "invalid_layout" and not context["kernel_needs_layout"]:
                raise self.fault_unreachable(kind, f"{kernel_fn} takes no layout argument")
            return
        if kind in ("null_input", "null_output", "invalid_layout"):
            raise self.fault_unreachable(kind, f"{kernel_fn} does not check pointers or layout")
        if kind == "null_weight_sum_ctx" and kernel_fn != "arm_convolve_wrapper_s8":
            raise self.fault_unreachable(kind, f"{kernel_fn} takes no weight-sum context")
        if kind == "partial_filter_group" and int(context["filter_dims"]["c"]) < 2:
            raise self.fault_unreachable(kind, "needs a filter depth of at least 2")
        if kind == "output_not_whole_groups" and int(context["input_dims"]["c"]) < 2 * int(context["filter_dims"]["c"]):
            raise self.fault_unreachable(kind, "needs at least two groups")
        if kind == "channel_group_mismatch" and kernel_fn == "arm_convolve_wrapper_s4":
            raise self.fault_unreachable(kind, f"{kernel_fn} has no group divisibility guard")
        input_dims = context["input_dims"]
        filter_dims = context["filter_dims"]
        conv_params = context["conv_params"]
        if kind == "zero_stride":
            is_1xn = (
                kernel_fn == "arm_convolve_wrapper_s8"
                and input_dims["h"] == 1
                and filter_dims["h"] == 1
                and filter_dims["w"] > 1
                and int(conv_params["dilation_w"]) == 1
                and input_dims["c"] == filter_dims["c"]
            )
            if not is_1xn:
                raise self.fault_unreachable(
                    kind, "only the s8 1xN route (input h == 1, filter 1xN with N > 1, dilation.w == 1) checks stride.w"
                )

    @staticmethod
    def _pack_nt_n_weights(weights: np.ndarray, block_cols: int) -> np.ndarray:
        """Pack OHWI weights into CMSIS-NN `[K][N-block]` FP RHS layout."""
        if weights is None:
            raise ValueError("Cannot pack missing convolution weights")
        if len(weights.shape) != 4:
            raise ValueError(f"NT_N_PACKED convolution weights must be OHWI rank-4, got {weights.shape}")

        out_ch = int(weights.shape[0])
        rhs_cols = int(np.prod(weights.shape[1:]))
        rhs_rows_rounded = ((out_ch + block_cols - 1) // block_cols) * block_cols
        weights_matrix = weights.reshape(out_ch, rhs_cols)
        packed = np.zeros((rhs_rows_rounded // block_cols, rhs_cols, block_cols), dtype=weights.dtype)

        for block_idx, out_base in enumerate(range(0, rhs_rows_rounded, block_cols)):
            for k in range(rhs_cols):
                for lane in range(block_cols):
                    out_ch_idx = out_base + lane
                    if out_ch_idx < out_ch:
                        packed[block_idx, k, lane] = weights_matrix[out_ch_idx, k]

        return packed.reshape(-1)

    def uses_reference(self) -> bool:
        return True

    def _reference_case(self, kernel_info: Dict[str, Any], float_kernel: bool, float_dtype) -> Dict[str, Any]:
        """Input, weights, bias and their quantization drawn from the case seed, and the golden
        from the reference Conv2D (TFLite's ConvPerChannel; float exact then rounded once)."""
        return conv_reference_case(self, kernel_info, float_kernel, float_dtype)

    def _select_cmsis_convolve_kernel(self) -> Dict[str, str]:
        info = resolve_convolve_kernel(
            activation_dtype=self.desc.get("activation_dtype", "S8"),
            weight_dtype=self.desc.get("weight_dtype", "S8"),
            cpu=self.target_cpu,
        )
        info.setdefault("kernel_needs_layout", info["input_c_type"] in {"float", "float16_t"})
        info.setdefault("buffer_size_needs_layout", info["input_c_type"] in {"float", "float16_t"})

        hint = self._hint()

        variant = str(hint.get("kernel_variant", "")).lower()
        entry = self.desc.get("entry")
        if entry:
            if variant:
                raise ValueError(
                    f"{self.desc.get('name')}: entry {entry!r} is not supported with a kernel_variant hint"
                )
            info.update(
                resolve_entry(
                    "Convolve",
                    str(entry),
                    activation_dtype=self.desc.get("activation_dtype", "S8"),
                    weight_dtype=self.desc.get("weight_dtype", "S8"),
                    cpu=self.target_cpu,
                    desc=self.desc,
                )
            )
            check_entry_fault(self.desc, info)
            return info
        if not variant:
            return info

        if info["input_c_type"] not in {"float", "float16_t"}:
            raise ValueError(f"Convolve kernel_variant hints are only supported for FP descriptors, got {variant}")

        suffix = "f16" if info["input_c_type"] == "float16_t" else "f32"
        if variant == "wrapper":
            info["kernel_fn"] = f"arm_convolve_wrapper_{suffix}"
            info["kernel_get_buffer_size_fn"] = f"arm_convolve_wrapper_{suffix}_get_buffer_size"
            info["kernel_needs_layout"] = False
            info["buffer_size_needs_layout"] = False
        elif variant == "direct_1x1":
            info["kernel_fn"] = f"arm_convolve_1x1_{suffix}"
            info["kernel_get_buffer_size_fn"] = f"arm_convolve_1x1_{suffix}_get_buffer_size"
            info["kernel_needs_layout"] = True
            info["buffer_size_needs_layout"] = True
        elif variant == "direct_1_x_n":
            info["kernel_fn"] = f"arm_convolve_1_x_n_{suffix}"
            info["kernel_get_buffer_size_fn"] = f"arm_convolve_1_x_n_{suffix}_get_buffer_size"
            info["kernel_needs_layout"] = True
            info["buffer_size_needs_layout"] = True
        else:
            raise ValueError(f"Unsupported Convolve kernel_variant hint: {variant}")

        return info

    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Convolve.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']

        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_convolve_kernel()
        float_kernel = kernel_info["input_c_type"] in {"float", "float16_t"}
        float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
        weight_c_type = kernel_info.get("weight_c_type")
        if weight_c_type is None:
            raise ValueError(
                f"Kernel dispatch missing weight_c_type for Convolve descriptor '{name}' "
                f"({self.desc.get('activation_dtype', 'S8')} x {self.desc.get('weight_dtype', 'S8')})"
            )

        case = self._reference_case(kernel_info, float_kernel, float_dtype)
        input_shape, output_shape = case["input_shape"], case["output_shape"]
        input_q, weights, biases, output_data = case["input"], case["weights"], case["biases"], case["output"]
        conv_params, quant_params_dict = case["conv_params"], case["quant_params"]
        has_biases = biases is not None and biases.size > 0
        bias_dtype = kernel_info["bias_c_type"]

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        # OHWI with I = input depth / groups (CMSIS derives the group count from it); the
        # descriptor's HWIO input depth is not relied on, as nothing ever validated it.
        o, kh, kw, i = case["weights_shape"]
        filter_dims = {'n': o, 'h': kh, 'w': kw, 'c': i}

        weight_format_macro = "ARM_NN_WEIGHT_FORMAT_STANDARD"
        if float_kernel:
            weight_format = str(self._hint().get("weight_format", "STANDARD")).upper()
            if weight_format in {"NT_N_PACKED", "ARM_NN_WEIGHT_FORMAT_NT_N_PACKED"}:
                block_cols = 8 if kernel_info["input_c_type"] == "float16_t" else 4
                weights = self._pack_nt_n_weights(weights, block_cols)
                weight_format_macro = "ARM_NN_WEIGHT_FORMAT_NT_N_PACKED"
            elif weight_format not in {"STANDARD", "ARM_NN_WEIGHT_FORMAT_STANDARD"}:
                raise ValueError(f"Unsupported Convolve weight_format hint: {weight_format}")

        if float_kernel:
            output_data, nonfinite_context = self.apply_nonfinite_policy(
                output_data, reference=self.reference_probe, inputs=[input_q, case["filter_operand"], case["bias_operand"]]
            )
        else:
            nonfinite_context = {}

        # Format arrays
        weights_array_str = builder.format_array_as_c_literal(weights) if weights is not None else ""
        biases_array_str = builder.format_array_as_c_literal(biases) if has_biases else ""
        input_data_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)

        # Calculate buffer size max (conservative estimate)
        # Use activation_dtype to determine if this is S8 or S16 convolution
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if float_kernel:
            element_size = np.dtype(float_dtype).itemsize
            # The patch-GEMM sizers ask for 8 packed patch rows (ARM_NN_CONV_NHWC_PATCH_GEMM_F*_MAX_TILE_ROWS),
            # which exceeds the tensor total for a long patch over a small input.
            patch_tile = 8 * filter_dims['h'] * filter_dims['w'] * input_dims['c'] * element_size
            buffer_size_max = max(
                1024,
                patch_tile,
                int(
                    (input_dims['n'] * input_dims['h'] * input_dims['w'] * input_dims['c']
                     + filter_dims['n'] * filter_dims['h'] * filter_dims['w'] * max(filter_dims['c'], 1)
                     + output_dims['n'] * output_dims['h'] * output_dims['w'] * output_dims['c']) * element_size
                ),
            )
        else:
            buffer_size_max = builder.calculate_buffer_size_max(
                input_dims, filter_dims, output_dims, 
                output_dtype=activation_dtype
            )
        entry_scratch_bytes = kernel_info.get("entry_scratch_bytes")
        if entry_scratch_bytes is not None:
            buffer_size_max = max(buffer_size_max, int(entry_scratch_bytes))

        # An entry gets the weight-sum pre-pass and the struct-typed bias exactly when its
        # prototype takes them; the wrappers keep the rules they always had.
        contract_decl = None
        if kernel_info.get("entry_family") == "contract":
            from helia_core_tester.contract import render as contract_render
            from helia_core_tester.contract.bind import takes

            contract_decl = contract_render.load_current_contracts().require(kernel_info["kernel_fn"])
        bias_is_struct = kernel_info["kernel_fn"] == "arm_convolve_wrapper_s16" or (
            contract_decl is not None and takes(contract_decl, "bias_data")
            and "cmsis_nn_bias_data" in next(p.c_type for p in contract_decl.params if p.name in ("bias_data", "bias"))
        )

        # Build template context
        context = {
            'name': name,
            'input_dims': input_dims,
            'filter_dims': filter_dims,
            'output_dims': output_dims,
            'conv_params': conv_params,
            'weights_array': weights_array_str,
            'biases_array': biases_array_str,
            'has_biases': has_biases,
            'input_data_array': input_data_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'weight_dtype': weight_c_type,
            'bias_dtype': bias_dtype,
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'kernel_needs_layout': bool(kernel_info.get("kernel_needs_layout", False)),
            'buffer_size_needs_layout': bool(kernel_info.get("buffer_size_needs_layout", False)),
            'call_style': kernel_info.get("call_style", "baseline"),
            'buffer_size_max': buffer_size_max,
            'float_kernel': float_kernel,
            'weight_format_macro': weight_format_macro,
            'conv_params_type': (
                'cmsis_nn_conv_params_f16'
                if kernel_info["input_c_type"] == "float16_t"
                else ('cmsis_nn_conv_params_f32' if float_kernel else 'cmsis_nn_conv_params')
            ),
            'kernel_layout': kernel_info.get("layout", "ARM_NN_LAYOUT_NHWC"),
            # Selects common/standalone/benchmark.j2's backend: "fvp" (default,
            # DWT-only) or "hardware" (DWT + PMU, Apollo510/Cortex-M55 real
            # silicon). No CLI flag exists yet for this -- set via env var so
            # benchmarking scripts can select it without deeper Config/CLI plumbing.
            'benchmark_target': os.environ.get("HELIA_BENCH_TARGET", "fvp"),
            'entry_family': kernel_info.get("entry_family"),
            'conv_s8_weight_sum': kernel_info["kernel_fn"] == "arm_convolve_wrapper_s8"
            or (contract_decl is not None and takes(contract_decl, "weight_sum_ctx")),
            'bias_is_struct': bias_is_struct,
            'entry_scratch_bytes': entry_scratch_bytes,
            'entry_extra_sizers': kernel_info.get("entry_extra_sizers"),
            'expected_status': self.expected_status(),
            # The entry lives only on ns-cmsis-nn's MVE paths, so it declines on a build without them.
            'autovectorize_declines': bool(self.desc.get("autovectorize_declines", False)),
            'autovectorize_declines_if': autovectorize_declines_if(kernel_info["input_c_type"]),
        }
        if float_kernel:
            context['conv_activation_min_literal'] = builder.format_float_literal(conv_params['activation_min'])
            context['conv_activation_max_literal'] = builder.format_float_literal(conv_params['activation_max'])
        else:
            context['quant_params'] = quant_params_dict
        context.update(nonfinite_context)

        pool = convolve_argument_pool(context, has_biases=has_biases, bias_is_struct=bias_is_struct)
        fault = self.fault_kind()
        if fault:
            self._check_fault_reachable(fault, context)
            context.update(self.fault_context())
            pool = convolve_fault(pool, fault, context)

        self.render_harness_files(
            output_dir,
            stem="convolve",
            context=context,
            pool=pool,
            validation_key="ConvolutionFunctions/convolve/convolve.c.j2",
            label="Convolution",
        )

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Convolve'),
            'operator_name': 'convolve'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
