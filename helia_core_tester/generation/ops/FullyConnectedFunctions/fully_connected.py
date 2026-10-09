"""
FullyConnected operation implementation with dtype-aware quantization.
"""

from typing import Dict, Any, Optional
import numpy as np
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.entry import check_entry_fault, resolve_entry
from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, Define, GuardedBuffer, Provider, SizeQuery
from helia_core_tester.generation.harness.faults import common_fault, struct_copy, with_fault
from helia_core_tester.generation.kernel_dispatch import autovectorize_declines_if, resolve_fully_connected_kernel
from helia_core_tester.core.cpu_targets import get_cpu_profile
from pathlib import Path


FC_VALIDATION_KEY = "FullyConnectedFunctions/fully_connected/fully_connected.c.j2"
PER_CHANNEL_S16_SIZER = "arm_fully_connected_per_channel_s16_get_buffer_size"


def fc_quant_argument(context: Dict[str, Any]) -> str:
    """The quant_params argument the kernel's prototype asks for: the wrappers take the generic
    cmsis_nn_quant_params, the others the per-tensor or per-channel struct the header carries."""
    from helia_core_tester.contract.bind import param_type
    from helia_core_tester.contract.render import load_current_contracts, require_bound_symbol

    n, kernel_fn = context["name"], context["kernel_fn"]
    per_channel = bool(context["quant_params"].get("per_channel"))
    ctype = param_type(require_bound_symbol(load_current_contracts(), kernel_fn), "quant_params")
    if "cmsis_nn_quant_params" in ctype:
        return f"&{n}_quant_params_wrapper"
    if ("per_channel" in ctype) != per_channel:
        raise ValueError(f"{n}: {kernel_fn} takes {ctype.strip()}, but this case's quantization is "
                         f"{'per-channel' if per_channel else 'per-tensor'}")
    return f"&{n}_quant_params"


def fc_packed_entry(context: Dict[str, Any]) -> bool:
    """Whether the kernel reads a weight stream packed ahead of the call in place of the weights,
    kernel sums and quantization (arm_fully_connected_per_channel_packed_s8)."""
    from helia_core_tester.contract.bind import takes
    from helia_core_tester.contract.render import load_current_contracts, require_bound_symbol

    if context.get("float_kernel") or context.get("entry_family") != "contract":
        return False
    return takes(require_bound_symbol(load_current_contracts(), context["kernel_fn"]), "packed_data")


def fc_packed_stream_bound(filter_dims: Dict[str, Any]) -> int:
    """A bound in whole words on the packed stream: ceil(C/4) blocks of four K-rows padded to 16
    bytes plus 48 bytes of parameters. The run-time size query must fit it."""
    return ((int(filter_dims['c']) + 3) * (int(filter_dims['n']) + 27) + 3) // 4 * 4


def fc_packed_provider(context: Dict[str, Any]) -> Provider:
    """The stream the packed entry reads, built the way its caller must build it: kernel sums with
    the input offset and bias folded in (filter offset 0), then the entry's own packer."""
    from helia_core_tester.contract.render import load_current_contracts

    n, kernel_fn = context["name"], context["kernel_fn"]
    upper = n.upper()
    filter_dims = context["filter_dims"]
    sizer = f"{kernel_fn}_get_packed_size"
    if load_current_contracts().find(sizer) is None:
        raise ValueError(f"{n}: {kernel_fn} packs a weight stream but this checkout declares no {sizer}")
    bound = fc_packed_stream_bound(filter_dims)
    bias = f"{n}_biases" if context.get("has_bias_array", context["has_biases"]) else "NULL"
    setup = "\n".join([
        "    // The stream is built the way the entry's caller must build it: kernel sums with the",
        "    // input offset and bias folded in (filter offset 0), then the entry's packer.",
        f"    {n}_packed_used = (size_t)packed_size;",
        f"    HELIA_GUARD_STAMP_SLACK({n}_packed, {n}_packed_used);",
        f"    arm_cmsis_nn_status pack_status = arm_vector_sum_s8(",
        f"        {n}_kernel_sum,",
        f"        {int(filter_dims['n'])},",
        f"        {int(filter_dims['c'])},",
        f"        {n}_weights,",
        f"        {n}_fc_params.input_offset,",
        "        0,",
        f"        {bias}",
        "    );",
        "    if (pack_status == ARM_CMSIS_NN_SUCCESS) {",
        f"        pack_status = {kernel_fn}_pack(",
        f"            &{n}_filter_dims,",
        f"            {n}_weights,",
        f"            {n}_kernel_sum,",
        f"            &{n}_quant_params,",
        f"            (int8_t *){n}_packed",
        "        );",
        "    }",
        "    if (pack_status != ARM_CMSIS_NN_SUCCESS) {",
        "        return pack_status;",
        "    }",
    ])
    return Provider(
        param="packed_data",
        expr=f"(const int8_t *){n}_packed",
        # The bound keeps the harness's scratch macro name: the hardware bridge reads it as the stream size.
        declarations=(Define(f"{upper}_BUFFER_SIZE_MAX", str(bound), comment="Packed weight stream (max upper bound; "
                             "actual size queried at runtime)"),
                      Declaration(f"{n}_packed_used", "size_t", "0", storage="static",
                                  comment="Bytes of the stream the size query claimed")),
        # Word storage keeps the stream 4-byte aligned, as the entry requires.
        buffers=(GuardedBuffer(f"{n}_kernel_sum", "int32_t", str(int(filter_dims['c'])), label="kernel sums"),
                 GuardedBuffer(f"{n}_packed", "int32_t", f"{upper}_PACKED_WORDS",
                               count_value=f"({upper}_BUFFER_SIZE_MAX / 4)", label="packed weights")),
        setup=setup,
        size_query=SizeQuery(fn=sizer, result_var="packed_size", capacity=f"{upper}_BUFFER_SIZE_MAX"),
    )


def fc_packed_checks(context: Dict[str, Any]) -> str:
    """The gate and the slack check of a packed case, run after the call with `failures` in scope."""
    n = context["name"]
    label = '{{ validation_label | default("Fully connected") }}'
    expected = 1 if str(context.get("expected_status", "ARM_CMSIS_NN_SUCCESS")) == "ARM_CMSIS_NN_SUCCESS" else 0
    return "\n".join([
        f'    HELIA_GUARD_CHECK_SLACK({n}_packed, "{label} packed weights slack", {n}_packed_used, failures);',
        "    // The gate the entry's caller asks first; a declined case expects it closed.",
        "    HELIA_VALIDATE_SCALAR_EQ_INT(",
        f'        "{label}",',
        '        "arm_nn_fc_packed_s8_supported",',
        f"        {expected},",
        f"        arm_nn_fc_packed_s8_supported(&{n}_fc_params, &{n}_quant_params, &{n}_filter_dims)",
        "    );",
    ])


def fc_sizer(context: Dict[str, Any]) -> Optional[str]:
    """The scratch query the case calls: the descriptor's own for a contract entry (entry_sizer,
    or none with entry_scratch), none for s4, the per-channel s16 one for the s16 wrapper with
    per-channel quantization, otherwise the dispatch's."""
    if context.get("float_kernel") or context.get("entry_family") == "contract":
        return context["kernel_get_buffer_size_fn"]
    if context["kernel_fn"] == "arm_fully_connected_s4":
        return None
    if (context["kernel_fn"] == "arm_fully_connected_wrapper_s16" and context["quant_params"].get("per_channel")
            and context["output_dtype"] == "int16_t"):
        return PER_CHANNEL_S16_SIZER
    return context["kernel_get_buffer_size_fn"]


def fc_argument_pool(context: Dict[str, Any]) -> ArgumentPool:
    """Every value a FullyConnected case can pass to a public FC kernel or its scratch query."""
    n = context["name"]
    float_kernel = bool(context.get("float_kernel"))
    fc = context["fc_params"]

    def dims(d: Dict[str, Any]) -> Dict[str, Any]:
        return {"n": d["n"], "h": d["h"], "w": d["w"], "c": d["c"]}

    if float_kernel:
        params_init = {"activation": {"min": context["fc_activation_min_literal"],
                                      "max": context["fc_activation_max_literal"]},
                       "weight_format": "ARM_NN_WEIGHT_FORMAT_STANDARD"}
    else:
        params_init = {"input_offset": fc["input_offset"], "filter_offset": fc["filter_offset"],
                       "output_offset": fc["output_offset"],
                       "activation": {"min": fc["activation_min"], "max": fc["activation_max"]}}
    out_c = context["filter_dims"]["c"]
    header = [
        Declaration(f"{n}_input_dims", "cmsis_nn_dims", dims(context["input_dims"]), comment="Input dimensions"),
        Declaration(f"{n}_filter_dims", "cmsis_nn_dims", dims(context["filter_dims"]), comment="Filter dimensions"),
        Declaration(f"{n}_bias_dims", "cmsis_nn_dims", {"n": 1, "h": 1, "w": 1, "c": out_c}, comment="Bias dimensions"),
        Declaration(f"{n}_output_dims", "cmsis_nn_dims", dims(context["output_dims"]), comment="Output dimensions"),
        Declaration(f"{n}_fc_params", context.get("fc_params_type") or "cmsis_nn_fc_params", params_init,
                    comment="Fully connected parameters"),
    ]
    values = {
        "ctx": f"&{n}_ctx", "fc_params": f"&{n}_fc_params", "input_dims": f"&{n}_input_dims",
        "filter_dims": f"&{n}_filter_dims", "filter_data": f"{n}_weights", "bias_dims": f"&{n}_bias_dims",
        "bias_data": f"{n}_biases" if context["has_biases"] else "NULL", "output_dims": f"&{n}_output_dims",
        "layout": context.get("kernel_layout") or "ARM_NN_LAYOUT_NHWC",
    }
    source = []
    packed = fc_packed_entry(context)
    if packed and not context["quant_params"].get("per_channel"):
        raise ValueError(f"{n}: entry {context['kernel_fn']!r} takes per-channel quantization only; "
                         "a single output channel is generated per-tensor")
    if not float_kernel:
        quant = context["quant_params"]
        if quant.get("per_channel"):
            header += [
                Declaration(f"{n}_multiplier", "int32_t", ArrayLiteral(quant["multiplier_array"]), storage="static",
                            array=True, comment="Per-channel quantization"),
                Declaration(f"{n}_shift", "int32_t", ArrayLiteral(quant["shift_array"]), storage="static", array=True),
                Declaration(f"{n}_quant_params", "cmsis_nn_per_channel_quant_params",
                            {"multiplier": f"{n}_multiplier", "shift": f"{n}_shift"}),
            ]
            wrapper = {"multiplier": f"(int32_t*){n}_multiplier", "shift": f"(int32_t*){n}_shift", "is_per_channel": "1"}
        else:
            header += [
                Declaration(f"{n}_multiplier_val", "int32_t", str(quant["multiplier"]), storage="static",
                            comment="Per-tensor quantization"),
                Declaration(f"{n}_shift_val", "int32_t", str(quant["shift"]), storage="static"),
                Declaration(f"{n}_quant_params", "cmsis_nn_per_tensor_quant_params",
                            {"multiplier": str(quant["multiplier"]), "shift": str(quant["shift"])}),
            ]
            wrapper = {"multiplier": f"(int32_t*)&{n}_multiplier_val", "shift": f"(int32_t*)&{n}_shift_val",
                       "is_per_channel": "0"}
        # The packed entry takes its quantization through the stream its packer builds.
        values["quant_params"] = f"&{n}_quant_params" if packed else fc_quant_argument(context)
        if values["quant_params"].endswith("_wrapper"):
            source.append(Declaration(f"{n}_quant_params_wrapper", "cmsis_nn_quant_params", wrapper,
                                      comment="The wrappers take per-channel and per-tensor parameters alike"))
    bias_ctype = context["bias_dtype"]
    header.append(Declaration(f"{n}_weights", context.get("weight_dtype") or "int8_t",
                              ArrayLiteral(context["weights_array"]), array=True, comment="Weights"))
    if context.get("has_bias_array", context["has_biases"]):
        header.append(Declaration(f"{n}_biases", bias_ctype, ArrayLiteral(context["biases_array"]), array=True,
                                  comment="Biases"))
    else:
        header.append(Declaration(f"{n}_biases", f"{bias_ctype}*", "NULL", comment="No biases"))
    context_setup = ""
    if context.get("has_weight_sum") and not packed:
        header.append(Declaration(f"{n}_weight_sum", "int32_t", ArrayLiteral(context["weight_sum_array"]), array=True,
                                  extent=str(out_c), comment="Precomputed weight sum for s8 fully connected"))
        context_setup = (f"    // The context carries the precomputed kernel sums, not scratch.\n"
                         f"    {n}_ctx.buf = (uint8_t *){n}_weight_sum;\n"
                         f"    {n}_ctx.size = {out_c} * 4;  // sizeof(int32_t) = 4")
    header += [
        Declaration(f"{n}_input", context["input_dtype"], ArrayLiteral(context["input_data_array"]), array=True,
                    comment="Input data (for testing)"),
        Declaration(f"{n}_expected_output", context["output_dtype"], ArrayLiteral(context["expected_output_array"]),
                    array=True, comment="Expected output (golden)"),
    ]
    output = context["output_dims"]
    if packed:
        # The entry takes no context: the stream is a provider's buffer, not the case's scratch.
        return ArgumentPool(
            name=n, values=values, header=header, source=source, providers=(fc_packed_provider(context),),
            output_count=f"({output['n']} * {output['h']} * {output['w']} * {output['c']})", benchmark=False,
            scratch_buffer=False, extra_checks=fc_packed_checks(context), includes=('"arm_nnsupportfunctions.h"',),
        )
    return ArgumentPool(
        name=n, values=values, header=header, source=source,
        output_count=f"({output['n']} * {output['h']} * {output['w']} * {output['c']})", benchmark=False,
        no_scratch=(not float_kernel and context["kernel_fn"] == "arm_fully_connected_s4"
                    and context.get("entry_family") != "contract"),
        context_setup=context_setup,
    )


def fc_fault(pool: ArgumentPool, kind: str, context: Dict[str, Any]) -> ArgumentPool:
    """The pool of a FullyConnected fault case: the passing pool with the faulted argument edited."""
    n = context["name"]
    edit = common_fault(pool, kind, layout=context.get("kernel_layout"))
    if edit is None and kind == "filter_n_mismatch":
        d = context["input_dims"]
        edit = struct_copy(pool, kind, "filter_dims", "cmsis_nn_dims", f"{n}_filter_dims",
                           {"n": int(d["h"]) * int(d["w"]) * int(d["c"]) + 1})
    if edit is None:
        raise ValueError(f"{n}: no FullyConnected fault edit for {kind!r}")
    return with_fault(pool, edit)


class OpFullyConnected(OperationBase):
    """
    FullyConnected operation.
    """

    FAULT_KINDS = (
        "null_ctx_buf",
        "small_ctx_size",
        "filter_n_mismatch",
        "invalid_layout",
    )

    def _check_fault_reachable(self, kind: str, context: Dict[str, Any]) -> None:
        """Reject fault kinds the selected fully-connected kernel route does not diagnose."""
        kernel_fn = context["kernel_fn"]
        if context.get("float_kernel"):
            if kind not in ("filter_n_mismatch", "invalid_layout"):
                raise self.fault_unreachable(kind, f"{kernel_fn} has no such guard")
            if kind == "invalid_layout" and not context.get("kernel_needs_layout", True):
                raise self.fault_unreachable(kind, f"{kernel_fn} takes no layout argument")
            return
        if kind in ("filter_n_mismatch", "invalid_layout"):
            raise self.fault_unreachable(kind, f"{kernel_fn} does not check {kind}")
        if kernel_fn == "arm_fully_connected_s4":
            raise self.fault_unreachable(kind, "arm_fully_connected_s4 validates no arguments")
        per_channel = bool(context["quant_params"].get("per_channel"))
        if kernel_fn == "arm_fully_connected_wrapper_s16":
            if not per_channel:
                raise self.fault_unreachable(kind, "arm_fully_connected_s16 validates no arguments")
            return
        if kind == "small_ctx_size":
            raise self.fault_unreachable(kind, f"{kernel_fn} does not check ctx->size")
        if "mve" not in self.required_capabilities():
            raise self.fault_unreachable(
                kind, f"{kernel_fn} only checks ctx->buf under ARM_MATH_MVEI; add required_capabilities: [mve]"
            )

    def _render_fully_connected(self, output_dir: Path, context: Dict[str, Any]) -> None:
        pool = fc_argument_pool(context)
        fault = self.fault_kind()
        if fault:
            self._check_fault_reachable(fault, context)
            context.update(self.fault_context())
            pool = fc_fault(pool, fault, context)
        self.render_harness_files(output_dir, stem="fully_connected", context=context, pool=pool,
                                  validation_key=FC_VALIDATION_KEY, label="Fully connected", sizer_fn=fc_sizer(context))

    def uses_reference(self) -> bool:
        return True

    def _generate_int_reference(self, output_dir: Path, kernel_info: Dict[str, Any]) -> None:
        """Render an s8/s16 (s8 or s4 weights) case whose golden comes from the
        TFLM reference fully-connected kernel."""
        from helia_core_tester.generation.reference import weighted
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        builder = TemplateContextBuilder()
        weight_dtype = str(self.desc.get("weight_dtype", "S8")).upper()
        extras = (self.desc.get("hint") or {}).get("extras") or {}
        filter_offset = int(extras.get("force_filter_offset", 0)) if weight_dtype != "S4" else 0

        spec = weighted.fc_spec(self.desc)
        case = weighted.build_weighted_case(
            self.desc, spec, self.reference_rng("weights"), self.generate_input_data, weights_offset=filter_offset
        )
        self._reference_call = case.call
        batch, features = spec.input_shape[0], spec.weight_shape[1]
        units = spec.weight_shape[0]
        input_dims = {"n": batch, "h": 1, "w": 1, "c": features}
        filter_dims = {"n": features, "h": 1, "w": 1, "c": units}
        output_dims = {"n": batch, "h": 1, "w": 1, "c": units}
        input_zp = case.input_quant.zero_point
        fc_params = {
            "input_offset": -input_zp,
            "filter_offset": filter_offset,
            "output_offset": case.output_quant.zero_point,
            "activation_min": case.act_min,
            "activation_max": case.act_max,
        }
        quant_params_dict = case.quant_context(builder)
        output_dtype = np.int16 if kernel_info["output_c_type"] == "int16_t" else np.int8

        weights = case.weights_q
        biases = case.bias_q
        # The MVE s8 kernel reads the bias and input-offset fold from precomputed
        # kernel sums (and a NULL bias); other builds read the bias itself.
        weight_sum = None
        has_weight_sum = False
        folded_bias = None
        if weight_dtype != "S4" and self._should_precompute_weight_sum(weights, output_dtype):
            from helia_core_tester.generation.ops.ConvolutionFunctions.depthwise_conv import vector_sum_s8

            weight_sum = vector_sum_s8(
                vector_data=weights,
                vector_cols=features,
                vector_rows=units,
                lhs_offset=-input_zp,
                rhs_offset=filter_offset,
                bias_data=biases,
            ).astype(np.int32)
            has_weight_sum = True
            if biases is not None:
                folded_bias = biases
                biases = None

        has_biases = biases is not None
        biases_array_str = builder.format_array_as_c_literal(biases) if has_biases else ""
        # A bias folded into the kernel sum still appears as an array in the header:
        # the hardware bridge reads it back to rebuild the kernel sum.
        has_bias_array = has_biases
        if folded_bias is not None:
            biases_array_str = builder.format_array_as_c_literal(folded_bias.astype(np.int32))
            has_bias_array = True

        is_s16 = kernel_info["input_c_type"] == "int16_t"
        if is_s16 and quant_params_dict.get("per_channel", False):
            buffer_size_max = output_dims["c"] * 4
        elif weight_dtype == "S4":
            buffer_size_max = 0
        else:
            buffer_size_max = builder.calculate_fc_buffer_size_max(
                filter_dims, output_dtype=self.desc.get("activation_dtype", "S8")
            )
        entry_scratch_bytes = kernel_info.get("entry_scratch_bytes")
        if entry_scratch_bytes is not None:
            buffer_size_max = max(buffer_size_max, int(entry_scratch_bytes))

        context = {
            "name": name,
            "entry_family": kernel_info.get("entry_family"),
            "entry_scratch_bytes": entry_scratch_bytes,
            "entry_extra_sizers": kernel_info.get("entry_extra_sizers"),
            "input_dims": input_dims,
            "filter_dims": filter_dims,
            "output_dims": output_dims,
            "fc_params": fc_params,
            "quant_params": quant_params_dict,
            "weights_array": builder.format_array_as_c_literal(case.weights_c),
            "biases_array": biases_array_str,
            "has_biases": has_biases,
            "has_bias_array": has_bias_array,
            "input_data_array": builder.format_array_as_c_literal(case.input_q.flatten()),
            "expected_output_array": builder.format_array_as_c_literal(case.output_q.flatten()),
            "input_dtype": kernel_info["input_c_type"],
            "output_dtype": kernel_info["output_c_type"],
            "bias_dtype": kernel_info["bias_c_type"],
            "kernel_fn": kernel_info["kernel_fn"],
            "kernel_get_buffer_size_fn": kernel_info["kernel_get_buffer_size_fn"],
            "call_style": kernel_info.get("call_style", "baseline"),
            "buffer_size_max": buffer_size_max,
            "weight_sum_array": builder.format_array_as_c_literal(weight_sum) if weight_sum is not None else "",
            "has_weight_sum": has_weight_sum,
            "expected_status": self.expected_status(),
            # The entry lives only on ns-cmsis-nn's MVE paths, so it declines on a build without them.
            "autovectorize_declines": bool(self.desc.get("autovectorize_declines", False)),
            "autovectorize_declines_if": autovectorize_declines_if(kernel_info["input_c_type"]),
        }
        self._render_fully_connected(output_dir, context)
        cmake_content = self.render_template(
            "common/CMakeLists.txt.j2",
            {"name": name, "operator": self.desc.get("operator", "FullyConnected"), "operator_name": "fully_connected"},
        )
        (output_dir / "CMakeLists.txt").write_text(cmake_content)

    def _select_cmsis_fc_kernel(self) -> Dict[str, str]:
        info = resolve_fully_connected_kernel(
            activation_dtype=self.desc.get('activation_dtype', 'S8'),
            weight_dtype=self.desc.get('weight_dtype', 'S8'),
            cpu=self.target_cpu,
        )
        entry = self.desc.get("entry")
        if entry:
            info.update(
                resolve_entry(
                    "FullyConnected",
                    str(entry),
                    activation_dtype=self.desc.get("activation_dtype", "S8"),
                    weight_dtype=self.desc.get("weight_dtype", "S8"),
                    cpu=self.target_cpu,
                    desc=self.desc,
                )
            )
            check_entry_fault(self.desc, info)
        return info

    def _supports_weight_sum(self) -> bool:
        """Check if platform supports weight sum optimization.

        arm_nn_vec_mat_mult_t_s8 reads the precomputed kernel sum only under
        ARM_MATH_MVEI; every other build reads the bias pointer instead. The
        generator folds the bias into the kernel sum and then passes a NULL
        bias, so claiming support on a non-MVE target drops the bias-add.
        """
        return get_cpu_profile(self.target_cpu).has_mve
    
    def _should_precompute_weight_sum(self, weights: Optional[np.ndarray], output_dtype: np.dtype) -> bool:
        """Determine if weight sum should be precomputed."""
        return (
            output_dtype == np.int8
            and self._supports_weight_sum()
            and weights is not None
            and weights.size > 0
        )
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for FullyConnected operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_fc_kernel()
        float_kernel = kernel_info["input_c_type"] in {"float", "float16_t"}
        if not float_kernel:
            self._generate_int_reference(output_dir, kernel_info)
            return

        from helia_core_tester.generation.reference import weighted

        builder = TemplateContextBuilder()
        float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
        spec = weighted.fc_spec(self.desc)
        input_shape = spec.input_shape
        case = weighted.build_float_case(
            self.desc, spec, self.reference_rng("weights"),
            lambda: self._sample_uniform(input_shape), float_dtype,
        )
        self._reference_call = case.call
        out_units, features = spec.weight_shape
        filter_dims = {'n': features, 'h': 1, 'w': 1, 'c': out_units}
        input_dims = {'n': int(input_shape[0]), 'h': 1, 'w': 1, 'c': features}
        output_dims = {'n': int(input_shape[0]), 'h': 1, 'w': 1, 'c': out_units}
        unquantized = {'scale': 1.0, 'zero_point': 0, 'per_channel': False}
        fc_params = builder.build_fc_params(self.desc, unquantized, unquantized, unquantized)
        weights, biases, input_data = case.weights, case.bias, case.input
        has_biases = biases is not None
        output_data, nonfinite_context = self.apply_nonfinite_policy(
            case.output, reference=case.reference, inputs=[input_data]
        )
        weights_array_str = builder.format_array_as_c_literal(weights) if weights is not None else ""
        biases_array_str = builder.format_array_as_c_literal(biases) if has_biases else ""
        input_data_array_str = builder.format_array_as_c_literal(np.asarray(input_data, dtype=float_dtype).flatten())
        expected_output_array_str = builder.format_array_as_c_literal(np.asarray(output_data, dtype=float_dtype).flatten())
        element_size = np.dtype(float_dtype).itemsize
        buffer_size_max = max(
            1024,
            int(
                (
                    input_dims['n'] * input_dims['c']
                    + filter_dims['n'] * filter_dims['c']
                    + output_dims['n'] * output_dims['c']
                ) * element_size
            ),
        )

        entry_scratch_bytes = kernel_info.get("entry_scratch_bytes")
        if entry_scratch_bytes is not None:
            buffer_size_max = max(buffer_size_max, int(entry_scratch_bytes))
        self.reject_autovectorize_declines()
        context = {
            'name': name,
            'input_dims': input_dims,
            'filter_dims': filter_dims,
            'output_dims': output_dims,
            'fc_params': fc_params,
            'weights_array': weights_array_str,
            'biases_array': biases_array_str,
            'has_biases': has_biases,
            'has_bias_array': has_biases,
            'input_data_array': input_data_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'weight_dtype': kernel_info.get("weight_c_type", kernel_info["input_c_type"]),
            'bias_dtype': kernel_info["bias_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'call_style': kernel_info.get("call_style", "baseline"),
            'buffer_size_max': buffer_size_max,
            'weight_sum_array': "",
            'has_weight_sum': False,
            'float_kernel': True,
            'fc_params_type': kernel_info.get("fc_params_type", 'cmsis_nn_fc_params_f32'),
            'kernel_layout': kernel_info.get("layout", "ARM_NN_LAYOUT_NHWC"),
            'entry_family': kernel_info.get("entry_family"),
            'entry_scratch_bytes': entry_scratch_bytes,
            'entry_extra_sizers': kernel_info.get("entry_extra_sizers"),
            'fc_activation_min_literal': builder.format_float_literal(fc_params['activation_min']),
            'fc_activation_max_literal': builder.format_float_literal(fc_params['activation_max']),
            'validation_mode': 'float',
        }
        context.update(nonfinite_context)
        self._render_fully_connected(output_dir, context)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'FullyConnected'),
            'operator_name': 'fully_connected'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        return
