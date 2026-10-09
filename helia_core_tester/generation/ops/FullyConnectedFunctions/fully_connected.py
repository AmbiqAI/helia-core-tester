"""
FullyConnected operation implementation with dtype-aware quantization.
"""

from typing import Dict, Any, Optional
import numpy as np
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.fixed_batch import converter_for_batched_model
from helia_core_tester.generation.entry import check_entry_fault, resolve_entry
from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, Define, GuardedBuffer, Provider, SizeQuery
from helia_core_tester.generation.harness.faults import common_fault, struct_copy, with_fault
from helia_core_tester.generation.kernel_dispatch import autovectorize_declines_if, resolve_fully_connected_kernel
from helia_core_tester.core.cpu_targets import get_cpu_profile
import keras
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
        # Integer cases take their golden from the TFLM reference kernels; float
        # cases stay on the converter path until the float suites move.
        return str(self.desc.get("activation_dtype", "S8")).upper() in {"S8", "S16"}

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

    def needs_keras_model(self) -> bool:
        return True
    
    def build_keras_model(self):
        """Build Keras model for FullyConnected operation."""
        input_shape = self.desc['input_shape']
        filter_shape = self.desc['filter_shape']
        
        # Handle both 2D [batch, features] and 4D [batch, h, w, c] input shapes
        if len(input_shape) == 2:
            # 2D input: [batch, features]
            input_features = input_shape[1]
            batch_size = input_shape[0]
            needs_flatten = False
        else:
            # 4D input: [batch, h, w, c]
            # Calculate total features: h * w * c
            input_features = input_shape[1] * input_shape[2] * input_shape[3]
            batch_size = input_shape[0]
            needs_flatten = True
        
        # Extract output units from filter_shape
        if len(filter_shape) == 2:
            output_units = filter_shape[0]
        else:
            output_units = filter_shape[0]
        
        # Get activation and use_bias from descriptor
        activation_str = self.desc.get('activation', 'NONE')
        use_bias = self.desc.get('use_bias', True)
        
        # Build model with input layer matching descriptor shape
        if needs_flatten:
            # 4D input: [batch, h, w, c]
            inputs = keras.layers.Input(shape=input_shape[1:], batch_size=batch_size, name='input')
            # Flatten to [batch, h*w*c]
            x = keras.layers.Flatten()(inputs)
        else:
            # 2D input: [batch, features]
            inputs = keras.layers.Input(shape=(input_features,), batch_size=batch_size, name='input')
            x = inputs
        
        # A zero bias_initializer (Dense's default) produces an all-zero bias
        # tensor, which the TFLite converter's constant-folding optimizer
        # strips from the graph entirely -- the generated CMSIS-NN test then
        # calls the kernel with a NULL bias pointer, leaving the bias-add
        # path completely untested. Float cases only: integer cases use the
        # reference kernels.
        if not use_bias:
            bias_initializer = 'zeros'
        else:
            bias_initializer = keras.initializers.RandomUniform(minval=-0.25, maxval=0.25, seed=self.seed)

        # Dense layer without activation (we'll apply activation separately if needed)
        x = keras.layers.Dense(
            output_units,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=keras.initializers.GlorotUniform(seed=1234),
            bias_initializer=bias_initializer,
            name='dense'
        )(x)
        
        # Apply activation if specified
        if activation_str == 'RELU':
            x = keras.layers.ReLU()(x)
        elif activation_str == 'RELU6':
            x = keras.layers.ReLU(max_value=6)(x)
        elif activation_str == 'TANH':
            x = keras.layers.Activation('tanh')(x)
        elif activation_str == 'SIGMOID':
            x = keras.layers.Activation('sigmoid')(x)
        elif activation_str != 'NONE':
            raise ValueError(f"Unsupported activation: {activation_str}")
        
        model = keras.models.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        self.round_float16_weights(model)
        import tensorflow as tf
        
        # Create converter
        converter = converter_for_batched_model(model, [self.desc['input_shape']])
        
        # Apply quantization based on activation_dtype
        activation_dtype = str(self.desc.get('activation_dtype', 'S8')).upper()
        
        if activation_dtype == 'S8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif activation_dtype == 'S16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        elif activation_dtype == 'FP16':
            converter.optimizations = []
            converter.target_spec.supported_types = [tf.float16]
        elif activation_dtype == 'FP32':
            converter.optimizations = []
        
        # Force per-tensor quantization when requested
        force_per_tensor = bool(self.desc.get("hint", {}).get("force_per_tensor", False))
        if force_per_tensor and hasattr(converter, "_experimental_disable_per_channel"):
            converter._experimental_disable_per_channel = True

        # Generate representative dataset
        def representative_data_gen():
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
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

    def _find_fully_connected_op_index(self, model: Any, subgraph: Any) -> int:
        """Find the FULLY_CONNECTED operator index in the subgraph (fallback to 0)."""
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")

        try:
            from ai_edge_litert import schema_py_generated as litert

            for op_idx, op in enumerate(subgraph.operators):
                opcode = model.operatorCodes[op.opcodeIndex]
                if opcode.builtinCode == litert.BuiltinOperator.FULLY_CONNECTED:
                    return op_idx
        except Exception:
            pass

        return 0
    
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

        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        fc_op_index = self._find_fully_connected_op_index(model, subgraph)
        op_tensors = get_operator_tensors_from_litert(model, subgraph, fc_op_index)
        
        # Extract shapes from LiteRT
        input_shape = op_tensors['inputs'][0]['shape']
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        # Extract quantization parameters from LiteRT
        input_quant_litert = op_tensors['inputs'][0]['quantization']
        output_quant_litert = op_tensors['outputs'][0]['quantization']
        
        input_quant = {
            'scale': input_quant_litert.get('scale', 1.0),
            'zero_point': input_quant_litert.get('zero_point', 0),
            'per_channel': input_quant_litert.get('per_channel', False)
        }
        output_quant = {
            'scale': output_quant_litert.get('scale', 1.0),
            'zero_point': output_quant_litert.get('zero_point', 0),
            'per_channel': output_quant_litert.get('per_channel', False)
        }
        
        # Extract weights and biases for the actual FULLY_CONNECTED op.
        # For 4D inputs, op[0] may be RESHAPE, so using operator index 0 can be wrong.
        weights = op_tensors.get('weights')
        biases = op_tensors.get('biases')
        
        # Get weight quantization from LiteRT
        from helia_core_tester.generation.utils.litert_utils import (
            get_tensor_data_from_litert, get_tensor_quantization_from_litert,
            get_tensor_shape_from_litert
        )
        
        weight_quant = None
        if weights is not None:
            # Search all tensors to find the one matching our weights
            input_indices = set(subgraph.inputs)
            output_indices = set(subgraph.outputs)
            
            for tensor_idx, tensor in enumerate(subgraph.tensors):
                if tensor_idx in input_indices or tensor_idx in output_indices:
                    continue
                
                tensor_data = get_tensor_data_from_litert(tensor, model)
                tensor_shape = get_tensor_shape_from_litert(tensor)
                
                if (tensor_data is not None and tensor_shape is not None and 
                    len(tensor_shape) > 1 and tensor_data.shape == weights.shape and
                    np.array_equal(tensor_data, weights)):
                    weight_quant = get_tensor_quantization_from_litert(tensor)
                    break
            
            # Fallback: check operator inputs for weight tensor
            if weight_quant is None:
                for input_tensor_info in op_tensors['inputs']:
                    if input_tensor_info['data'] is not None and len(input_tensor_info['shape']) > 1:
                        weight_quant = input_tensor_info.get('quantization')
                        break
        
        # Prepare weight quantization dict
        if weight_quant is None:
            # No weight-tensor quantization could be recovered from the
            # converted model. Silently substituting the output tensor's
            # quantization would produce incorrect per-channel/per-tensor
            # weight scales and a wrong multiplier, potentially masking a real
            # kernel/golden mismatch. Fail loudly instead.
            raise RuntimeError(
                f"FullyConnected descriptor '{name}' weight quantization could "
                "not be recovered from the converted TFLite model; refusing to "
                "substitute unrelated output quantization"
            )
        
        weight_quant_dict = {
            'scale': weight_quant.get('scale', 1.0),
            'zero_point': weight_quant.get('zero_point', 0),
            'per_channel': weight_quant.get('per_channel', False)
        }
        
        # Validate weights shape
        if weights is not None:
            filter_shape = tuple(weights.shape)
            if len(filter_shape) == 1:
                # Try to infer 2D shape
                if len(output_shape) == 2:
                    output_units = output_shape[1]
                    input_features = filter_shape[0] // output_units if filter_shape[0] % output_units == 0 else filter_shape[0]
                    if filter_shape[0] == output_units * input_features:
                        weights = weights.reshape(output_units, input_features)
                        filter_shape = tuple(weights.shape)
                    else:
                        raise ValueError(f"Cannot infer 2D shape from 1D weights shape {filter_shape}")
                else:
                    raise ValueError(f"Unsupported filter shape: {filter_shape} (1D)")
            
            if len(filter_shape) != 2:
                raise ValueError(f"Unsupported filter shape: {filter_shape}")
            
            if not float_kernel and weights.dtype != np.int8:
                weights = weights.astype(np.int8)
            
            # TFLite format: [output_units, input_features]
            filter_dims = {
                'n': int(filter_shape[1]),  # input_features (col_dim)
                'h': 1,
                'w': 1,
                'c': int(filter_shape[0])   # output_units (row_dim)
            }
        else:
            # Fallback: descriptor format
            fs = tuple(self.desc['filter_shape'])
            if len(fs) != 2:
                raise ValueError(f"Unsupported filter_shape in descriptor: {fs}")
            filter_dims = {
                'n': int(fs[1]),  # input_features
                'h': 1,
                'w': 1,
                'c': int(fs[0])   # output_units
            }
        
        builder = TemplateContextBuilder()
        
        # Compute input dimensions
        if len(input_shape) == 2:
            input_dims = {
                'n': int(input_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(input_shape[1])
            }
        elif len(input_shape) == 4:
            # Flatten: [batch, h, w, c] -> features = h * w * c
            input_dims = {
                'n': int(input_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(input_shape[1] * input_shape[2] * input_shape[3])
            }
        else:
            input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        
        # Compute output dimensions - use weights shape to get correct output_units
        if weights is not None and len(weights.shape) == 2:
            correct_output_units = int(weights.shape[0])
            batch_size = int(output_shape[0]) if len(output_shape) >= 1 else int(input_shape[0])

            if len(output_shape) == 2 and output_shape[1] != correct_output_units:
                # The converter/LiteRT-reported output shape disagrees with the
                # weight tensor's output-unit dimension. Silently rewriting
                # output_dims from the weight shape would hide a real
                # converter/kernel contract error behind an auto-corrected
                # harness. Fail loudly instead.
                raise RuntimeError(
                    f"FullyConnected descriptor '{name}' has LiteRT "
                    f"output_shape[1] ({output_shape[1]}) that disagrees with "
                    f"weights.shape[0] ({correct_output_units}); refusing to "
                    "silently override output dims"
                )

            output_dims = {
                'n': batch_size,
                'h': 1,
                'w': 1,
                'c': correct_output_units
            }
        elif len(output_shape) == 2:
            output_dims = {
                'n': int(output_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(output_shape[1])
            }
        else:
            output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        float_kernel = kernel_info["input_c_type"] in {"float", "float16_t"}
        if float_kernel:
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            if weights is not None and weights.dtype != float_dtype:
                weights = weights.astype(float_dtype)
            has_biases = biases is not None and biases.size > 0
            if has_biases and biases.dtype != float_dtype:
                biases = biases.astype(float_dtype)

            fc_params = builder.build_fc_params(
                self.desc,
                input_quant,
                weight_quant_dict,
                output_quant,
            )

            input_data = np.asarray(self._sample_uniform(input_shape), dtype=float_dtype)
            from helia_core_tester.generation.utils.litert_utils import run_inference_litert
            interpreter_input_dtype = self.load_litert_interpreter(str(tflite_path)).get_input_details()[0]['dtype']
            def float_reference(operands, _dtype=float_dtype, _in_dtype=interpreter_input_dtype):
                return np.asarray(
                    run_inference_litert(
                        str(tflite_path),
                        operands[0].astype(_in_dtype),
                        subgraph_index=0,
                    ),
                    dtype=_dtype,
                )

            output_data = float_reference([input_data])
            output_data, nonfinite_context = self.apply_nonfinite_policy(
                output_data, reference=float_reference, inputs=[input_data]
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
