"""Convolve operation implementation."""

from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
import tensorflow as tf
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.fixed_batch import converter_for_batched_model
from helia_core_tester.generation.ops._shared.bias_init import (
    HoistedBiasInjectionError,
    SignedMagnitudeUniform,
    bias_is_hoisted_by_lowering,
    inject_hoisted_dilation_bias,
)
from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, GuardedBuffer, Provider
from helia_core_tester.generation.harness.faults import common_fault, null_context_buffer, struct_copy, with_fault
from helia_core_tester.generation.entry import check_entry_fault, resolve_entry
from helia_core_tester.generation.kernel_dispatch import resolve_convolve_kernel



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
    )

    def _hint(self) -> Dict[str, Any]:
        hint = self.desc.get("hint", {})
        return hint if isinstance(hint, dict) else {}

    def _check_fault_reachable(self, kind: str, context: Dict[str, Any]) -> None:
        kernel_fn = context["kernel_fn"]
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

    def needs_keras_model(self) -> bool:
        return str(self.desc.get("weight_dtype", "S8")).upper() != "S4"
    
    def build_keras_model(self) -> tf.keras.Model:
        input_shape = self.desc['input_shape']
        filter_shape = self.desc['filter_shape']
        groups = int(self.desc.get('groups', 1))
        
        tf.keras.utils.set_random_seed(17)
        
        padding = self.desc.get('padding', 'valid')
        if padding is not None:
            padding = str(padding).lower()
        else:
            padding = 'valid'
        
        activation = self.desc.get('activation', 'NONE')
        act = None if activation in (None, 'NONE', 'none') else activation.lower()
        
        dilation = self.desc.get('dilation', [1, 1])
        if isinstance(dilation, (int, float)):
            dilation = [int(dilation), int(dilation)]
        elif isinstance(dilation, (list, tuple)):
            if len(dilation) != 2:
                raise ValueError(f"Invalid dilation: {dilation}. Must be 2 integers or a single integer")
            dilation = [int(dilation[0]), int(dilation[1])]
        
        if any(d <= 0 for d in dilation):
            raise ValueError(f"Invalid dilation values: {dilation}. Must be positive integers")

        if len(input_shape) != 4:
            raise ValueError(f"Convolve input_shape must be NHWC rank-4, got: {input_shape}")
        if len(filter_shape) != 4:
            raise ValueError(f"Convolve filter_shape must be HWIO rank-4, got: {filter_shape}")
        if groups <= 0:
            raise ValueError(f"Convolve groups must be > 0, got: {groups}")

        input_channels = int(input_shape[3])
        output_filters = int(filter_shape[3])
        if input_channels % groups != 0:
            raise ValueError(
                f"Convolve input channels ({input_channels}) must be divisible by groups ({groups})"
            )
        if output_filters % groups != 0:
            raise ValueError(
                f"Convolve output filters ({output_filters}) must be divisible by groups ({groups})"
            )
        
        x = tf.keras.Input(
            shape=input_shape[1:],
            batch_size=input_shape[0] if len(input_shape) > 0 else None,
            dtype=tf.float32,
            name='input'
        )
        
        use_bias = self.desc.get('use_bias', True)
        # A zero bias_initializer produces an all-zero bias tensor, which the
        # TFLite converter's constant-folding optimizer strips from the graph
        # entirely -- the generated CMSIS-NN test then calls the kernel with a
        # NULL bias pointer, leaving the bias-add path completely untested.
        # Use a nonzero uniform bias, deterministic from the case seed, so
        # real bias data flows through the golden and the kernel call.
        #
        # The quantized magnitude has to clear one output quantization step,
        # or a dropped bias-add still reproduces the golden bit for bit. The
        # int suite calibrates from inputs in [-32, 32) and the bias itself
        # widens the calibrated output range by 2*maxval, so the floor is set
        # against the widest range a case can produce: every bias-carrying
        # channel clears at least one output step with margin, the bias
        # spends only a fraction of the dynamic range at the narrowest range,
        # and it stays orders of magnitude below the int32 accumulator's
        # headroom. The float range is unchanged.
        #
        # A quantized dilated conv keeps a zero Keras bias: it lowers to
        # SpaceToBatchND -> Conv2D -> BatchToSpaceND (-> Add), and a bias set
        # here would land in a trailing Add in the output quantization domain
        # instead of in the CONV_2D the kernel stands in for. Those cases get
        # their bias written into the CONV_2D placeholder after conversion by
        # inject_hoisted_dilation_bias, ahead of the golden run.
        _case_is_float = str(self.tensor_dtype("input", default="S8")).upper() in {"FP32", "FP16"}
        # Fixed-batch conversion retains native dilation and its bias operand.
        _bias_hoisted_by_lowering = (
            input_shape[0] == 1
            and bias_is_hoisted_by_lowering(dilation, is_float=_case_is_float)
        )
        if not use_bias or _bias_hoisted_by_lowering:
            bias_initializer = 'zeros'
        elif _case_is_float:
            bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.25, maxval=0.25, seed=self.seed)
        else:
            bias_initializer = SignedMagnitudeUniform(minval=2.0, maxval=4.0, seed=self.seed)

        conv = tf.keras.layers.Conv2D(
            filters=output_filters,
            kernel_size=tuple(filter_shape[0:2]),
            strides=tuple(self.desc.get('strides', [1, 1])),
            dilation_rate=tuple(dilation),
            padding=padding,
            groups=groups,
            use_bias=use_bias,
            activation=act,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1234),
            bias_initializer=bias_initializer,
            name='conv_2d'
        )(x)
        
        model = tf.keras.Model(inputs=[x], outputs=conv, name='conv_2d')
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        weight_dtype = str(self.desc.get("weight_dtype", "S8")).upper()
        if weight_dtype == "S4":
            from helia_core_tester.generation.utils.litert_builder import build_conv2d_s4_op

            extras = self.desc.get("hint", {}).get("extras", {})
            input_scale = float(extras.get("input_scale", 4.0))
            input_zp = int(extras.get("input_zero_point", 3))
            weight_scale = extras.get("weight_scale", 1.0)
            output_scale = float(extras.get("output_scale", 4.0))
            output_zp = int(extras.get("output_zero_point", 0))

            input_shape = tuple(self.desc["input_shape"])
            fs = tuple(self.desc["filter_shape"])  # H, W, I, O
            filter_shape = (fs[3], fs[0], fs[1], fs[2])  # O, H, W, I

            out_ch = filter_shape[0]
            per_channel = bool(extras.get("per_channel", True))
            if per_channel:
                if isinstance(weight_scale, (list, tuple, np.ndarray)):
                    weight_scales = list(float(v) for v in weight_scale)
                else:
                    weight_scales = [float(weight_scale)] * out_ch
            else:
                weight_scales = [float(weight_scale)]

            weight_zps = [0] * len(weight_scales)

            weights_int4 = self.rng.integers(-8, 8, size=filter_shape).astype(np.int8)
            biases = None
            if self.desc.get("use_bias", True):
                biases = self.rng.integers(-128, 128, size=(out_ch,), dtype=np.int32)

            tflite_model = build_conv2d_s4_op(
                input_shape=input_shape,
                filter_shape=filter_shape,
                strides=self.desc.get("strides", [1, 1]),
                padding=self.desc.get("padding", "valid"),
                dilation=self.desc.get("dilation", [1, 1]),
                use_bias=self.desc.get("use_bias", True),
                input_quant=([input_scale], [input_zp]),
                weight_quant=(weight_scales, weight_zps),
                output_quant=([output_scale], [output_zp]),
                weights_int4=weights_int4,
                biases=biases,
            )
            with open(out_path, "wb") as f:
                f.write(tflite_model)
            return

        converter = converter_for_batched_model(model, [self.desc['input_shape']])
        
        activation_dtype = str(self.desc.get('activation_dtype', 'S8')).upper()
        
        if activation_dtype == 'S8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif activation_dtype == 'S16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        elif activation_dtype == 'FP16':
            converter.optimizations = []
            converter.target_spec.supported_types = [tf.float16]
        elif activation_dtype == 'FP32':
            converter.optimizations = []
        
        def representative_data_gen():
            rep_rng = np.random.default_rng(42)
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = rep_rng.integers(-32, 32, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = rep_rng.integers(-32, 32, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = rep_rng.integers(-32, 32, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)

        hoisted = bias_is_hoisted_by_lowering(
            self.desc.get('dilation', [1, 1]),
            is_float=str(self.tensor_dtype("input", default="S8")).upper() in {"FP32", "FP16"},
        )
        if self.desc.get('use_bias', True) and hoisted and self.desc['input_shape'][0] == 1:
            try:
                inject_hoisted_dilation_bias(out_path, rep_seed)
            except HoistedBiasInjectionError as exc:
                raise ValueError(
                    f"{self.desc.get('name')}: no accumulator-scale bias was injected "
                    f"into the lowered dilated conv, so the case cannot detect a "
                    f"dropped bias-add: {exc}"
                ) from exc

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
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

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

        # Load LiteRT model for tensor extraction
        from helia_core_tester.generation.utils.litert_utils import (
            load_litert_model,
            get_operator_tensors_from_litert,
            get_tensor_shape_from_litert,
            get_tensor_quantization_from_litert,
        )
        
        model, subgraph = load_litert_model(str(tflite_path))
        
        # Get operator tensors from the actual CONV_2D op.
        # Dilated graphs can be lowered as SpaceToBatch -> Conv2D -> BatchToSpace,
        # so selecting op index 0 can bind to the wrong tensors.
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")
        
        conv_op_index = 0
        bts_op_index = None
        hoisted_bias_data = None
        try:
            from ai_edge_litert import schema_py_generated as litert

            found_conv = False
            for i, op in enumerate(subgraph.operators):
                opcode = model.operatorCodes[op.opcodeIndex]
                if not found_conv and opcode.builtinCode == litert.BuiltinOperator.CONV_2D:
                    conv_op_index = i
                    found_conv = True
                if opcode.builtinCode == litert.BuiltinOperator.BATCH_TO_SPACE_ND:
                    bts_op_index = i

            # Dilated graphs lower to SpaceToBatchND -> Conv2D -> BatchToSpaceND.
            # TF's MLIR converter hoists the bias-add out of CONV_2D in this
            # pattern: the CONV_2D op keeps only a zero-filled placeholder
            # bias, and the real bias is applied via a separate ADD op after
            # BatchToSpaceND. If we naively pull "biases" from the CONV_2D
            # op's own inputs (as done above via get_operator_tensors_from_litert),
            # we get that zero placeholder instead of the bias the golden
            # output was actually computed with. Find the hoisted bias, if
            # this pattern is present, so the CMSIS kernel is called with the
            # same bias the golden reflects.
            #
            # Scoped to float kernels only: the same ADD-after-BatchToSpaceND
            # shape exists in quantized (S8/S16) graphs too, but there the
            # hoisted ADD operates in the quantized output domain (its
            # operand is not a plain int32 accumulator bias), so reusing this
            # extraction for quantized dtypes would silently substitute the
            # wrong tensor. Those cases instead get an accumulator-scale bias
            # written into the placeholder at conversion time by
            # inject_hoisted_dilation_bias, and the placeholder read above is
            # then the right tensor.
            if float_kernel and bts_op_index is not None:
                from helia_core_tester.generation.utils.litert_utils import get_tensor_data_from_litert

                bts_outs = subgraph.operators[bts_op_index].outputs
                bts_output_idx = int(bts_outs[0]) if bts_outs is not None and len(bts_outs) > 0 else None
                if bts_output_idx is not None:
                    for i in range(bts_op_index + 1, len(subgraph.operators)):
                        op = subgraph.operators[i]
                        opcode = model.operatorCodes[op.opcodeIndex]
                        if opcode.builtinCode != litert.BuiltinOperator.ADD:
                            continue
                        if bts_output_idx not in list(op.inputs):
                            continue
                        for input_idx in op.inputs:
                            if int(input_idx) == bts_output_idx:
                                continue
                            candidate = subgraph.tensors[int(input_idx)]
                            candidate_data = get_tensor_data_from_litert(candidate, model)
                            # Adversarial-review hardening: the quantized graph's
                            # hoisted ADD operand clears the opcode/constness/ndim
                            # gates too (dtype int16, quantized-output domain) --
                            # only dtype and exact per-channel length make this
                            # extraction safe. A broadcast scalar (shape (1,))
                            # would otherwise be emitted and read out of bounds
                            # by the kernel (output_dims.c elements).
                            if (
                                candidate_data is not None
                                and candidate_data.ndim == 1
                                and candidate_data.dtype.kind == "f"
                                and candidate_data.shape[0] == int(self.desc["filter_shape"][3])
                            ):
                                hoisted_bias_data = candidate_data
                                break
                        break
        except Exception:
            conv_op_index = 0
            bts_op_index = None
            hoisted_bias_data = None

        op_tensors = get_operator_tensors_from_litert(model, subgraph, conv_op_index)
        
        # Extract shapes from LiteRT
        if not op_tensors['inputs']:
            raise ValueError("No input tensors found")
        if not op_tensors['outputs']:
            raise ValueError("No output tensors found")
        
        input_tensor = None
        output_tensor = None
        if subgraph.inputs is not None and len(subgraph.inputs) > 0:
            input_tensor = subgraph.tensors[int(subgraph.inputs[0])]
        if subgraph.outputs is not None and len(subgraph.outputs) > 0:
            output_tensor = subgraph.tensors[int(subgraph.outputs[0])]

        expected_tensor = None
        if bts_op_index is not None:
            bts_outs = subgraph.operators[bts_op_index].outputs
            if bts_outs is not None and len(bts_outs) > 0:
                expected_tensor = subgraph.tensors[int(bts_outs[0])]

        input_shape = get_tensor_shape_from_litert(input_tensor) if input_tensor is not None else None
        output_shape = (
            get_tensor_shape_from_litert(expected_tensor)
            if expected_tensor is not None
            else (get_tensor_shape_from_litert(output_tensor) if output_tensor is not None else None)
        )
        if input_shape is None:
            input_shape = op_tensors['inputs'][0]['shape']
        if output_shape is None:
            output_shape = op_tensors['outputs'][0]['shape']

        input_shape = tuple(input_shape)
        output_shape = tuple(output_shape)
        
        # Extract quantization parameters from LiteRT
        if expected_tensor is not None:
            input_quant = (
                get_tensor_quantization_from_litert(input_tensor)
                if input_tensor is not None
                else op_tensors['inputs'][0]['quantization']
            )
            output_quant = get_tensor_quantization_from_litert(expected_tensor)
        elif input_tensor is not None and output_tensor is not None:
            input_quant = get_tensor_quantization_from_litert(input_tensor)
            output_quant = get_tensor_quantization_from_litert(output_tensor)
        else:
            input_quant = op_tensors['inputs'][0]['quantization']
            output_quant = op_tensors['outputs'][0]['quantization']
        
        # Find weight quantization (from weight tensor in inputs)
        weight_quant = None
        for input_tensor_info in op_tensors['inputs']:
            if input_tensor_info['data'] is not None and len(input_tensor_info['shape']) > 1:
                weight_quant = input_tensor_info['quantization']
                break
        
        quant_params = {
            'input': input_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False},
            'output': output_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False},
            'weight': weight_quant or input_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False}
        }
        
        # Extract weights and biases from LiteRT
        weights = op_tensors['weights']
        biases = op_tensors['biases']
        if hoisted_bias_data is not None:
            # See the SpaceToBatchND/BatchToSpaceND comment above: this
            # dilated-conv graph applies its real bias via a post-BatchToSpace
            # ADD op, not inside CONV_2D. Use that bias instead of the zero
            # placeholder CONV_2D carries, so the kernel call matches the
            # golden's bias.
            biases = hoisted_bias_data
        weight_dtype = str(self.desc.get("weight_dtype", "S8")).upper()
        if weight_dtype == "S4":
            from helia_core_tester.generation.utils.litert_utils import get_tensor_data_packed_from_litert
            packed_weights = None
            for input_tensor_info in op_tensors['inputs']:
                if input_tensor_info['data'] is not None and len(input_tensor_info['shape']) > 1:
                    packed_weights = get_tensor_data_packed_from_litert(input_tensor_info['tensor'], model)
                    break
            if packed_weights is None:
                raise ValueError("Packed S4 weights not found in LiteRT model")
            weights = packed_weights.astype(np.int8)

        # Weight tensor for TFLite Conv2D is OHWI in practice; shape will be (O, H, W, I)
        if weight_dtype == "S4":
            fs = tuple(self.desc['filter_shape'])
            filter_shape = (fs[3], fs[0], fs[1], fs[2])  # OHWI
        elif weights is not None:
            filter_shape = tuple(weights.shape)
            if not float_kernel and weights.dtype != np.int8:
                weights = weights.astype(np.int8)
        else:
            # Fallback: descriptor is HWIO (kh, kw, in, out)
            fs = tuple(self.desc['filter_shape'])
            filter_shape = (fs[3], fs[0], fs[1], fs[2])  # OHWI

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        # CMSIS expects OHWI dims
        # For grouped convolutions, CMSIS calculates groups = input_ch / filter_ch
        # TFLite stores filters with input_ch channels (all groups), but CMSIS expects input_ch/groups
        # So we need to adjust filter_dims.c for grouped convolutions
        groups = self.desc.get('groups', 1)
        filter_dims = {
            'n': int(filter_shape[0]),
            'h': int(filter_shape[1]),
            'w': int(filter_shape[2]),
            'c': int(filter_shape[3]),  # Use full input channels, NOT divided by groups
        }

        # Correct kernel size for padding math
        kernel_hw = (filter_dims['h'], filter_dims['w'])

        # Build convolution parameters (fix SAME padding + offsets)
        conv_params = builder.build_conv_params(
            self.desc,
            input_shape,
            kernel_hw,
            output_shape,
            quant_params['input'],
            quant_params['output']
        )

        quant_params_dict = None
        if not float_kernel:
            # Build quantization parameters
            # CRITICAL: The effective scale for multiplier/shift is NOT output_scale directly!
            # It should be: effective_scale = (input_scale * weight_scale) / output_scale
            # This matches CMSIS-NN test code in op_utils.py line 194
            input_quant = quant_params['input']
            output_quant = quant_params['output']
            weight_quant = quant_params.get('weight', output_quant)
            
            input_scale = input_quant.get('scale', 1.0)
            if isinstance(input_scale, (list, np.ndarray)):
                input_scale = float(input_scale[0])
            else:
                input_scale = float(input_scale)

            output_scale = output_quant.get('scale', 1.0)
            if isinstance(output_scale, (list, np.ndarray)):
                output_scale = float(output_scale[0])
            else:
                output_scale = float(output_scale)

            weight_scale = weight_quant.get('scale', 1.0)
            per_channel = bool(weight_quant.get('per_channel', False))
            
            # Calculate effective scales: (input_scale * weight_scale) / output_scale
            if per_channel and isinstance(weight_scale, np.ndarray):
                # In double precision, as TFLite computes it; the weight scales are float32.
                effective_scales = (input_scale * weight_scale.astype(np.float64)) / output_scale
                effective_quant = {
                    'scale': effective_scales,
                    'zero_point': output_quant.get('zero_point', 0),
                    'per_channel': True
                }
                quant_params_dict = builder.build_quant_params(effective_quant, per_channel=True)
                quant_params_dict['effective_scales'] = effective_scales
                from helia_core_tester.generation.utils.tflite_utils import calculate_per_channel_multiplier_shift
                multipliers_raw, shifts_raw = calculate_per_channel_multiplier_shift(effective_scales)
                quant_params_dict['multiplier_array_raw'] = multipliers_raw
                quant_params_dict['shift_array_raw'] = shifts_raw
            else:
                if isinstance(weight_scale, (list, np.ndarray)):
                    weight_scale = float(weight_scale[0])
                else:
                    weight_scale = float(weight_scale)
                effective_scale = float((input_scale * weight_scale) / output_scale)
                effective_quant = {
                    'scale': effective_scale,
                    'zero_point': output_quant.get('zero_point', 0),
                    'per_channel': False
                }
                quant_params_dict = builder.build_quant_params(effective_quant, per_channel=False)
                quant_params_dict['effective_scale'] = effective_scale
                from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
                multiplier_raw, shift_raw = calculate_multiplier_shift(effective_scale)
                quant_params_dict['multiplier_raw'] = multiplier_raw
                quant_params_dict['shift_raw'] = shift_raw
            
            quant_params_dict['per_channel'] = per_channel

        # Generate input data and quantize to the interpreter's real input dtype
        # IMPORTANT: Reset RNG to seed to ensure input data matches what was used
        # during TFLite conversion (representative dataset generation may have advanced RNG)
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        if float_kernel:
            input_data = self._sample_uniform(input_shape)
        else:
            input_data = self.generate_input_data()
        self.rng.__setstate__(rng_state)
        if float_kernel:
            input_q = np.asarray(input_data, dtype=float_dtype)
            interpreter_input_dtype = self.load_litert_interpreter(str(tflite_path)).get_input_details()[0]['dtype']

            def float_reference(operands, _dtype=float_dtype, _in_dtype=interpreter_input_dtype):
                return self.run_inference(
                    str(tflite_path), operands[0].astype(_in_dtype)
                ).astype(_dtype)

            output_data = float_reference([input_q])
        else:
            input_scale = float(self._quant_param_scalar(quant_params['input'], 'scale', 1.0))
            input_zp = int(self._quant_param_scalar(quant_params['input'], 'zero_point', 0))

            if kernel_info["input_c_type"] == "int8_t":
                qmin, qmax = -128, 127
                np_in_dtype = np.int8
            elif kernel_info["input_c_type"] == "int16_t":
                qmin, qmax = -32768, 32767
                np_in_dtype = np.int16
            else:
                raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")

            input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
            input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)

            # Run inference (dtype must match interpreter input)
            output_data = self.run_inference(str(tflite_path), input_q)

        # Bias handling (S16 wrapper expects int64 bias)
        has_biases = biases is not None and getattr(biases, "size", 0) > 0
        bias_dtype = kernel_info["bias_c_type"]
        if has_biases:
            if float_kernel and biases.dtype != float_dtype:
                biases = biases.astype(float_dtype)
            elif bias_dtype == "int64_t" and biases.dtype != np.int64:
                biases = biases.astype(np.int64)
            elif bias_dtype == "int32_t" and biases.dtype != np.int32:
                biases = biases.astype(np.int32)
        if float_kernel and weights is not None and weights.dtype != float_dtype:
            weights = weights.astype(float_dtype)

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
                output_data, reference=float_reference, inputs=[input_q]
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
            'expected_status': self.expected_status(),
            # The entry lives only on ns-cmsis-nn's MVE integer paths, so it declines on a build
            # that compiles them out (HELIA_CMSIS_NN_INT_AUTOVECTORIZE, set by CMakeLists.txt).
            'autovectorize_declines': bool(self.desc.get("autovectorize_declines", False)),
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
